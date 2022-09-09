"section5: Q-learning with parametric function approximators"

import pickle
from typing import List, Tuple
import uuid
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from section1 import get_random_initial_state, trajectory_yield
from section2 import get_expected_policy_return, plot_expected_policy_return
from section3 import save_from_policy
from section4 import (
    CountStopCriterion,
    extra_tree_generator,
    extract_policy,
    fitted_q,
    merge_datasets,
    random_trainset_near_terminal,
    random_trajectory_trainset,
    random_uniform_trainset,
    save_figs,
)

cpu_device = torch.device("cpu")
# GPU is slower : we can't benefit from batch processing...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = cpu_device

print(f"Using {device} device")


class TwoHeadsModule(nn.Module):
    def __init__(self):
        super(TwoHeadsModule, self).__init__()
        self.pos_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )
        self.neg_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh(),
        )

    def q_function_batched(self, input: torch.Tensor):
        "input: torch.Tensor of shape [N, 3]"

        return torch.where(
            (input[:, 2] == 4).view((-1, 1)),
            self.pos_head(input[:, :2]),
            self.neg_head(input[:, :2]),
        )

    def prediction_batched(self, input: torch.Tensor) -> torch.Tensor:
        "input: torch.Tensor of shape [N, 3]"

        pos = self.pos_head(input[:, :2])
        neg = self.neg_head(input[:, :2])

        return 8 * (pos > neg) - 4

    def predict(self, psu: np.ndarray) -> np.ndarray:
        "numpy compatible predict function"
        psu_t = torch.from_numpy(psu).float()

        return self.q_function_batched(psu_t).numpy()

    def prediction_nonbatched(self, input: torch.Tensor) -> torch.Tensor:
        "input: torch.Tensor of shape [N, 3]"

        pos = self.pos_head(input[:2])
        neg = self.neg_head(input[:2])

        return 8 * (pos > neg) - 4

    def policy_func(self, p: np.ndarray, s: np.ndarray):
        # convert to torch.Tensor
        input_t = torch.Tensor(np.stack([p, s], axis=-1))

        # predict
        prediction = self.prediction_batched(input_t)

        # convert back to numpy
        return prediction[:, 0].numpy()

    def get_policy_e_greedy_fun(self, epsilon: float):
        # with epsilon probability, sample at random
        # with 1-epsilon probability, use the best learnt policy

        def _policy(p: np.ndarray, s: np.ndarray):
            input_t = torch.Tensor(np.stack([p, s], axis=-1))
            predictions = self.prediction_nonbatched(input_t)
            predictions = predictions.view((-1,)).numpy()

            random_prediction = 8 * (np.random.random(predictions.shape) < 0.5) - 4
            random_mask = np.random.random(predictions.shape) < epsilon

            predictions[random_mask] = random_prediction[random_mask]

            # handle scalars
            if not np.empty_like(p).shape:
                return float(predictions)

            return predictions

        return _policy

    def value_function(self, p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # dim=-1 doesn't changed anything for scalar values (shape=[])
        input = torch.stack([p, s], dim=-1)

        pos_q = self.pos_head(input)
        neg_q = self.neg_head(input)

        return torch.maximum(pos_q, neg_q)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        "input: torch.Tensor of shape [3] -> []"

        if input[2] == 4:
            return self.pos_head(input[:2])
        return self.neg_head(input[:2])

    def train_greedy(
        self,
        n_epochs: int,
        sample_size: int,
        disc_factor: float,
        trajectory_len: int,
        epsilon: float,
        learning_rate_fn,
        print_every: int = 100,
    ):
        self.train()

        counter = 0

        for epoch in range(n_epochs):
            if callable(learning_rate_fn):
                learning_rate = learning_rate_fn(epoch)
            else:
                learning_rate = learning_rate_fn

            replay_buffer = [None] * trajectory_len
            max_update = 0.0

            for t, ((p, s), u, r, (n_p, n_s)) in enumerate(
                trajectory_yield(
                    get_random_initial_state(),
                    self.get_policy_e_greedy_fun(epsilon),
                    update_count=trajectory_len,
                    early_stop=True,
                )
            ):
                replay_buffer[t] = (
                    torch.from_numpy(np.stack([p, s, u], axis=-1)).float().to(device),
                    torch.from_numpy(r).float().to(device),
                    torch.from_numpy(n_p).float().to(device),
                    torch.from_numpy(n_s).float().to(device),
                )

                indices = torch.randint(t + 1, (sample_size,))

                for idx in indices:
                    sample = replay_buffer[idx]

                    max_update, stop = self._update_step(
                        *sample,
                        disc_factor,
                        learning_rate,
                        max_update,
                        (counter + 1) % print_every == 0,
                        f"{epoch=: 4d}/{n_epochs} {t=: 4d}/{trajectory_len}",
                        tol=-1,  # never stop based on max_update
                    )
                    counter += 1
                    if stop:
                        return

    def fit(
        self,
        ds_x,
        _,
        n_p,
        n_s,
        r,
        n_epochs: int,
        disc_factor: float,
        learning_rate_fn,
        print_every: int = 100,
    ):

        # build dataset for pytorch
        data_loader = DataLoader(
            TensorDataset(
                torch.Tensor(ds_x).to(device),
                torch.Tensor(n_p).to(device),
                torch.Tensor(n_s).to(device),
                torch.Tensor(r).to(device),
            ),
            batch_size=None,  # no minibatch
            shuffle=True,
        )
        trainset_size = len(data_loader)

        self.train()

        for epoch in range(n_epochs):
            if callable(learning_rate_fn):
                learning_rate = learning_rate_fn(epoch)
            else:
                learning_rate = learning_rate_fn
            max_update = 0.0

            for b_idx, (psu, n_p, n_s, r_) in enumerate(data_loader):

                state_prefix = f"{epoch=: 4d}/{n_epochs} {b_idx=: 6d}/{trainset_size}"

                max_update, stop = self._update_step(
                    psu,
                    r_,
                    n_p,
                    n_s,
                    disc_factor,
                    learning_rate,
                    max_update,
                    (b_idx + 1) % print_every == 0,
                    state_prefix,
                )
                if stop:
                    return

    def _update_step(
        self,
        psu,
        r,
        n_p,
        n_s,
        disc_factor,
        learning_rate,
        max_update: float,
        print_condition: bool,
        print_prefix: str,
        tol: float = 1e-5,
        norm_type=None,
    ) -> Tuple[float, bool]:

        q_value = self(psu)

        q_value.backward()

        # update
        with torch.no_grad():
            # compute delta
            next_value = self.value_function(n_p, n_s)
            delta = r + disc_factor * next_value - q_value

            if norm_type is not None:
                norm = torch.stack(
                    [
                        torch.flatten(param.grad)
                        for param in self.parameters()
                        if param.grad is not None
                    ]
                )
                norm = torch.norm(norm, float("inf"))
            else:
                norm = 1

            for param in self.parameters():
                # NOTE some grad WILL be None, as the whole path is not done
                # for every input, because the two heads are not necessary
                if param.grad is None:
                    continue

                update_term = learning_rate * delta * param.grad / norm
                param.add_(update_term)

                max_update = max(max_update, float(update_term.abs().max()))

            self.zero_grad()

            if max_update < tol:
                print(f"stopping after {print_prefix} " f"{max_update=}")
                return -1.0, True

            if print_condition:
                if next_value.abs().max() > 1:
                    print(f"{next_value=} may indicated explosion")

                print(f"{print_prefix} update: {max_update=:.3E}")
                max_update *= 0.1

                mean, _ = get_expected_policy_return(50, 20, self.policy_func, 0.95)
                print(f"{print_prefix} ECR: {mean[-1]= :+.2f}")
                # if mean[-1] > 0.3:
                #     return max_update, True

        return max_update, False


def train_offline(dest: str):
    model = TwoHeadsModule().to(device)

    model.fit(
        *merge_datasets(
            random_trajectory_trainset(200, 100),
            random_trainset_near_terminal(600, 400),
            random_uniform_trainset(500),
        ),
        n_epochs=15,
        disc_factor=0.95,
        learning_rate_fn=lambda e: 0.1 * 0.9**e,
        print_every=100,
    )

    # now: save model to disk
    torch.save(model.state_dict(), dest)

    print(f"model saved at {dest}")


def train_online(dest: str, n_epochs: int = 1000):
    model = TwoHeadsModule().float()
    model.train_greedy(
        n_epochs=n_epochs,
        sample_size=10,
        disc_factor=0.95,
        trajectory_len=200,
        learning_rate_fn=0.1,
        epsilon=0.25,
        print_every=100,
    )

    if dest:
        # now: save model to disk
        torch.save(model.state_dict(), dest)
        print(f"model saved at {dest}")

    return model


def experiment(epoch_trials: List[int], repeat: int = 10):
    if not epoch_trials:
        raise ValueError("empty list")
    ecr_dict = {}
    suffix = "-".join(str(e) for e in epoch_trials)

    for n_epochs in epoch_trials:
        ecr_dict[n_epochs] = np.array(
            [
                get_expected_policy_return(
                    200, 100, train_online(None, n_epochs).policy_func, 0.95
                )[0][-1]
                for _ in range(repeat)
            ]
        )

    for n_epochs in epoch_trials:
        mean = np.mean(ecr_dict[n_epochs])
        std = np.std(ecr_dict[n_epochs])
        print(f"{n_epochs=} {mean=} {std=}")

    with open(f"ecr_dump_{suffix}.pkl", mode="wb") as f:
        pickle.dump(ecr_dict, f)


def comparison_et(epoch_trials: List[int], repeat: int = 10):
    if not epoch_trials:
        raise ValueError("empty list")
    ecr_dict = {}
    suffix = "-".join(str(e) for e in epoch_trials)

    for n_epochs in epoch_trials:
        ecr_dict[n_epochs] = np.array(
            [
                get_expected_policy_return(
                    200,
                    100,
                    extract_policy(
                        fitted_q(
                            n_iteration=200,
                            dataset=random_trajectory_trainset(n_epochs, 200),
                            model_generator=extra_tree_generator,
                            criterion=CountStopCriterion(200),
                            disc_factor=0.95,
                            print_y_diff=False,
                        )
                    ),
                    0.95,
                )[0][-1]
                for _ in range(repeat)
            ]
        )

    for n_epochs in epoch_trials:
        mean = np.mean(ecr_dict[n_epochs])
        std = np.std(ecr_dict[n_epochs])
        print(f"{n_epochs=} {mean=} {std=}")

    with open(f"et_ecr_dump_{suffix}.pkl", mode="wb") as f:
        pickle.dump(ecr_dict, f)


if __name__ == "__main__":
    # experiment([int(a) for a in sys.argv[1:]])
    # comparison_et([int(a) for a in sys.argv[1:]])

    model_path = f"model_{uuid.uuid4()}.pth"

    if len(sys.argv) == 1:
        train_online(model_path)

    if len(sys.argv) == 2:
        model_path = sys.argv[1]

    print(f"laoding model at {model_path}")

    # load from disk (to cpu)
    model = TwoHeadsModule()
    model.load_state_dict(torch.load(model_path, map_location=cpu_device))

    prefix = "param_QLearning"

    with torch.no_grad():

        # expected cumulative return
        mean, std = get_expected_policy_return(200, 50, model.policy_func, 0.95)
        print(f"{mean[-3:]=} {std[-3:]=}")

        plot_expected_policy_return(
            200, 100, model.policy_func, 0.95, dest=f"{prefix}_ecr.png", show=False
        )

        save_figs(model, prefix=prefix, type="larger")

        def _policy(p: np.ndarray, s: np.ndarray):
            input_t = torch.Tensor(np.stack([p, s], axis=-1))
            predictions = model.prediction_nonbatched(input_t)
            predictions = predictions.view((-1,)).numpy()

            # handle scalars
            if not np.empty_like(p).shape:
                return float(predictions)

            return predictions

        try:
            save_from_policy(_policy, prefix, length=50)
        except:
            pass  # will fail if no video device is found (e.g. ssh connections)

        for (p, s), u, r, (n_p, n_s) in trajectory_yield(
            get_random_initial_state(),
            _policy,
            update_count=45,
        ):
            print(
                f"x_t=({p:+.3f}, {s:+.3f}), {u=:+.1f}, {r=:+.1f} x_t+1=({n_p:+.3f}, {n_s:+.3f})"
            )

    import matplotlib.pyplot as plt

    x = np.array([1, 5, 10, 50, 100])

    et = np.array([-0.17, 0.04, 0.06, 0.10, 0.25])
    nn = np.array([-0.53, -0.44, -0.21, -0.12, -0.11])

    plt.plot(x, et, label="Fit. Q-Ite. ExtraTrees")
    plt.plot(x, nn, label="Q-Learning NN")

    plt.ylabel("ECR")
    plt.xlabel("Number of epochs")

    plt.legend()

    plt.show()
