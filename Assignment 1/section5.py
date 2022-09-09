import matplotlib.pyplot as plt

from section4 import *


_INFINITY_NORM = "||·||∞"
_L2_NORM = "||·||2"

disc_factor = 0.99


def update_q_table(q: np.ndarray, sample, alpha):
    (x, y), action, reward, (next_x, next_y) = sample
    action_idx = action_space.index(action)

    q[action_idx, x, y] = (1 - alpha) * q[action_idx, x, y] + alpha * (
        reward + disc_factor * q[:, next_x, next_y].max()
    )


def get_true_j(noise_distribution: np.ndarray) -> np.ndarray:
    """"""
    true_r = get_true_reward(noise_distribution)
    true_p = get_true_prob(noise_distribution)
    true_q = get_q_table(true_r, true_p, N=1508, discount_factor=disc_factor)
    return true_q.max(axis=0)


def setup_norm_dict(norms):
    if not norms:
        return {}

    # single norm
    if norms in [_INFINITY_NORM, _L2_NORM]:
        return {norms: []}

    for norm in norms:
        if norm not in [_INFINITY_NORM, _L2_NORM]:
            raise ValueError(f"bad {norm = }")

    return {norm: [] for norm in norms}


def update_norm_dict(norm_dict, true_j, learned_q):
    if not norm_dict:
        return

    diff = (true_j - learned_q.max(axis=0)).flatten()

    for norm, lst in norm_dict.items():
        if norm == _INFINITY_NORM:
            lst.append(np.linalg.norm(diff, np.inf))
        elif norm == _L2_NORM:
            lst.append(np.linalg.norm(diff, 2))
        else:
            raise ValueError(f"bad {norm = }")


class Agent:
    def __init__(self, epsilon=0.5) -> None:
        self.reset(epsilon)

    def reset(self, epsilon: float = None):
        self.q = np.zeros((len(action_space), m, n))
        if epsilon is not None:
            self.epsilon = epsilon

    def policy(self, state):
        "epsilon greedy policy"
        if np.random.random() < self.epsilon:
            return random_policy(state)
        return action_space[self.q[(slice(None),) + state].argmax()]

    def softmax_policy(self, state):
        tau = 1000
        proba = np.exp(self.q[:, state[0], state[1]] / tau)
        proba = proba / np.sum(proba)
        return action_space[np.random.choice(len(action_space), p=proba)]

    def train_offline(
        self,
        *,
        n_transition: int,
        learning_rate: float,
        learning_rate_decay: float,
        noise_distribution: np.ndarray,
    ):
        alpha = learning_rate

        for t, sample in enumerate(
            gen_history(
                length=n_transition,
                policy=random_policy,
                noise_distribution=noise_distribution,
                initial_state=(3, 0),
            )
        ):
            update_q_table(self.q, sample, alpha)
            alpha *= learning_rate_decay

    def train_in_order(
        self,
        *,
        n_episode: int,
        n_transition: int,
        learning_rate: float,
        learning_rate_decay: float,
        noise_distribution: np.ndarray,
        save_norm: str = None,
        use_softmax_policy: bool = False,
    ):
        norm_dict = setup_norm_dict(save_norm)
        if norm_dict:
            true_j = get_true_j(noise_distribution)

        if use_softmax_policy:
            policy = self.softmax_policy
        else:
            policy = self.policy

        for episode in range(n_episode):
            alpha = learning_rate

            for t, sample in enumerate(
                gen_history(
                    length=n_transition,
                    policy=policy,
                    noise_distribution=noise_distribution,
                    initial_state=(3, 0),
                )
            ):
                update_q_table(self.q, sample, alpha)
                alpha *= learning_rate_decay

            if norm_dict:
                update_norm_dict(norm_dict, true_j, self.q)

        return norm_dict

    def experiment_1(
        self,
        noise_distribution: np.ndarray,
        n_episode=100,
        save_norm=None,
    ):
        return self.train_in_order(
            n_episode=n_episode,
            n_transition=1000,
            learning_rate=0.05,
            learning_rate_decay=1.0,
            noise_distribution=noise_distribution,
            save_norm=save_norm,
        )

    def experiment_2(
        self,
        noise_distribution: np.ndarray,
        n_episode=100,
        save_norm=None,
    ):
        return self.train_in_order(
            n_episode=n_episode,
            n_transition=1000,
            learning_rate=0.05,
            learning_rate_decay=0.8,
            noise_distribution=noise_distribution,
            save_norm=save_norm,
        )

    def train_with_replay_buffer(
        self,
        *,
        n_episode: int,
        n_transition: int,
        learning_rate: float,
        learning_rate_decay: float,
        noise_distribution: np.ndarray,
        buffer_n_sample: int,
        save_norm: str = None,
    ):
        norm_dict = setup_norm_dict(save_norm)
        if norm_dict:
            true_j = get_true_j(noise_distribution)

        for episode in range(n_episode):
            alpha = learning_rate
            replay_buffer = [None] * n_transition

            for t, sample in enumerate(
                gen_history(
                    length=n_transition,
                    policy=self.policy,
                    noise_distribution=noise_distribution,
                    initial_state=(3, 0),
                )
            ):
                replay_buffer[t] = sample
                random_indices = np.random.randint(t + 1, size=(buffer_n_sample,))

                for buffer_idx in random_indices:
                    update_q_table(
                        self.q,
                        replay_buffer[buffer_idx],
                        alpha,
                    )
                alpha *= learning_rate_decay

            if norm_dict:
                update_norm_dict(norm_dict, true_j, self.q)

        return norm_dict

    def experiment_3(
        self,
        noise_distribution: np.ndarray,
        n_episode=100,
        save_norm=None,
    ):
        return self.train_with_replay_buffer(
            n_episode=n_episode,
            n_transition=10000,
            learning_rate=0.05,
            learning_rate_decay=1.0,
            noise_distribution=noise_distribution,
            buffer_n_sample=10,
            save_norm=save_norm,
        )


def plot_norms(norm_dict, filename):

    fig, ax = plt.subplots()
    ax.set_title(f"Convergence speed for J_N ({settings} {disc_factor})")

    ax.set_xlabel("Number of completed episodes")

    lines = []

    if _INFINITY_NORM in norm_dict:
        if _L2_NORM in norm_dict:
            ax2 = ax.twinx()

        data = norm_dict[_INFINITY_NORM]
        ax.set_ylabel(_INFINITY_NORM)
        data_mean = data.mean(axis=0)
        ax.set_ylim((data_mean.min() - 1, data_mean.max() + 1))

        lines += ax.plot(
            range(data.shape[1]), data_mean, color="blue", label=_INFINITY_NORM
        )

        if data.shape[0] > 1:
            data_std = data.std(axis=0, ddof=1)
            data_low = data_mean - data_std
            data_high = data_mean + data_std

            ax.fill_between(
                range(data.shape[1]), data_low, data_high, color="blue", alpha=0.3
            )

        if _L2_NORM in norm_dict:
            ax = ax2

    if _L2_NORM in norm_dict:

        data = norm_dict[_L2_NORM]
        ax.set_ylabel(_L2_NORM)
        data_mean = data.mean(axis=0)

        lines += ax.plot(
            range(data.shape[1]), data_mean, color="orange", label=_L2_NORM
        )

        if data.shape[0] > 1:
            data_std = data.std(axis=0, ddof=1)
            data_low = data_mean - data_std
            data_high = data_mean + data_std

            ax.fill_between(
                range(data.shape[1]), data_low, data_high, color="orange", alpha=0.3
            )

    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    fig.tight_layout()
    fig.savefig(filename)


def print_results(agent: Agent, name: str):
    print(name)

    print("J^µ* = ")
    print(agent.q.max(axis=0))
    print("Optimal policy µ* = ")
    print(match_actions(agent.q.argmax(axis=0)))
    print("")


def main():

    print("True J^µ* = ")
    print(get_true_j(noise_distribution))

    n_episode = 100
    save_norm = [_INFINITY_NORM, _L2_NORM]
    repeat = 1 if IS_DETERMINISTIC else 10

    # 5.2
    agent = Agent()
    agent.train_offline(
        n_transition=h_len,
        learning_rate=0.05,
        learning_rate_decay=1.0,
        noise_distribution=noise_distribution,
    )
    print_results(agent, f"offline learning {settings}")

    def do(exp: "function", path: str, name: str):
        norm_dict = setup_norm_dict(save_norm)
        for key in norm_dict.keys():
            norm_dict[key] = np.zeros((0, n_episode))

        agent_ = Agent()
        for _ in range(repeat):
            agent_.reset()
            res = exp(agent_, noise_distribution, save_norm=save_norm)

            for key, val in norm_dict.items():
                norm_dict[key] = np.concatenate([val, [res[key]]])

        plot_norms(norm_dict, path)
        print_results(agent_, name)

    
    # 5.3 if disc_factor=0.99 or 5.4 if disc_factor=0.4
    do(
        Agent.experiment_1,
        f"experiment_1_{settings}_{disc_factor}.png",
        f"experiment 1 {settings}",
    )
    do(
        Agent.experiment_2,
        f"experiment_2_{settings}_{disc_factor}.png",
        f"experiment 2 {settings}",
    )
    do(
        Agent.experiment_3,
        f"experiment_3_{settings}_{disc_factor}.png",
        f"experiment 3 {settings}",
    )
    
    # 5.5
    def _q5_5(agent_: Agent, noise_distribution: np.ndarray, save_norm: str = None):
        return agent_.train_in_order(
            n_episode=n_episode,
            n_transition=1000,
            learning_rate=0.05,
            learning_rate_decay=1.0,
            noise_distribution=noise_distribution,
            save_norm=save_norm,
            use_softmax_policy=True,
        )
    do(
        _q5_5,
        f"q5_5_{settings}_{disc_factor}.png",
        f"q5_5 {settings}",
    )


if __name__ == "__main__":
    IS_DETERMINISTIC = True
    if IS_DETERMINISTIC:
        h_len = int(2e6)
        noise_distribution = np.array([[1.0, 0.0]])
        settings = "deterministic"
    else:
        h_len = int(2e6)
        noise_distribution = np.array([[0.5, 0.0], [0.5, 1.0]])
        settings = "stochastic"

    main()
