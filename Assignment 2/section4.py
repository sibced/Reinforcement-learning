import pickle

from matplotlib import cm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from section2 import *


class BaseStopCriterion:
    def __call__(self, t, model, ds_x, ds_y, n_p, n_s, r, next_ds_y) -> bool:
        # prevent divergence
        if np.abs(next_ds_y).max() > 1e4:
            raise ValueError("unstable")
        return False


class PolicyVariationStopCriterion(BaseStopCriterion):
    def __init__(self, ts_size: int, crit_size: int = 100):
        self.indices = np.random.randint(low=0, high=ts_size + 1, size=(crit_size,))
        self.last_pred = 0

        self.x_pos = None
        self.x_neg = None

        self.count = 0

    def __call__(self, t: int, model, ds_x, ds_y, n_p, n_s, r, next_ds_y):
        super().__call__(t, model, ds_x, ds_y, n_p, n_s, r, next_ds_y)

        # lazy init
        if self.x_pos is None:
            self.x_pos = np.copy(ds_x[self.indices])
            self.x_neg = np.copy(self.x_pos)

            self.x_pos[:, 2] = +4
            self.x_neg[:, 2] = -4

        q_pos = model.predict(self.x_pos)
        q_neg = model.predict(self.x_neg)

        pred = 8 * (q_pos > q_neg) - 4

        delta = np.abs(self.last_pred - pred)
        eps = np.max(delta) - 0.1

        print(f"{t=: 4d} {eps=:+.3e}")

        self.last_pred = pred

        # prevent early stopping : policy must be heterogeneous
        if pred.min() == pred.max():
            return False

        if eps < 0:
            self.count += 1
        else:
            self.count = 0

        return self.count >= 10


class CountStopCriterion(BaseStopCriterion):
    def __init__(self, val: int):
        self.val = val

    def __call__(self, t, *args):
        super().__call__(t, *args)
        return t > self.val

class SubOptimalStopCriterion(CountStopCriterion):
    def __init__(self, disc_factor: float, bound: float = 0.01):
        self.B_r = 1
        self.N = int( np.log( bound * (1-disc_factor)**2/(2*self.B_r) )/ np.log(disc_factor) )
        super().__init__(self.N)

    def __call__(self, t, *args):
        return super().__call__(t, *args)

def random_uniform_trainset(n_gen: int):

    p_val = 2 * np.random.random((n_gen,)) - 1
    s_val = 6 * np.random.random((n_gen,)) - 3
    u_val = 8 * (np.random.random((n_gen,)) < 0.5) - 4

    r, (n_p, n_s) = reward_and_update(p_val, s_val, u_val, duration=None)

    ds_x = np.stack([p_val, s_val, u_val], axis=-1)
    ds_y = r

    # print(f"r>0 count: {(r>0).sum()}")

    return ds_x, ds_y, n_p, n_s, r

def random_trainset_near_terminal(
    n_high_speed = 200,
    n_high_p = 200,
):
    p_val = np.empty((n_high_p + n_high_speed,))
    s_val = np.empty((n_high_p + n_high_speed,))
    # entirely random actions
    u_val = 8 * (np.random.random((n_high_p + n_high_speed,)) < 0.5) - 4

    low = 0
    high = n_high_speed // 2
    
    p_val[low:high] = 2 * np.random.random((high-low,)) - 1
    s_val[low:high] = 6 - 0.5 * np.random.random((high-low,))

    low = high
    high = n_high_speed

    p_val[low:high] = 2 * np.random.random((high-low,)) - 1
    s_val[low:high] = 0.5 * np.random.random((high-low,))

    low = high
    high = n_high_speed + n_high_p // 2
    
    p_val[low:high] = 1 - 0.5 * np.random.random((high-low,))
    s_val[low:high] = 6 * np.random.random((high-low,)) - 3

    low = high
    high = n_high_speed + n_high_p
    
    p_val[low:high] = -1 + 0.5 * np.random.random((high-low,))
    s_val[low:high] = 6 * np.random.random((high-low,)) - 3

    r, (n_p, n_s) = reward_and_update(p_val, s_val, u_val, duration=None)

    ds_x = np.stack([p_val, s_val, u_val], axis=-1)
    ds_y = r
    
    return ds_x, ds_y, n_p, n_s, r


def success_rate(limit=int(1e5)):
    "print how many trajectory were successful using a random policy"

    state0 = get_random_initial_state(size=(1000,))
    counts = limit + 1 + np.zeros_like(state0[0])
    for idx, (_, _, r, _) in enumerate(
        trajectory_yield(state0, policy_random, update_count=limit)
    ):
        value = np.where(r > 0, idx, limit + 1)
        counts = np.minimum(counts, value)
        if idx % 100 == 0:
            print(idx)

    success_count = (counts < 1001).sum()
    print(f"{success_count} trajectories reached the goal")
    max_len = counts[counts < 1001].max()
    print(f"the longest one took {max_len} steps")


def random_trajectory_trainset(n_trajectory: int, trajectory_len: int):

    # independent trajectories can be computed in parallel
    state0 = get_random_initial_state(size=(n_trajectory,))
    policy = policy_random

    p_lst = [None] * trajectory_len
    s_lst = [None] * trajectory_len
    u_lst = [None] * trajectory_len
    r_lst = [None] * trajectory_len
    n_p_lst = [None] * trajectory_len
    n_s_lst = [None] * trajectory_len

    for idx, ((p, s), u, r, (n_p, n_s)) in enumerate(
        trajectory_yield(state0, policy, update_count=trajectory_len)
    ):

        p_lst[idx] = p
        s_lst[idx] = s
        u_lst[idx] = u
        r_lst[idx] = r
        n_p_lst[idx] = n_p
        n_s_lst[idx] = n_s

    # how many successful / failure ?
    # total_rewards = sum(r_lst)  # sum each trajectories

    # success_rate = (total_rewards > 0.01).sum() / total_rewards.size
    # print(f"{success_rate=:+.2f}")
    # failure_rate = (total_rewards < -0.01).sum() / total_rewards.size
    # print(f"{failure_rate=:+.2f}")
    # null_rate = 1 - success_rate - failure_rate
    # print(f"{null_rate=:+.2f}")

    p = np.concatenate(p_lst)
    s = np.concatenate(s_lst)
    u = np.concatenate(u_lst)
    r = np.concatenate(r_lst)
    n_p = np.concatenate(n_p_lst)
    n_s = np.concatenate(n_s_lst)

    # print(f"r>0 count: {(r>0).sum()}")

    ds_x = np.stack([p, s, u], axis=-1)
    ds_y = r

    return ds_x, ds_y, n_p, n_s, r

def merge_datasets(*datasets):
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    
    ds_x = np.concatenate([ds[0] for ds in datasets])
    ds_y = np.concatenate([ds[1] for ds in datasets])
    n_p = np.concatenate([ds[2] for ds in datasets])
    n_s = np.concatenate([ds[3] for ds in datasets])
    r = np.concatenate([ds[4] for ds in datasets])
    
    return ds_x, ds_y, n_p, n_s, r


def next_q_target(
    model,
    n_p,
    n_s,
    r,
    disc_factor,
):

    # input space : next state with all possible action combinations
    x_pos = np.stack([n_p, n_s, +4 * np.ones_like(n_p)], axis=-1)
    x_neg = np.stack([n_p, n_s, -4 * np.ones_like(n_p)], axis=-1)

    # compute the Q-function
    pred_pos = model.predict(x_pos)
    pred_neg = model.predict(x_neg)

    # build the next training set
    return r + disc_factor * np.maximum(pred_pos, pred_neg)


def fitted_q(
    n_iteration: int,
    disc_factor: float,
    dataset,
    model_generator: "function",
    criterion,
    print_y_stats: bool = False,
    print_y_diff: bool = True,
    print_iteration: bool = True,
    save_every_n: int = -1,
    print_score: bool = False,
):

    ds_x, ds_y, n_p, n_s, r = dataset

    if print_y_stats:
        print(
            f"{ds_y.min()=:+.3E}, {ds_y.max()=:+.3E}, {ds_y.mean()=:+.3E}, {ds_y.std()=:+.3E}"
        )

    if save_every_n < 1:
        save_every_n = n_iteration + 1

    model = None

    for t in range(n_iteration):
        if print_iteration:
            print(f"interation {t=:4d}")

        model = model_generator(ds_x, ds_y)

        if print_score:
            print(f"{model.score(ds_x, ds_y)=}")

        n_ds_y = next_q_target(model, n_p, n_s, r, disc_factor)

        diff = np.max(np.abs(n_ds_y - ds_y))
        ds_y = n_ds_y

        # if print_y_diff:
            # print(f"\t{diff=:.3E} {(ds_y.max() - ds_y.min())=}")

        if diff > 1e4:
            raise RuntimeError("divergence encountered")

        if print_y_stats:
            print(
                f"{ds_y.min()=:+.3E}, {ds_y.max()=:+.3E}, {ds_y.mean()=:+.3E}, {ds_y.std()=:+.3E}"
            )

        if criterion(t, model, ds_x, ds_y, n_p, n_s, r, n_ds_y):
            break

        if (t + 1) % save_every_n == 0:

            def policy_(p, s):
                neg = np.stack([p, s, -4 * np.ones_like(p)], axis=-1)
                pos = np.stack([p, s, +4 * np.ones_like(p)], axis=-1)

                q_pos = model.predict(pos)
                q_neg = model.predict(neg)

                return 8 * (q_pos > q_neg) - 4

            print(get_expected_policy_return(50, 10, policy_, disc_factor)[0][-1])
            # save_figs(model, t=t + 1, N=n_iteration)

    return model


def extra_tree_generator(ds_x, ds_y):
    et = ExtraTreesRegressor(n_jobs=4)
    et.fit(ds_x, ds_y)
    return et

def lr_regressor(ds_x, ds_y):
    lr = LinearRegression()
    lr.fit(ds_x, ds_y)
    return lr

def mlp_generator(ds_x, ds_y):
    mlp = MLPRegressor(
        hidden_layer_sizes=(5, 5),
        activation="logistic",
    )
    mlp.fit(ds_x, ds_y)
    return mlp


def get_mlp_warmstart_generator():
    model = MLPRegressor(
        hidden_layer_sizes=(5, 5),
        activation="logistic",
        warm_start=True,
    )

    def _generator(ds_x, ds_y):
        model.fit(ds_x, ds_y)
        return model

    return _generator


generators = {'mlp':get_mlp_warmstart_generator,
              'lr':lr_regressor,
              'et':extra_tree_generator}

def save_figs(model, prefix: str = "fitted_q", **kwargs):

    p = np.linspace(-1, 1, 201)
    s = np.linspace(-3, 3, 601)

    pp, ss = np.meshgrid(p, s)
    shape_2d = pp.shape
    pp = pp.flatten()
    ss = ss.flatten()

    fig, (ax_l, ax_r) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))
    cmap = cm.coolwarm_r

    kv_slug = "_".join(f"{k}={v}" for k, v in kwargs.items())
    kv_text = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    if kv_text:
        kv_text = " (" + kv_text + ")"

    ax_r.set_title("Q((p, s), u=+4)" + kv_text)
    ax_r.set_xlabel("p")
    ax_l.set_title("Q((p, s), u=-4)" + kv_text)
    ax_l.set_xlabel("p")
    ax_l.set_ylabel("s")

    # q_pos = np.sin(9 * np.exp(-pp**2 -ss**2))
    ds_pos = np.stack([pp, ss, 4 * np.ones_like(pp)], axis=-1)
    q_pos = model.predict(ds_pos)
    q_pos = q_pos.reshape(shape_2d)

    # q_neg = np.sin(6 * np.exp(-pp**2 - ss**2))
    ds_neg = np.stack([pp, ss, -4 * np.ones_like(pp)], axis=-1)
    q_neg = model.predict(ds_neg)
    q_neg = q_neg.reshape(shape_2d)

    # print(f"{np.max(np.abs(q_neg-q_pos))=}")

    vmin = min(q_pos.min(), q_neg.min())
    vmax = max(q_pos.max(), q_neg.max())

    pc = ax_r.pcolormesh(p, s, q_pos, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=ax_r)

    pc = ax_l.pcolormesh(p, s, q_neg, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pc, ax=ax_l)

    fig.tight_layout()
    fig.savefig(f"{prefix}_{type(model).__name__}_{kv_slug}.png")

    plt.close(fig)

    # policy
    policy = 8 * (q_pos > q_neg) - 4
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    ax.set_title("mu((p, s))" + kv_text)
    ax.set_xlabel("p")
    ax.set_ylabel("s")

    pc = ax.pcolormesh(p, s, policy, cmap=cmap, vmin=-5, vmax=+5)
    fig.colorbar(pc, ax=ax)

    fig.tight_layout()
    fig.savefig(f"{prefix}_{type(model).__name__}_policy_{kv_slug}.png")

    plt.close(fig)


def extract_policy(model):
    def policy_(p, s):
        neg = np.stack([p, s, -4 * np.ones_like(p)], axis=-1)
        pos = np.stack([p, s, +4 * np.ones_like(p)], axis=-1)

        q_pos = model.predict(pos)
        q_neg = model.predict(neg)

        return 8 * (q_pos > q_neg) - 4

    return policy_


def expected_return_model(
    model,
    length: int,
    repeat: int,
    disc_factor: float,
    dest: str = "exp_return.png",
    show: bool = True,
):

    return plot_expected_policy_return(
        length, repeat, extract_policy(model), disc_factor, dest=dest, show=show
    )

datasets = {'random_trajectory_trainset':'traj',
            'merge_datasets':'mixt'}

def fit_and_plot(model_name):

    n_iteration = 500
    ds_size = 20000

    """
    dataset = merge_datasets(
        random_trajectory_trainset(1000, 200),
        random_trainset_near_terminal(6000, 4000),
        random_uniform_trainset(5000),
    )
    """
    dataset = random_trajectory_trainset(100, 200)
    dataset_name = datasets['random_trajectory_trainset']
    # dataset = random_uniform_trainset(ds_size)

    model_gen = generators[model_name]

    np.random.seed()

    #criterion = PolicyVariationStopCriterion(len(dataset), crit_size=100)
    criterion = CountStopCriterion(n_iteration)
    #criterion = SubOptimalStopCriterion(disc_factor=0.95)

    model = fitted_q(
        dataset=dataset,
        model_generator=model_gen,
        criterion=criterion,
        ## args
        n_iteration=n_iteration,
        disc_factor=0.95,
        save_every_n=5,
        print_y_diff=True,
        print_iteration=False,
    )

    with open(f"{model_name}_{n_iteration}_{dataset_name}_pol.pkl", mode="wb") as file:
        pickle.dump(model, file)

    print(
        expected_return_model(
            model, 100, 50, 0.95, dest=f"{model_name}_{n_iteration}_{dataset_name}_pol.png", show=False
        )
    )

    class lit:
        def __init__(self, val: str) -> None:
            self.val = val

        def __str__(self) -> str:
            return self.val

    save_figs(model, ds=lit(dataset_name), crit=lit("count"))
    save_figs(model, ds=lit(dataset_name), crit=lit("pol"))

if __name__ == "__main__":
    #fit_and_plot_mlp()
    fit_and_plot('lr')
    #fit_and_plot('et')
