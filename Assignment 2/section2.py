from section1 import *
import matplotlib.pyplot as plt

def get_expected_policy_return(
    length: int,
    repeat: int,
    policy: "function",
    disc_factor: float,
):
    cum_return = np.zeros((repeat, length + 1))

    state0 = get_random_initial_state((repeat,))
    for t, (_, _, r, _) in enumerate(trajectory_yield(state0, policy, length), start=1):
        cum_return[:, t] = disc_factor**t * r + cum_return[:, t - 1]

    mean = cum_return.mean(axis=0)
    std = cum_return.std(axis=0)
    
    return mean, std


def plot_expected_policy_return(
    length: int,
    repeat: int,
    policy: "function",
    disc_factor: float,
    dest: str = "exp_return.png",
    show: bool = True,
):

    x_values = list(range(length + 1))
    mean, std = get_expected_policy_return(length, repeat, policy, disc_factor)

    fig = plt.figure(figsize=(6, 4))
    axis = fig.subplots(1, 1)

    axis.set_title("Expected Cumulative Return of a policy")
    axis.set_ylabel("Cumulative return")
    axis.set_xlabel("Time (s)")
    axis.set_ylim((-0.1, 0.60))

    axis.plot(x_values, mean, label="mean")
    axis.fill_between(x_values, mean - std, mean + std, alpha=0.3)

    axis.legend()

    fig.tight_layout()
    if dest:
        fig.savefig(dest)
    if show:
        plt.show()
    plt.close(fig)

    return mean[-1]


def question_2():
    return plot_expected_policy_return(100, 50, policy_random, 0.95)


if __name__ == "__main__":
    get_expected_policy_return(100, 50, policy_random, 0.95)
    # question_2()
