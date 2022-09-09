"""section1_v

a state is a tuple of float `(p, s)`
an action is a float `u`

the parameter $m$ was hard-coded as 1 and removed to simplify some expressions
"""

import math

import numpy as np

_G = 9.81
_DT = 0.001


def hill(p):
    left = p * p**2
    right = p / (1 + 5 * p**2) ** 0.5
    return np.where(p < 0, left, right)


def hill_prime(p):
    left = 1 + 2 * p
    right = 1 / (1 + 5 * p**2) ** 1.5
    return np.where(p < 0, left, right)


def hill_second(p: float) -> float:
    left = 2
    right = -15 * p / (1 + 5 * p**2) ** 2.5
    return np.where(p < 0, left, right)


def one_step(p, s, action):
    h_p1 = hill_prime(p)
    h_p2 = hill_second(p)

    dp_dt = s
    ds_dt = (action - h_p1 * (_G + h_p2 * s**2)) / (1 + h_p1**2)

    return p + _DT * dp_dt, s + _DT * ds_dt


def update(p, s, action, duration=None):
    # euler integration
    duration = duration or 0.1

    terminal_check = terminal_state(p, s)

    n_p, n_s = np.copy(p), np.copy(s)

    for _ in range(math.ceil(duration / _DT)):
        n_p, n_s = one_step(n_p, n_s, action)

    return np.where(terminal_check, p, n_p), np.where(terminal_check, s, n_s)


def state_reward(p, s):
    r = np.where(p > 1, 1.0, 0.0)
    r = np.where((p < -1) | (np.abs(s) > 3), -1.0, r)

    return r


def terminal_state(p, s):
    return (np.abs(p) > 1) | (np.abs(s) > 3)


def reward_and_update(p, s, action, duration):

    n_p, n_s = update(p, s, action, duration)
    r = state_reward(n_p, n_s)

    r = np.where(terminal_state(p, s), 0, r)

    return r, (n_p, n_s)


#


def policy_accelerate(p, s):
    return 4 * np.ones_like(p)


def policy_random(p, s):
    return 8 * (np.random.random(np.empty_like(p).shape) < 0.5) - 4


def policy_heuristic(p, s):
    action = np.where((p < 0.2) & (p > -0.3) & (s < 0.3) & (s > -2), -4, 4)
    # action = np.where((np.abs(p) < 0.4) & (np.abs(s) < 0.8), -4, 4)

    return action


#


def get_random_initial_state(size=()):
    s = np.zeros(size)
    p = np.random.uniform(-0.1, 0.1, size=s.shape)
    return p, s


def trajectory_yield(
    initial_state,
    policy_fun: "function",
    update_count: int,
    update_step: float = None,
    early_stop: bool = False,
):

    p, s = initial_state
    for _ in range(update_count):
        action = policy_fun(p, s)
        reward, next_state = reward_and_update(p, s, action, update_step)

        yield (p, s), action, reward, next_state

        p, s = next_state

        if not early_stop:
            continue

        if terminal_state(p, s).all():
            return

def question_1():
    for (p, s), u, r, (np, ns) in trajectory_yield(
        get_random_initial_state(),
        policy_heuristic,
        update_count=45,
    ):
        print(
            f"x_t=({p:+.3f}, {s:+.3f}), {u=:+.1f}, {r=:+.1f} x_t+1=({np:+.3f}, {ns:+.3f})"
        )


if __name__ == "__main__":
    question_1()
