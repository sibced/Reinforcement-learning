"utilities: (GAE, trajectory sampling, ECR)"

import gym
import numpy as np
from scipy import signal

GAMMA = 0.99


def discounted_cumulative_sum(
    values,  # array of T float values
    discount: float,
):  # return an array of T float values

    # https://stackoverflow.com/a/47971187/5770818

    # print(f"{values=!r}")

    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=values[::-1])

    return y[::-1]

    # this is equivalent to and faster than :
    return np.array(
        [
            sum(discount**k * values[t + k] for k in range(len(values) - t))
            for t in range(len(values))
        ]
    )


def compute_rewards_to_go(
    rewards,  # array of T float values
):
    "returns an array for the rewards to go at each step"
    # print(f"{type(rewards)=} {len(rewards)=}")
    # try:
    #     print(f"{rewards[0]=!r}")
    # except IndexError:
    #     pass
    # try:
    #     print(f"{rewards.shape=}")
    # except AttributeError:
    #     pass

    return discounted_cumulative_sum(rewards, GAMMA)

    # this is equivalent to and faster than :
    return np.array(
        [
            sum(GAMMA**k * rewards[t + k] for k in range(len(rewards) - t))
            for t in range(len(rewards))
        ]
    )

def expected_cumulative_return(
    env,
    policy,
    max_len: int,
    n_samples: int,
):

    def _do_once():
        _, _, rewards = sample_trajectory(env, policy, max_len)
        return compute_rewards_to_go(rewards)[0]
    
    return sum(_do_once() for _ in range(n_samples)) / n_samples
        


def estimate_advantage(
    state_values,  # array of T float values
    rewards,  # array of T float values
    lambda_: float,
):
    trajectory_len = rewards.shape[0]

    # force numpy instead of PyTorch: only numpy supports arr[::-1]
    state_values = np.array(state_values[:, 0])

    deltas = GAMMA * state_values[1:] - state_values[:-1] + rewards

    # TODO is this right ? it doesn't seem to be
    offset = GAMMA ** (trajectory_len - 1) * state_values[-1]

    return offset + discounted_cumulative_sum(deltas, GAMMA * lambda_)

    # this is equivalent to and faster than :
    return offset + np.array(
        [
            sum(
                (GAMMA * lambda_) ** k * deltas[t + k]
                for k in range(trajectory_len - t)
            )
            for t in range(trajectory_len)
        ]
    )


def sample_trajectory(
    env: gym.Env,  # a gym.Env
    policy_network,  # a Callable[[np.ndarray], np.ndarray]
    max_len: int,
):
    """sample a trajectory

    Args:
        env (gym.Env): a GYM environment
        policy_network: a function that maps observation to action for the environment
        max_len (int): length limit to the number of transition in the trajectory

    Returns:
        o_t [0, T] : ALL observation
        u_t [0, T) : action taking after observing o_t
        r_t [0, T) : reward perceived after doing u_t with observation o_t
    """

    observation_arr = np.empty((max_len + 1,) + env.observation_space.shape, dtype="f4")
    action_arr = np.empty((max_len,) + env.action_space.shape, dtype="f4")
    reward_arr = np.empty((max_len,), dtype="f4")

    observation_arr[0] = env.reset()

    for i in range(max_len):
        action = policy_network(observation_arr[i])
        next_observation, reward, done, _ = env.step(action)

        action_arr[i] = action
        reward_arr[i] = reward
        observation_arr[i + 1] = next_observation

        if done:
            max_len = i + 1
            break

    return (
        observation_arr[: max_len + 1],
        action_arr[:max_len],
        reward_arr[:max_len],
    )