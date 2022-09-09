import numpy as np
from pybulletgym.envs.roboschool.envs.pendulum.inverted_double_pendulum_env import \
    InvertedDoublePendulumBulletEnv
from sklearn.ensemble import ExtraTreesRegressor

from utils import compute_rewards_to_go

GAMMA = 0.95  # to reduce nb of iterations
B_r = 12.89  # (p_x,p_y) = (0,0) => max reward = 12.89
# FQI works for discrete environment
action_space = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])


def sample_trajectory(env, policy, max_len):
    "different from utils: different return values"

    observation_arr = np.empty(
        (max_len,) + env.observation_space.shape, dtype=env.observation_space.dtype
    )
    action_arr = np.empty(
        (max_len,) + env.action_space.shape, dtype=env.action_space.dtype
    )
    reward_arr = np.empty((max_len,), dtype="f4")
    next_observation_arr = np.empty(
        (max_len,) + env.observation_space.shape, dtype=env.observation_space.dtype
    )

    observation_arr[0] = env.reset()
    for i in range(max_len):
        action = policy(observation_arr[i].reshape(1, -1))

        next_observation, reward, done, _ = env.step(action)
        action_arr[i] = action
        reward_arr[i] = reward
        next_observation_arr[i] = next_observation

        if done or i == max_len - 1:
            observation_arr = observation_arr[: i + 1]
            action_arr = action_arr[: i + 1]
            reward_arr = reward_arr[: i + 1]
            next_observation_arr = next_observation_arr[: i + 1]
            break
        else:
            observation_arr[i + 1] = next_observation

    return (observation_arr, action_arr, reward_arr, next_observation_arr)


def next_q_target(model, observation, reward):
    "get the next Y for the FQI algorithm"
    # observation: (n, 9)
    # rewards: (n, 1)

    action = np.zeros((len(observation), 1), dtype="f4")
    ds_x = np.concatenate([observation, action], axis=1)

    q_values = np.zeros((len(action_space), len(observation)))

    for idx, action in enumerate(action_space):
        ds_x[:, -1] = action
        q_values[idx, :] = model.predict(ds_x)

    return reward + GAMMA * q_values.max(axis=0)


def get_policy(model, epsilon: float = None):
    "return a (possibly epsilon-greedy) policy"
    def _policy(observation):
        action = np.zeros((len(observation), 1), dtype="f4")
        ds_x = np.concatenate([observation, action], axis=1)

        q_values = np.zeros((len(action_space), len(observation)))

        for idx, action in enumerate(action_space):
            ds_x[:, -1] = action
            q_values[idx, :] = model.predict(ds_x)

        indices = q_values.argmax(axis=0)

        return action_space[indices]

    def _greedy_policy(observation):

        good_actions = _policy(observation)
        rand_actions = np.random.choice(action_space, (len(observation),))

        rand_mask = np.random.rand(len(observation)) < epsilon

        return np.where(rand_mask, rand_actions, good_actions)

    if epsilon:
        return _greedy_policy

    return _policy


def random_policy(observation):
    return np.random.choice(action_space, (len(observation),))


def do_fqi(trajectory_length):

    n_iterations = int(np.log(0.1 * (1 - GAMMA) ** 2 / (2 * B_r)) / np.log(GAMMA))

    model = ExtraTreesRegressor(n_jobs=4)
    policy = get_policy(model)

    with InvertedDoublePendulumBulletEnv() as env:
        observations, actions, rewards, next_observations = zip(
            *(
                sample_trajectory(env, random_policy, trajectory_length)
                for _ in range(50)
            )
        )

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)

        for iteration in range(n_iterations):
            print(f"{iteration=:4d} started")
            ds_x = np.concatenate([observations, actions], axis=1)
            ds_y = rewards
            print(f"{iteration=:4d} made dataset")
            model.fit(ds_x, ds_y)
            print(f"{iteration=:4d} fitted")

            ds_y = next_q_target(model, next_observations, rewards)
            print(f"{iteration=:4d} next dataset prepared")

            ecr = (
                sum(
                    compute_rewards_to_go(sample_trajectory(env, policy, 50)[2])[0]
                    for _ in range(5)
                )
                / 5
            )
            print(f"{iteration=:4d} {ecr=:.4f}")


if __name__ == "__main__":
    test_length = 150
    do_fqi(test_length)


"""
iteration=   0 started
iteration=   0 made dataset
iteration=   0 fitted
iteration=   0 next dataset prepared
iteration=   0 ecr=228.4658
iteration=   1 started
iteration=   1 made dataset
iteration=   1 fitted
iteration=   1 next dataset prepared
iteration=   1 ecr=274.4154
iteration=   2 started
iteration=   2 made dataset
iteration=   2 fitted
iteration=   2 next dataset prepared
iteration=   2 ecr=255.0989
iteration=   3 started
iteration=   3 made dataset
iteration=   3 fitted
iteration=   3 next dataset prepared
iteration=   3 ecr=224.6316
iteration=   4 started
iteration=   4 made dataset
iteration=   4 fitted
iteration=   4 next dataset prepared
iteration=   4 ecr=313.5742
iteration=   5 started
iteration=   5 made dataset
iteration=   5 fitted
iteration=   5 next dataset prepared
iteration=   5 ecr=256.1562
iteration=   6 started
iteration=   6 made dataset
iteration=   6 fitted
iteration=   6 next dataset prepared
iteration=   6 ecr=271.9097
iteration=   7 started
iteration=   7 made dataset
iteration=   7 fitted
iteration=   7 next dataset prepared
iteration=   7 ecr=244.5282
iteration=   8 started
iteration=   8 made dataset
iteration=   8 fitted
iteration=   8 next dataset prepared
iteration=   8 ecr=278.1899
iteration=   9 started
iteration=   9 made dataset
iteration=   9 fitted
iteration=   9 next dataset prepared
iteration=   9 ecr=271.8285
iteration=  10 started
iteration=  10 made dataset
iteration=  10 fitted
iteration=  10 next dataset prepared
iteration=  10 ecr=263.6586
iteration=  11 started
iteration=  11 made dataset
iteration=  11 fitted
iteration=  11 next dataset prepared
iteration=  11 ecr=301.1216
iteration=  12 started
iteration=  12 made dataset
iteration=  12 fitted
iteration=  12 next dataset prepared
iteration=  12 ecr=237.5262
iteration=  13 started
iteration=  13 made dataset
iteration=  13 fitted
iteration=  13 next dataset prepared
iteration=  13 ecr=250.7663
iteration=  14 started
iteration=  14 made dataset
iteration=  14 fitted
iteration=  14 next dataset prepared
iteration=  14 ecr=303.1702
iteration=  15 started
iteration=  15 made dataset
iteration=  15 fitted
iteration=  15 next dataset prepared
iteration=  15 ecr=290.5644
iteration=  16 started
iteration=  16 made dataset
iteration=  16 fitted
iteration=  16 next dataset prepared
"""