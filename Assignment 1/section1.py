import itertools
import numpy as np

# n -> x -> number of cols, x increasing to the right
# m -> y -> number of rows, y increasing to the bottom
n, m = 5, 5

# list of (y, x)
state_space = list(itertools.product(range(m), range(n)))

# list of (dy, dx): UP, DOWN, LEFT, RIGHT
action_space = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# grid of rewards
g = np.array([
    [-3, 1, -5, 0, 19],
    [6, 3, 8, 9, 10],
    [5, -8, 4, 1, -8],
    [6, -9, 4, 19, -5],
    [-20, -17, -4, -3, 9],
], dtype=int)

assert g.shape == (m, n)

## update

def update_deterministic(state, action):
    return (
        min(m-1, max(0, state[0]+action[0])),
        min(n-1, max(0, state[1]+action[1])),
    )

def update(state, action, w):
    if w <= 0.5:
        return update_deterministic(state, action)
    return (0, 0)

## reward

def state_reward(destination):
    return g[destination]

def evaluate_reward(state, action, w):
    return state_reward(update(state, action, w))

## combined next state + reward

def reward_and_next_state(state, action, w):
    next_state = update(state, action, w)
    return state_reward(next_state), next_state

## policy

def random_policy(state):
    return action_space[np.random.randint(len(action_space))]

def right_policy(state):
    return (0, 1)

## simulation

def gen_history(length, policy, noise_distribution, initial_state=(3, 0)):
    state = initial_state

    for _ in range(length):
        w = np.random.choice(noise_distribution[:, 1], p=noise_distribution[:, 0])
        action = policy(state)
        reward, next_x = reward_and_next_state(state, action, w)

        yield state, action, reward, next_x

        state = next_x

def simulate(policy, noise_distribution):

    for state, u, r, next_x in gen_history(
        length=10,
        policy=policy,
        noise_distribution=noise_distribution,
        initial_state=(3, 0)
    ):

        print(8*' ' + f'{(state, u, r, next_x)!r},')

        state = next_x


if __name__ == "__main__":
    print('determinstic world:')
    simulate(random_policy, np.array([[1.0, 0.0]]))

    print('\nnon determinstic world:')
    simulate(random_policy, np.array([[0.5, 0.0], [0.5, 1.0]]))
