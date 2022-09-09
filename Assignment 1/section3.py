from section2 import *


#Reward function r(x, u) of MDP
def get_true_reward(
    noise_distribution,
):
    reward = np.zeros((len(action_space), m, n))

    for x, y in state_space:
        for i_action, action in enumerate(action_space):
            for prob, w in noise_distribution:
                val = prob * evaluate_reward((x, y), action, w)
                reward[i_action, x, y] += val
    return reward


#Transition probabilities p(x'|x,u) of MDP
def get_true_prob(
    noise_distribution,
):

    true_prob = np.zeros((len(action_space), m, n, m, n))

    for x0, y0 in state_space:
        for i_action, action in enumerate(action_space):
            for xf, yf in state_space:
                for prob, w in noise_distribution:
                    if (xf, yf) == update((x0, y0), action, w):
                        val = prob * 1
                        true_prob[i_action, x0, y0, xf, yf] += val

    return true_prob


def main_deterministic():
    noise_distribution = np.array(
        [
            [1.0, 0.0],
        ]
    )

    return main_print(get_true_reward(noise_distribution), get_true_prob(noise_distribution))


def main_stochastic():
    noise_distribution = np.array(
        [
            [0.5, 0.0],
            [0.5, 1.0],
        ]
    )

    return main_print(get_true_reward(noise_distribution), get_true_prob(noise_distribution))

def Q(true_reward, true_prob):

    #Q_0(x, u) ≡ 0
    Q_now = np.zeros((len(action_space), m, n))

    N = int( np.log( 0.1 * (1-disc_factor)**2/ (2*B_r) )/ np.log(disc_factor) )
    #N=2000000

    for _ in range(N):
        max_Q = np.amax(Q_now, axis=0)
        Q_now = true_reward + disc_factor * np.sum((true_prob*max_Q), axis=(3,4))
    
    return Q_now

def match_actions(index_table):
    for x in range(index_table.shape[0]):
        for y in range(index_table.shape[1]):
            if y < index_table.shape[1] - 1:
                print(action_space[index_table[x][y]], end = ', ')
            else:
                print(action_space[index_table[x][y]])
        print("\n")

def main_print(true_reward, true_prob):
    
    q = Q(true_reward, true_prob)
    
    opt_policy = q.argmax(axis=0)
    print("\nOptimal policy µ*_N = ")
    print(opt_policy)
    print("which corresponds to :")
    match_actions(opt_policy)
    
    print("J^µ*_N = ")
    j = np.amax(q, axis=0)
    print(j)
    return j


if __name__ == "__main__":
    #main_stochastic()
    print("Deterministic domain")
    main_deterministic()

    print("\nStochastic domain")
    main_stochastic()