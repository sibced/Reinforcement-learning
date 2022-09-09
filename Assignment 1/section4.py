import matplotlib.pyplot as plt

from section3 import *

def inf_dist(l, r):
    "returns the distance between two arrays"
    return np.amax(np.absolute(r - l))


def iterate_r_p_estimation(history):

    # build the tables
    action_count = len(action_space)

    count_act_sta = np.zeros((action_count, m, n))
    count_act_sta_sta = np.zeros((action_count, m, n, m, n))

    reward_sum = np.zeros((action_count, m, n))
    reward = np.zeros((action_count, m, n))

    prob = np.zeros((action_count, m, n, m, n))

    for (y0, x0), u, r, (yf, xf) in history:
        i_u = action_space.index(u)

        count_act_sta[i_u, y0, x0] += 1
        count_act_sta_sta[i_u, y0, x0, yf, xf] += 1

        reward_sum[i_u, y0, x0] += r

        reward[i_u, y0, x0] = reward_sum[i_u, y0, x0] / count_act_sta[i_u, y0, x0]

        prob[i_u, y0, x0] = count_act_sta_sta[i_u, y0, x0] / count_act_sta[i_u, y0, x0]

        yield reward, prob


def get_r_p_estimation(history):
    reward, prob = None, None
    for reward, prob in iterate_r_p_estimation(history):
        pass
    if reward is None or prob is None:
        raise ValueError("no reward found from history")
    return reward, prob


def plot_r_p_distance(
    history,
    true_reward,
    true_prob,
):
    # suppose true_reward & true_prob are defaultdict too

    reward_dist_speed = []
    prob_dist_speed = []

    for reward, prob in iterate_r_p_estimation(history):
        reward_dist_speed.append(inf_dist(reward, true_reward))

        prob_dist_speed.append(inf_dist(prob, true_prob))

    # plot
    indices = list(range(len(reward_dist_speed)))

    fig, ax1 = plt.subplots()
    ax1.set_title("Convergence speed")

    height = max(g.max(), max(reward_dist_speed))
    ax1.set_ylim(-0.05 * height, height + 0.05 * height)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Inf. norm for the reward error (--)")
    ax1.plot(indices, reward_dist_speed, "--", color="blue", label="rewards")

    ax2 = ax1.twinx()

    ax2.set_ylabel("Inf. norm for the probability error (-)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.plot(indices, prob_dist_speed, "-", color="orange", label="prob")

    fig.tight_layout()
    fig.savefig("tmp.png")
    
    return reward, prob


def get_q_table(
    reward,
    prob,
    N=1000,
    discount_factor=0.99,
) -> np.ndarray:

    q_prev = np.zeros((len(action_space), m, n))
    q_curr = np.zeros_like(q_prev)

    for _ in range(N):
        for y0, x0 in state_space:
            for i_action, _ in enumerate(action_space):

                exp = 0.0
                for yf, xf in state_space:
                    exp += prob[i_action, y0, x0, yf, xf] * q_prev[:, yf, xf].max(axis=0)

                val = reward[i_action, y0, x0] + discount_factor * exp
                q_curr[i_action, y0, x0] = val

        # swap tables
        q_prev, q_curr = q_curr, q_prev

    return q_prev


def get_j_table(
    reward,
    prob,
    N=100,
    discount_factor=0.99,
):
    q_table = get_q_table(reward, prob, N, discount_factor)
    return q_table.max(axis=0)


def get_mu_table(
    reward,
    prob,
    N=100,
    discount_factor=0.99,
):
    q_table = get_q_table(reward, prob, N, discount_factor)

    indices = q_table.argmax(axis=0)
    return [[action_space[idx] for idx in row] for row in indices]

def get_mu_fun(
    reward,
    prob,
    N=100,
    discount_factor=0.99,
):
    policy_table = get_mu_table(reward, prob, N, discount_factor)
    
    def _policy(state):
        return policy_table[state[0]][state[1]]
    return _policy


IS_DETERMINISTIC = False

def main_policy():
    if IS_DETERMINISTIC:
        h_len = 1000
        noise_distribution = np.array([[1.0, 0.0]])
    else:
        h_len = int(2e6)
        noise_distribution = np.array([[0.5, 0.0], [0.5, 1.0]])
    N = 1508

    history = gen_history(h_len, random_policy, noise_distribution)
    
    true_reward = get_true_reward(noise_distribution)
    true_prob = get_true_prob(noise_distribution)
    
    reward, prob = plot_r_p_distance(history, true_reward, true_prob)

    policy = get_mu_fun(reward, prob, N, discount_factor=0.99)
    
    print("\nPolicy:")
    for row in range(m):
        print(f'\t{row}', end='')
        for col in range(n):
            action = policy((row, col))
            print(f' & ${action}$', end='')

        print(' \\\\\n\\hline\n', end='')
    
    # th
    expected_return_th = expected_return(policy, noise_distribution)
    
    # practical
    expected_return_pr = get_j_table(reward, prob, N, discount_factor=0.99)
    
    print('\nValue:')
    for row in range(m):
        print(f'\t{row}', end='')
        for col in range(n):
            th = expected_return_th[row][col]
            ex = expected_return_pr[row][col]
            print(f' & ${ex: .2f}\\;{th: .2f}$', end='')

        print(' \\\\\n\\hline\n', end='')
    
    print(f"{inf_dist(expected_return_th, expected_return_pr)=}")


def print_q_inf_norm():
    ""
    if IS_DETERMINISTIC:
        h_len = 1000
        noise_distribution = np.array([[1.0, 0.0]])
    else:
        h_len = int(2e6)
        noise_distribution = np.array([[0.5, 0.0], [0.5, 1.0]])
    N = 1508

    history = gen_history(h_len, random_policy, noise_distribution)
    
    true_reward = get_true_reward(noise_distribution)
    true_prob = get_true_prob(noise_distribution)
    
    true_q_table = get_q_table(true_reward, true_prob)
    
    for idx, (reward, prob) in enumerate(iterate_r_p_estimation(history)):
        if idx & (idx - 1) == 0:  # power of 2 and 0
            q_table = get_q_table(reward, prob, N)
            dist = inf_dist(q_table, true_q_table)
            print(f"{idx=} {dist: .2f} {inf_dist(q_table.max(axis=0), true_q_table.max(axis=0)): .2f}")
    print(f"{idx=} {dist: .2f} {inf_dist(q_table.max(axis=0), true_q_table.max(axis=0)): .2f}")
    

def main():
    # interesting seeds for stochastic settings: 146
    seed = np.random.randint(1000) * 1 + 146
    np.random.seed(seed)
    print(f'{seed=}')

    # main_policy()
    print_q_inf_norm()
    


if __name__ == "__main__":
    main()


"""
146

Policy:
        0 & $(1, 0)$ & $(0, 1)$ & $(0, 1)$ & $(0, 1)$ & $(-1, 0)$ \\
\hline
        1 & $(0, 1)$ & $(0, 1)$ & $(0, 1)$ & $(0, 1)$ & $(-1, 0)$ \\
\hline
        2 & $(-1, 0)$ & $(0, 1)$ & $(-1, 0)$ & $(-1, 0)$ & $(-1, 0)$ \\
\hline
        3 & $(-1, 0)$ & $(0, 1)$ & $(0, 1)$ & $(-1, 0)$ & $(-1, 0)$ \\
\hline
        4 & $(-1, 0)$ & $(0, 1)$ & $(-1, 0)$ & $(-1, 0)$ & $(0, -1)$ \\
\hline

Value:
        0 & $ 1842.03\; 1841.93$ & $ 1857.19\; 1857.09$ & $ 1881.00\; 1880.90$ & $ 1900.00\; 1899.90$ & $ 1900.00\; 1899.90$ \\
\hline
        1 & $ 1854.58\; 1854.48$ & $ 1870.28\; 1870.18$ & $ 1881.09\; 1880.99$ & $ 1891.00\; 1890.90$ & $ 1900.00\; 1899.90$ \\
\hline
        2 & $ 1842.03\; 1841.93$ & $ 1855.58\; 1855.48$ & $ 1870.28\; 1870.18$ & $ 1881.09\; 1880.99$ & $ 1891.00\; 1890.90$ \\
\hline
        3 & $ 1828.61\; 1828.51$ & $ 1849.01\; 1848.91$ & $ 1863.65\; 1863.55$ & $ 1863.28\; 1863.18$ & $ 1864.09\; 1863.99$ \\
\hline
        4 & $ 1816.32\; 1816.22$ & $ 1826.52\; 1826.42$ & $ 1849.01\; 1848.91$ & $ 1863.65\; 1863.55$ & $ 1842.01\; 1841.91$ \\
\hline
inf_dist(expected_return_th, expected_return_pr)=0.09978961316824098
"""


"""
146

Policy:
        0 & $(1, 0)$ & $(1, 0)$ & $(0, 1)$ & $(0, 1)$ & $(-1, 0)$ \\
\hline
        1 & $(0, 1)$ & $(0, 1)$ & $(0, 1)$ & $(0, 1)$ & $(-1, 0)$ \\
\hline
        2 & $(-1, 0)$ & $(0, 1)$ & $(1, 0)$ & $(1, 0)$ & $(-1, 0)$ \\
\hline
        3 & $(0, -1)$ & $(0, 1)$ & $(0, 1)$ & $(0, -1)$ & $(0, -1)$ \\
\hline
        4 & $(-1, 0)$ & $(0, 1)$ & $(-1, 0)$ & $(-1, 0)$ & $(1, 0)$ \\
\hline

Value:
        0 & $ 157.64\; 159.44$ & $ 157.76\; 159.63$ & $ 161.66\; 162.62$ & $ 172.20\; 172.12$ & $ 173.18\; 172.12$ \\
\hline
        1 & $ 157.79\; 159.63$ & $ 161.08\; 163.04$ & $ 162.67\; 164.89$ & $ 165.08\; 167.62$ & $ 169.21\; 172.12$ \\
\hline
        2 & $ 157.65\; 159.44$ & $ 158.63\; 159.70$ & $ 161.60\; 162.19$ & $ 165.91\; 167.20$ & $ 164.51\; 167.62$ \\
\hline
        3 & $ 157.19\; 159.25$ & $ 160.66\; 162.19$ & $ 166.92\; 167.20$ & $ 160.45\; 162.19$ & $ 169.58\; 167.20$ \\
\hline
        4 & $ 157.42\; 159.25$ & $ 153.24\; 155.70$ & $ 159.03\; 162.19$ & $ 163.91\; 167.20$ & $ 167.71\; 162.22$ \\
\hline
inf_dist(expected_return_th, expected_return_pr)=5.489367351231152
"""

"""
(info8003_env) ➜  info8003 git:(master) ✗ python section4.py
seed=748
idx=1  1899.92  1899.92
idx=2  1899.92  1899.92
idx=3  1899.92  1899.92
idx=5  1899.92  1899.92
idx=9  1899.92  1899.92
idx=17  1899.92  1899.92
idx=33  1899.92  1899.92
idx=65  1899.92  1899.92
idx=129  1899.92  1899.92
idx=257  1899.92  10.07
idx=513  1899.92  0.08
idx=1000 1899.92  0.08
(info8003_env) ➜  info8003 git:(master) ✗ python section4.py
seed=748
idx=1  172.12  172.12
idx=2  172.12  172.12
idx=3  172.12  172.12
idx=5  172.12  172.12
idx=9  172.12  172.12
idx=17  440.56  440.37
idx=33  440.56  440.37
idx=65  172.12  172.12
idx=129  172.12  172.12
idx=257  391.00  390.81
idx=513  391.00  390.81
idx=1025  391.00  390.81
idx=2049  287.37  280.14
idx=4097  287.37  280.14
idx=8193  172.12  172.12
idx=16385  167.62  167.21
idx=32769  167.21  167.21
idx=65537  167.21  167.21
idx=131073  167.21  167.21
idx=262145  162.22  162.22
idx=524289  162.22  9.20
idx=1048577  162.22  11.56
idx=2000000  162.b2  11.56
"""