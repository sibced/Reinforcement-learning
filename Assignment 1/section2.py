from section1 import *


disc_factor = 0.99
B_r = np.amax(g)

def expected_return(policy, noise_distribution):
    
    #J_0^mu(x,y) = 0 for all (x,y) in X
    J_prev = np.zeros(g.shape)

    #Number of iterations
    N = int( np.log( 0.1 * (1-disc_factor)/B_r )/ np.log(disc_factor) )

    for _ in range(N):
        J_now = np.zeros(J_prev.shape)

        for state in state_space:
                u = policy(state)
                
                for prob, w in noise_distribution:
                    r = evaluate_reward(state, u, w)
                    next_state = update(state, u, w)
                    val = r + disc_factor*J_prev[next_state]
                    J_now[state] += prob * val
                    
        #For next iteration
        J_prev = J_now
    
    print(J_prev)  


if __name__ == "__main__":
    print('Expected return of policy in Deterministic domain')
    expected_return(right_policy, noise_distribution=np.array([[1.0, 0.0],]))
    
    print('\nExpected return of policy in Stochastic domain')
    expected_return(right_policy, noise_distribution=np.array([[0.5, 0.0],[0.5, 1.0],]))
