# ODMCP_project_PPO
Main project for the Optimal Decision Making for complex problem course


Algorithm for one iteration (according to the paper)

- need a policy network (with theta weights)
- need a value network (with phi weights)

compute N trajectories of T time steps
    - sample a random initial state
    - evolve in the environment with the current policy
    - compute the value function of all states (using the value network)
    - compute the "reward-to-go" for the policy baseline trick
    - estimate the advantage after each timesteps (no reward at state 0)
  
- keep a copy of the current policy weights (theta)

for each epoch (K) : (/!\ use minibatches of size M < NT)
    - compute the probability of an action given a state for the OLD policy network (or current at the first iteration ?)
    - compute the probability of an action given a state for the NEW policy network
    - compute the ration r
    - compute the clipping
    - compute the clipped loss

    - do an update based on the loss for the policy network
    
    - do an update based on the rewards for the value network ???


compute the loss
optimize the loss with K epochs, using minibatches of size M < NT



NOTE: eq10 and eq11 of the main paper seem to be wrong...
The paper cites Mni+16 for the first one but the paper doesn't give the same version

 
\begin{align*}
\text{Mni+16: }A_t &= -V(s_t) + \gamma^k V(s_{t+k}) + \sum_{i=0}^{k-1}\gamma^i r_{t+i} \\
\text{replacing $k$ per $T-1$ : }A_t &= -V(s_t) + \gamma^{T-t} V(s_{T}) + \sum_{i=0}^{T-t-1}\gamma^i r_{t+i} \\
\text{paper: }A_t &= -V(s_t) + \gamma^{T-t} V(s_{T}) + r_t + \gamma r_{t+1} + \ldots + \gamma^{T-t+1} r_{T-1}\\
&= -V(s_t) + \gamma^{T-t} V(s_{T}) + r + ???
\end{align*}


the definition of the loss also seem to have an issue :
    the clipped loss would account for A twice
    the understand on how to clip is easier with the Keras example


TODO: check the definition for the advantage estimator and build the sum expression

