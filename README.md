# Tabulated Reinforcement Learning
This project illustrates Reinforcement Learning methods in a simple Grid World environment. Within the environment, there are obstacles, absorbing states that terminate the agent with a negative reward, and a goal state, which gives a positive reward to the agent. The goal of the agent is to reach the goal state in as few steps as possible, since the reward is discounted at each step. In the project two reinforcement learning algorithms are implemented, Monte Carlo Learning and SARSA, and compared to optimisation algorithms, namely value and policy iteration. This project was developped as an assignment at Imperial College London, and the environment was provided.

### Optimisation - Policy Iteration Algorithm
The optimal value function and optimal policy was calculated with the policy iteration algorithm, since the process was Markovian. The algorithm performs policy evaluation and policy improvement iteratively until the policy converges and the optimal policy and the optimal value function is found. At the start of the policy iteration algorithm a policy is initialised in which action one is always chosen. Then policy evaluation is run with that policy to find the values of the states with that policy. The policy evaluation algorithm iterates until the biggest difference between the new value of a state and the old one is bigger than a given threshold, which was set to 0.0001. The values of the states are evaluated iteratively using Bellman’s equations. The policy improvement algorithm is then run with the value function produced by the policy evaluation algorithm and it chooses the action in every state that leads the agent to the highest value successor state. Then policy evaluation is run with the new policy and the algorithm iterates between policy evaluation and improvement until the policy converges and the optimal policy and optimal value function is found. The discount factor γ was set to γ=0.4

### Monte Carlo Learning
The optimal value function was estimated by implementing a First Visit Online Monte Carlo Iterative Optimisation algorithm, which relies on sampling only. The starting state for each simulation is chosen randomly out of the valid non-terminal states, which helps the agent explore the environment. In each iteration of the algorithm, a trace was produced with the current policy, which was then used to update the Q function (using online averaging) with values produced by the First Visit evaluation of the trace. After the Q function was updated from the trace a new ε-greedy policy is calculated from the updated Q function. A discount factor of γ=0.4 was used. The exploration parameter ε was dynamically set after each iteration with Equation 1 where k is the number of episodes. This was done to ensure that the agent explores all the possible actions in each state in the beginning of the algorithm, but as the number of iterations increase and the world is explored, it will tend to choose the optimal actions. This also ensures that the ε-greedy operation is GLIE, since ε tends to zero as the number of iterations tend to infinity.
> Equation 1 ε=1/k

The learning rate α was also dynamically set after each iteration using Equation 2 where k is the number of episodes. The learning rate controls the rate of forgetting old episodes.
> Equation 2 α=1/k                   

The implemented algorithm calculated an optimal policy and Q function, which was then used to calculate the optimal value function using Equation 3.
> Equation 3 V^π (s)=∑(a∈A)π(s,a)Q^π (s,a)

### SARSA (State-Action-Reward-State-Action)
Slide 207


## Getting Started
1. Clone the project and create a virtual environment
2. Install the required packages in the virtual environment
   ```
   pip3 install -r requirements.txt
   ```
