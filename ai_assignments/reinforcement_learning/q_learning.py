from .. environment import Environment, Outcome
import numpy as np


def eps_greedy(rng, qs, epsilon):
    # this function makes an epsilon greedy decision
    # it trades off exploration and exploitation
    # exploration: trying out new options that may lead
    #              to better outcomes in the future
    # exploitation: choosing the best option based on past experience
    if rng.uniform(0, 1) < epsilon:
        # Random action selection
        return rng.randint(len(qs))
    else:
        # Greedy action selection
        return np.argmax(qs)


class QLearning():
    def train(self, env: Environment):
        ########################################
        # please leave untouched
        rng = np.random.RandomState(1234)
        alpha = 0.2
        epsilon = 0.3
        gamma = env.get_gamma()
        n_episodes = 10000
        ########################################

        ########################################
        # initialize the Q-'table'
        Q = np.zeros((env.get_n_states(), env.get_n_actions()))
        ########################################

        # TODO #################################
        # Initialize the Q-'table'
        Q = np.zeros((env.get_n_states(), env.get_n_actions()))

        for episode in range(1, n_episodes + 1):
            state = env.reset()
            done = False

            while not done:
                # Choose action using epsilon-greedy policy
                action = eps_greedy(rng, Q[state], epsilon)

                # Take action and observe the next state and reward
                next_state, reward, done = env.step(action)

                # Q-learning update
                best_next_action = np.argmax(Q[next_state])
                Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

                state = next_state  # Update state for the next iteration

        # Compute deterministic policy from Q value function
        policy = np.zeros((env.get_n_states(), env.get_n_actions()), dtype=np.int64)
        policy[np.arange(len(policy)), np.argmax(Q, axis=1)] = 1

        # Compute state value function V from Q
        V = np.max(Q, axis=1)

        return Outcome(n_episodes, policy, V=V, Q=Q)
        ########################################

        ########################################

        # compute a deterministic policy from the Q value function
        policy = np.zeros((env.get_n_states(), env.get_n_actions()), dtype=np.int64)
        policy[np.arange(len(policy)), np.argmax(Q, axis=1)] = 1
        # the state value function V can be computed easily from Q
        # by taking the action that leads to the max future reward
        V = np.max(Q, axis=1)

        ########################################

        return Outcome(n_episodes, policy, V=V, Q=Q)
