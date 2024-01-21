from collections import defaultdict
import numpy as np


class Blackjack:
    """
    A class representing a Blackjack agent using on-policy first-visit Monte Carlo control for ε-soft policies.

    This implementation is based on the methods described in Barto's Reinforcement Learning book. The agent learns
    an optimal policy for playing Blackjack, where decisions are based on the current state of the game. The state
    is defined by the sum of the player's cards, the dealer's showing card, and whether the player has a usable ace.

    Attributes:
        gamma (float): Discount factor for future rewards (default is 1.0).
        epsilon (float): Probability for exploration in ε-soft policy (default is 0.05).
        actions (list): List of possible actions; 0 for hit, 1 for stick.
        n_actions (int): Number of possible actions.
        policy (numpy.ndarray): Initial policy, updated during learning.
        player_hand (list): Cards in the player's hand.
        dealer_showing_card (int): The dealer's visible card.
        dealer_hand (list): Cards in the dealer's hand.
        state_action_function (dict): State-action value function.

    Methods:
        initialize_policy(): Initializes the policy randomly.
        draw_card(): Simulates drawing a card from the deck.
        has_usable_ace(hand): Checks if the hand contains a usable ace.
        sum_cards(hand): Returns the sum of the cards in the hand.
        calculate_state(): Returns the current state based on the player's and dealer's hands.
        get_action(state): Chooses an action based on the current state.
        get_state_value_function(): Computes the state value function from the state-action function.
        calculate_reward(): Calculates the reward after a game round.
        play_hand(action, state): Simulates playing a hand in the game.
        select_greedy_action(state): Selects the best action based on the current policy.
        update_policy(state): Updates the policy based on the state-action values.
        on_policy_first_visit_mc(n_episodes): Performs the on-policy first-visit Monte Carlo control.

    Example:
        >>> blackjack_agent = Blackjack()
        >>> state_action_function, policy = blackjack_agent.on_policy_first_visit_mc(n_episodes=100000)
        # Use the learned state_action_function and policy for decision making in Blackjack.
    """

    def __init__(self, gamma=1.0, epsilon=0.05):
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [0, 1]  # hit, stick
        self.n_actions = len(self.actions)
        self.policy = None
        self.player_hand = None
        self.dealer_showing_card = None
        self.dealer_hand = None
        self.state_action_function = None

    def initialize_policy(self):
        self.policy = np.random.random((22, 11, 2))

    def draw_card(self):
        return min(np.random.randint(1, 14), 10)

    def has_usable_ace(self, hand):
        if 1 in hand and sum(hand) + 10 <= 21:
            return 1
        return 0

    def sum_cards(self, hand):
        if self.has_usable_ace(hand):
            return sum(hand) + 10
        return sum(hand)

    def calculate_state(self):
        return (
            self.sum_cards(self.player_hand),
            self.dealer_showing_card,
            self.has_usable_ace(self.player_hand),
        )

    def get_action(self, state):
        return np.random.choice(
            self.actions, p=[self.policy[state], 1 - self.policy[state]]
        )

    def get_state_value_function(self):
        state_value_function = np.zeros((22, 11, 2))

        for player_sum in range(0, 22):
            for dealer_card in range(0, 11):
                for usable_ace in [0, 1]:
                    state = player_sum, dealer_card, usable_ace
                    action_values = []

                    for action in self.actions:
                        action_values.append(
                            self.state_action_function.get((state, action), 0)
                        )

                    state_value_function[state] = max(action_values)

        return state_value_function

    def calculate_reward(self):
        player_sum = self.sum_cards(self.player_hand)
        dealer_sum = self.sum_cards(self.dealer_hand)

        if dealer_sum > 21 or player_sum > dealer_sum:
            return 1
        if player_sum < dealer_sum:
            return -1
        return 0

    def play_hand(self, action, state):
        done = False

        if not action:
            self.player_hand.append(self.draw_card())

            if self.sum_cards(self.player_hand) > 21:
                return self.calculate_state(), -1, True

            return self.calculate_state(), 0, False

        while self.sum_cards(self.dealer_hand) < 17:
            self.dealer_hand.append(self.draw_card())

        state = self.calculate_state()
        reward = self.calculate_reward()
        done = True

        return state, reward, done

    def select_greedy_action(self, state):
        action_values = []

        for action in self.actions:
            action_values.append(self.state_action_function[(state, action)])

        action_values = np.array(action_values)
        best_action = self.actions[
            np.argmax(
                np.random.random(action_values.shape[0])
                * (action_values == action_values.max())
            )
        ]

        return best_action

    def update_policy(self, state):
        best_action = self.select_greedy_action(state)

        policy = self.epsilon / self.n_actions

        if best_action == 0:
            policy += 1 - self.epsilon

        return policy

    def on_policy_first_visit_mc(self, n_episodes=100000):
        self.initialize_policy()
        self.state_action_function = defaultdict(float)
        n_visited = defaultdict(int)

        for _ in range(0, n_episodes):
            episode = []

            self.player_hand = [self.draw_card(), self.draw_card()]
            self.dealer_showing_card = self.draw_card()
            self.dealer_hand = [self.dealer_showing_card]

            done = False

            while not done:
                state = self.calculate_state()
                action = self.get_action(state)
                next_state, reward, done = self.play_hand(action, state)
                episode.append((state, action, reward))
                state = next_state

            G = 0

            for t, (state, action, reward) in reversed(list(enumerate(episode))):
                G = self.gamma * G + reward

                if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                    n_visited[(state, action)] += 1
                    self.state_action_function[(state, action)] += (
                        G - self.state_action_function[(state, action)]
                    ) / (n_visited[(state, action)])
                    self.policy[state] = self.update_policy(state)

        for state, action in n_visited.keys():
            self.policy[state] = self.select_greedy_action(state)

        return self.state_action_function, self.policy
