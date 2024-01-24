from collections import defaultdict
import numpy as np


class Blackjack:
    """
    A class to represent a Blackjack game environment for reinforcement learning using the Off-Policy Monte Carlo Control method.

    Attributes:
        gamma (float): Discount factor for future rewards.
        actions (list): List of possible actions, where 0 represents 'hit' and 1 represents 'stick'.
        behavior_policy (numpy.ndarray): The behavior policy under which the agent operates, expressed as probabilities.
        target_policy (numpy.ndarray): The target policy that the agent is trying to learn.
        player_hand (list): Current cards in the player's hand.
        dealer_showing_card (int): Card shown by the dealer.
        dealer_hand (list): Current cards in the dealer's hand.
        state_action_function (defaultdict): A dictionary mapping state-action pairs to their values.

    Methods:
        initialize_target_policy(): Initializes the target policy randomly.
        initialize_behavior_policy(): Initializes the behavior policy to a 50% chance of hitting.
        draw_card(): Draws a card with value between 1 and 10.
        has_usable_ace(hand): Checks if the hand has a usable ace.
        sum_cards(hand): Returns the sum of the card values in the hand.
        calculate_state(): Calculates the current state based on player's hand and dealer's showing card.
        get_action(state): Chooses an action based on the behavior policy for a given state.
        get_state_value_function(): Calculates the state-value function for the current policy.
        calculate_reward(): Calculates the reward after a hand is played.
        play_hand(action, state): Plays a hand of Blackjack based on the given action and state.
        select_greedy_action(state): Selects the action with the highest value for a given state.
        off_policy_mc_control(n_episodes): Performs the Off-Policy Monte Carlo Control algorithm over a specified number of episodes.
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.actions = [0, 1]  # hit, stick
        self.behavior_policy = None
        self.target_policy = None
        self.player_hand = None
        self.dealer_showing_card = None
        self.dealer_hand = None
        self.state_action_function = None

    def initialize_target_policy(self):
        self.target_policy = np.random.choice(
            self.actions, size=(22, 11, 2)
        )  # Actions themselves, not probabilities

    def initialize_behavior_policy(self):
        self.behavior_policy = 0.5 * np.ones((22, 11, 2))  # Probability for hit only

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
            self.actions,
            p=[self.behavior_policy[state], 1 - self.behavior_policy[state]],
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

        best_action = self.actions[np.argmax(action_values)]

        return best_action

    def off_policy_mc_control(self, n_episodes=100000):
        self.initialize_target_policy()
        self.initialize_behavior_policy()
        self.state_action_function = defaultdict(float)
        C = defaultdict(float)

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
            W = 1

            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                C[(state, action)] += W
                self.state_action_function[(state, action)] += (
                    W / C[(state, action)]
                ) * (G - self.state_action_function[(state, action)])
                self.target_policy[state] = self.select_greedy_action(state)

                if action != self.target_policy[state]:
                    break

                if action == 0:
                    b = self.behavior_policy[state]
                else:
                    b = 1 - self.behavior_policy[state]

                W *= 1.0 / b

        return self.state_action_function, self.target_policy
