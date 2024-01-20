from collections import defaultdict
import numpy as np


class Blackjack:
    """
    A Blackjack game simulator implementing the Exploring Starts Monte Carlo method for Reinforcement Learning,
    as described in Sutton and Barto's book.
    This simulator models the game of Blackjack, where the goal is to maximize the sum of card values without exceeding 21.
    It uses the Exploring Starts Monte Carlo method to estimate state-action values and develop an optimal playing policy.


    Attributes:
        gamma (float): Discount factor for future rewards.
        actions (list): Possible actions in the game, represented as [0, 1] for hit and stick.
        state_action_function (numpy array): Table for storing state-action values.
        policy (numpy array): Table representing the policy, mapping states to actions.
        player_hand (list): Current cards in the player's hand.
        dealer_showing_card (int): The card shown by the dealer.
        dealer_hand (list): Current cards in the dealer's hand.

    Methods:
        initialize_policy(): Initializes the policy randomly.
        draw_card(): Draws a card with values from 1 to 10 (with face cards being 10).
        has_usable_ace(hand): Checks if there is a usable ace in the hand.
        sum_cards(hand): Calculates the total value of the cards in hand.
        calculate_state(): Computes the current state of the game.
        get_action(state, is_first): Determines the action based on the current state and whether it's the first action.
        get_state_value_function(): Calculates the value function for each state.
        calculate_reward(): Computes the reward after the player's and dealer's hands are played.
        play_hand(action, state): Plays a hand based on the given action and state.
        select_greedy_action(state): Selects the best action based on the state-action function.
        exploring_starts_mc(n_episodes=100000): Runs the Exploring Starts Monte Carlo algorithm.
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.actions = [0, 1]  # hit, stick
        self.policy = None
        self.player_hand = None
        self.dealer_showing_card = None
        self.dealer_hand = None
        self.state_action_function = None

    def initialize_policy(self):
        self.policy = np.random.choice(self.actions, size=(22, 11, 2))

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

    def get_action(self, state, is_first):
        if is_first:
            return np.random.choice(self.actions)

        return self.policy[state]

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

    def exploring_starts_mc(self, n_episodes=100000):
        self.initialize_policy()
        self.state_action_function = defaultdict(float)
        n_visited = defaultdict(int)

        for _ in range(0, n_episodes):
            episode = []

            self.player_hand = [self.draw_card(), self.draw_card()]
            self.dealer_showing_card = self.draw_card()
            self.dealer_hand = [self.dealer_showing_card]

            done = False
            is_first = True

            while not done:
                state = self.calculate_state()
                action = self.get_action(state, is_first)
                is_first = False
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
                    self.policy[state] = self.select_greedy_action(state)

        return self.state_action_function, self.policy
