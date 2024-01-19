from collections import defaultdict
import numpy as np


class Blackjack:
    """
    A class that implements the game of Blackjack for the purpose of demonstrating
    the first-visit Monte Carlo method in Reinforcement Learning, as described in
    Sutton and Barto's book.

    The game of Blackjack is simulated, where the goal is to achieve a card sum
    closer to 21 than the dealer's sum without going over 21. The player's policy
    is fixed, and the state values are estimated using the first-visit Monte Carlo
    method. This method involves averaging the returns from the first visits to
    each state in multiple simulated games.

    Attributes:
    -----------
    gamma : float
        The discount factor for future rewards (default is 1.0).

    Methods:
    --------
    draw_card():
        Draws a card with values between 1 and 10 (inclusive), where face cards
        are counted as 10.

    has_usable_ace(hand):
        Checks if the hand has a usable Ace (counted as 11 without busting).

    sum_cards(hand):
        Calculates the sum of the cards in a hand, considering the usable Ace.

    create_value_function():
        Initializes the state value function to zeros for all state combinations.

    create_policy():
        Initializes the player's policy, which is fixed throughout the simulation.

    calculate_state():
        Determines the current state based on the player's hand, dealer's showing
        card, and usability of Ace.

    calculate_reward():
        Calculates the reward based on the final sums of the player's and dealer's hands.

    play_hand(action, state):
        Plays a hand based on the given action and returns the new state, reward,
        and a boolean indicating if the game is done.

    first_visit_mc(n_episodes=10000):
        Runs the first-visit Monte Carlo simulation for a given number of episodes,
        updating the value function based on the returns from each episode.

    Example:
    --------
    blackjack = Blackjack()
    v = blackjack.first_visit_mc(n_episodes=5000)
    print(v)

    This will output the estimated state values after 5000 episodes of Blackjack
    played under the fixed policy.
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.v = None
        self.policy = None
        self.player_hand = None
        self.dealer_showing_card = None
        self.dealer_hand = None

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

    def create_value_function(self):
        self.v = np.zeros((22, 22, 2))

    def create_policy(self):
        self.policy = np.zeros((22, 22, 2), dtype=int)
        self.policy[20:22, :, :] = 1

    def calculate_state(self):
        return (
            self.sum_cards(self.player_hand),
            self.dealer_showing_card,
            self.has_usable_ace(self.player_hand),
        )

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

    def first_visit_mc(self, n_episodes=10000):
        self.create_value_function()
        self.create_policy()

        returns = defaultdict(list)

        for _ in range(0, n_episodes):
            episode = []

            self.player_hand = [self.draw_card(), self.draw_card()]
            self.dealer_showing_card = self.draw_card()
            self.dealer_hand = [self.dealer_showing_card]

            done = False

            while not done:
                state = self.calculate_state()
                action = self.policy[state]
                next_state, reward, done = self.play_hand(action, state)
                episode.append((state, action, reward))
                state = next_state

            G = 0

            for t, (state, action, reward) in reversed(list(enumerate(episode))):
                G = self.gamma * G + reward

                if not state in [x[0] for x in episode[:t]]:
                    returns[state].append(G)
                    self.v[state] = np.mean(returns[state])

        return self.v
