import numpy as np
from collections import defaultdict

class Blackjack:
    def __init__(self, gamma=1.):
        self.gamma = gamma
    
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
        self.policy[20:22, :, : ] = 1 
        
    def calculate_state(self):
        return (self.sum_cards(self.player_hand), self.dealer_showing_card, self.has_usable_ace(self.player_hand))
        
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
            else:
                return self.calculate_state(), 0, False
            
        else:
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
        
        for i in range(0, n_episodes):            
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
                G = self.gamma*G + reward 
                
                if not state in [x[0] for x in episode[:t]]:
                    returns[state].append(G)
                    self.v[state] = np.mean(returns[state])
                    
        return self.v
