from mesa import Agent
import numpy as np


class NoiseTrader(Agent):
    def __init__(self, unique_id, model, sigma_noise):
        super().__init__(model)
        self.unique_id = unique_id
        self.sigma_noise = sigma_noise

    def generate_order(self):
        direction = np.random.choice([-1, 1])
        size = abs(np.random.normal(0, self.sigma_noise))
        return direction * size