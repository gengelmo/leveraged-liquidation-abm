from mesa import Agent
import numpy as np


class NoiseTrader(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def generate_order(self):
        direction = np.random.choice([-1, 1])
        size = abs(np.random.normal(0, self.model.sigma_noise))

        order = direction * size

        self.model.order_flow += order