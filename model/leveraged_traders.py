from mesa import Agent
import numpy as np

class LeveragedTrader(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        self.capital = np.random.uniform(50, 150)
        self.position = np.random.uniform(0, 5)

        self.prev_price = model.price

    def update_capital(self):
        price_change = self.model.price - self.prev_price
        self.capital += self.position * price_change
        self.prev_price = self.model.price