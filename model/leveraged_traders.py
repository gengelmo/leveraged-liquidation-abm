from mesa import Agent
import numpy as np

class LeveragedTrader(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        self.capital = np.random.uniform(50, 150)
        self.position = np.random.uniform(5, 15)
        self.margin_call = False
        self.margen_mantenimiento = 0.25

        self.prev_price = model.price

    @property
    def value(self):
        return abs(self.position * self.model.price)

    @property
    def leverage(self):
        if self.capital > 0:
            return self.value / self.capital
        else:
            return np.inf

    def update_capital(self):
        price_change = self.model.price - self.prev_price
        self.capital += self.position * price_change
        self.prev_price = self.model.price

    def check_margin_call(self):
        if self.capital < self.margen_mantenimiento * self.value:
            self.margin_call = True
        else:
            self.margin_call = False