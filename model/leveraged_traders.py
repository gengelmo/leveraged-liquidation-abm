from mesa import Agent
import numpy as np

class LeveragedTrader(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        
        self.capital = np.random.uniform(50, 150)
        self.position = np.random.uniform(5, 15)
        self.margin_call = False
        self.margen_mantenimiento = 0.25
        self.active = True

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
        # si ya no tiene posición → no participa
        if self.position <= 1e-6:
            self.margin_call = False
            return

        if self.capital < self.margen_mantenimiento * self.value:
            self.margin_call = True
        else:
            self.margin_call = False
    
    def liquidate(self, theta=0.5):
        sell_amount = theta * self.position

        self.position -= sell_amount

        # si ya está completamente liquidado
        if self.position <= 0.01:
            self.position = 0.0
            self.active = False
        
        return -sell_amount