from mesa import Agent
import numpy as np

class LeveragedTrader(Agent):
    def __init__(self, unique_id, model, theta, margen_mantenimiento,
             capital_min, capital_max,
             position_min, position_max):
        super().__init__(model)
        self.unique_id = unique_id
        
        self.capital = np.random.uniform(capital_min, capital_max)
        self.position = np.random.uniform(position_min, position_max)
        self.margin_call = False
        self.theta = theta
        self.margen_mantenimiento = margen_mantenimiento
        self.active = True

        self.prev_price = model.market.price

    @property
    def value(self):
        return abs(self.position * self.model.market.price)

    @property
    def leverage(self):
        if self.capital > 0:
            return self.value / self.capital
        else:
            return np.inf

    def update_capital(self):
        price_change = self.model.market.price - self.prev_price
        self.capital += self.position * price_change
        self.capital = max(self.capital, 0)  
        self.prev_price = self.model.market.price

    def check_margin_call(self):
        # si ya no tiene posición -> no participa
        if self.position <= 1e-6:
            self.margin_call = False
            return

        if self.capital < self.margen_mantenimiento * self.value:
            self.margin_call = True
        else:
            self.margin_call = False
    
    def liquidate(self):
        excess_leverage = max(0, self.leverage - (1 / self.margen_mantenimiento))

        sell_amount = min(self.position, self.theta * excess_leverage * self.position)

        self.position -= sell_amount

        if self.position <= 1e-6:
            self.position = 0
            self.active = False

        return -sell_amount