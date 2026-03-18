from mesa import Model
from mesa.time import RandomActivation
from model.agents import NoiseTrader
import numpy as np

class MarketModel(Model):
    def __init__(self, N_noise=50):
        super().__init__()

        self.price = 100.0
        self.order_flow = 0.0

        #parametros
        self.lambda_ = 0.01
        self.alpha = 0.5
        self.sigma_noise = 1

        self.schedule = RandomActivation(self)

        for i in range(N_noise):
            agent = NoiseTrader(i, self)
            self.schedule.add(agent)

    def compute_price_change(self):
        return self.lambda_ * np.sign(self.order_flow) * (abs(self.order_flow) ** self.alpha)

    def step(self):
        # Generar order flow aleatorio (simulacion de mercado)
        self.order_flow = 0.0

        self.schedule.step()

        # Calcular cambio de precio
        delta_p = self.compute_price_change()

        # Actualizar precio
        self.price = self.price * np.exp(delta_p)

