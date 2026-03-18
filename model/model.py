from mesa import Model
from mesa.time import RandomActivation
from model.noise_traders import NoiseTrader
from model.leveraged_traders import LeveragedTrader
import numpy as np

class MarketModel(Model):
    def __init__(self, N_noise=50, N_traders=20):
        super().__init__()

        self.price = 100.0
        self.order_flow = 0.0

        #parametros
        self.lambda_ = 0.01
        self.alpha = 0.5
        self.sigma_noise = 1

        self.schedule = RandomActivation(self)

        self.noise_traders = []
        self.leveraged_traders = []

        for i in range(N_noise):
            agent = NoiseTrader(i, self)
            self.noise_traders.append(agent)
            self.schedule.add(agent)

        # leveraged traders
        for i in range(N_traders):
            agent = LeveragedTrader(i + N_noise, self)
            self.leveraged_traders.append(agent)
            self.schedule.add(agent)

    def compute_price_change(self):
        return self.lambda_ * np.sign(self.order_flow) * (abs(self.order_flow) ** self.alpha)

    def step(self):
        # Generar order flow aleatorio (simulacion de mercado)
        self.order_flow = 0.0

        # Noise traders generan órdenes
        for agent in self.noise_traders:
            agent.generate_order()

        # Calcular cambio de precio
        delta_p = self.compute_price_change()

        # Actualizar precio
        self.price = self.price * np.exp(delta_p)

        # Traders actualizan capital
        for agent in self.leveraged_traders:
            agent.update_capital()
            agent.check_margin_call()