from mesa import Model
from mesa.time import RandomActivation
from model.noise_traders import NoiseTrader
from model.leveraged_traders import LeveragedTrader
import numpy as np

class MarketModel(Model):
    def __init__(self, N_noise=50, N_traders=50):
        super().__init__()

        self.price = 100.0
        self.order_flow = 0.0
        self.order_flow_noise = 0.0
        self.order_flow_liq = 0.0

        # metricas
        self.last_liquidations = 0.0

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

        # reset
        self.order_flow = 0.0
        self.order_flow_noise = 0.0
        self.order_flow_liq = 0.0
        self.last_liquidations = 0.0

        # Noise traders generan órdenes
        for agent in self.noise_traders:
            order = agent.generate_order()
            self.order_flow_noise += order

        # total order flow inicial
        self.order_flow = self.order_flow_noise

        # Actualización precio
        delta_p = self.compute_price_change()
        self.price = self.price * np.exp(delta_p)

        for agent in self.leveraged_traders:
            if agent.active:
                agent.update_capital()
                agent.check_margin_call()

        # Liquidaciones
        for agent in self.leveraged_traders:
            if agent.active and agent.margin_call:
                order = agent.liquidate()
                self.order_flow_liq += order
                self.last_liquidations += abs(order)

        # añadir liquidaciones al flujo total
        self.order_flow = self.order_flow_noise + self.order_flow_liq

        # Actualización de precio 
        delta_p = self.compute_price_change()
        self.price = self.price * np.exp(delta_p)