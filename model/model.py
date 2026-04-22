from mesa import Model
# from mesa.time import RandomActivation
from model.market import Market
from model.noise_traders import NoiseTrader
from model.leveraged_traders import LeveragedTrader
from mesa.datacollection import DataCollector
import numpy as np

class MarketModel(Model):
    def __init__(self, 
             N_noise=50, 
             N_traders=50,
             sigma_noise=0.5,
             lambda_=0.01,
             alpha=0.5,
             theta=0.5,
             margen_mantenimiento=0.25,
             capital_min=50,
             capital_max=150,
             position_min=5,
             position_max=15):
        super().__init__()

        self.market = Market(lambda_=lambda_, alpha=alpha)

        # self.schedule = RandomActivation(self)

        self.noise_traders = []
        self.leveraged_traders = []

        self.datacollector = DataCollector(
            model_reporters={
                "Price": lambda m: m.market.price,
                "MarginCalls": lambda m: m.margin_calls_count,
                "Liquidations": lambda m: m.market.last_liquidations,
                "Returns": lambda m: m.market.return_history[-1] if m.market.return_history else 0,
                "ActiveTraders": lambda m: sum(a.active for a in m.leveraged_traders),
                "AvgLeverage": lambda m: np.mean([a.leverage for a in m.leveraged_traders if a.active]) if any(a.active for a in m.leveraged_traders) else 0,
                "AvgCapital": lambda m: np.mean([a.capital for a in m.leveraged_traders if a.active]) if any(a.active for a in m.leveraged_traders) else 0,
            }
        )

        for i in range(N_noise):
            agent = NoiseTrader(i, self, sigma_noise)
            self.noise_traders.append(agent)
            # self.schedule.add(agent)

        for i in range(N_traders):
            agent = LeveragedTrader(
                i + N_noise,
                self,
                theta=theta,
                margen_mantenimiento=margen_mantenimiento,
                capital_min=capital_min,
                capital_max=capital_max,
                position_min=position_min,
                position_max=position_max
            )
            self.leveraged_traders.append(agent)
            # self.schedule.add(agent)

    def step(self):
        self.market.reset_flows()

        # 1. Noise traders → órdenes
        for agent in self.noise_traders:
            order = agent.generate_order()
            self.market.add_noise_order(order)

        # 2. impacto ruido → precio cambia
        self.market.order_flow = self.market.order_flow_noise
        self.market.update_price()

        # 3. actualizar capital 
        for agent in self.leveraged_traders:
            if agent.active:
                agent.update_capital()

        # 4. margin calls 
        self.margin_calls_count = 0
        for agent in self.leveraged_traders:
            if agent.active:
                agent.check_margin_call()
                if agent.margin_call:
                    self.margin_calls_count += 1

        # 5. liquidaciones
        for agent in self.leveraged_traders:
            if agent.active and agent.margin_call:
                order = agent.liquidate()
                self.market.add_liquidation_order(order)

        # 6. impacto liquidaciones
        self.market.order_flow = self.market.order_flow_liq
        self.market.update_price()

        # 7. actualizar capital otra vez (por el segundo movimiento)
        for agent in self.leveraged_traders:
            if agent.active:
                agent.update_capital()

        # 8. guardar datos
        self.datacollector.collect(self)
    

