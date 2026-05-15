from mesa import Model
from model.market import Market
from model.noise_traders import NoiseTrader
from model.leveraged_traders import LeveragedTrader
from mesa.datacollection import DataCollector
import numpy as np


class MarketModel(Model):
    """
    Modelo basado en agentes de un mercado financiero con traders apalancados.

    El modelo combina:
    - noise traders, que generan fluctuaciones aleatorias;
    - leveraged traders, que sufren margin calls y liquidaciones forzadas;
    - un mercado, que transforma el flujo de órdenes en cambios de precio.

    Los traders apalancados mantienen posiciones largas. Cuando su capital
    cae por debajo del margen de mantenimiento requerido, liquidan la cantidad
    necesaria para volver a cumplir dicho margen al precio actual.
    """

    def __init__(
        self,
        N_noise=50,
        N_traders=50,
        sigma_noise=0.5,
        lambda_=0.01,
        alpha=0.5,
        margen_mantenimiento=0.25,
        margen_mantenimiento_spread=0.0,
        capital_min=50,
        capital_max=150,
        capital_scale=1.0,
        position_min=5,
        position_max=15,
        position_scale=1.0,
    ):
        super().__init__()

        self.market = Market(lambda_=lambda_, alpha=alpha)

        self.noise_traders = []
        self.leveraged_traders = []

        self.margin_calls_count = 0
        self.step_return = 0.0

        # Escalamos capital y posiciones manteniendo heterogeneidad individual.
        effective_capital_min = capital_min * capital_scale
        effective_capital_max = capital_max * capital_scale
        effective_position_min = position_min * position_scale
        effective_position_max = position_max * position_scale

        self.margen_mantenimiento_mean = margen_mantenimiento
        self.margen_mantenimiento_spread = margen_mantenimiento_spread

        def draw_bounded(mean, spread, low, high):
            """
            Extrae un valor normal acotado.

            Se usa para introducir heterogeneidad en el margen de mantenimiento
            sin permitir valores económicamente imposibles.
            """
            if spread <= 0:
                return float(np.clip(mean, low, high))
            return float(np.clip(np.random.normal(mean, spread), low, high))

        # Crear noise traders.
        for i in range(N_noise):
            agent = NoiseTrader(i, self, sigma_noise)
            self.noise_traders.append(agent)

        # Crear traders apalancados.
        for i in range(N_traders):
            trader_margin = draw_bounded(
                margen_mantenimiento,
                margen_mantenimiento_spread,
                0.01,
                0.99,
            )

            agent = LeveragedTrader(
                i + N_noise,
                self,
                margen_mantenimiento=trader_margin,
                capital_min=effective_capital_min,
                capital_max=effective_capital_max,
                position_min=effective_position_min,
                position_max=effective_position_max,
            )

            self.leveraged_traders.append(agent)

        # Exposición inicial agregada. Útil para calcular ratios de liquidación.
        self.initial_total_position = sum(
            agent.position for agent in self.leveraged_traders
        )

        self.datacollector = DataCollector(
            model_reporters={
                "Price": lambda m: m.market.price,
                "Returns": lambda m: m.step_return,
                "MarginCalls": lambda m: m.margin_calls_count,
                "Liquidations": lambda m: m.market.last_liquidations,
                "ActiveTraders": lambda m: sum(
                    agent.active for agent in m.leveraged_traders
                ),
                "AvgLeverage": lambda m: (
                    np.mean(
                        [
                            agent.leverage
                            for agent in m.leveraged_traders
                            if agent.active
                        ]
                    )
                    if any(agent.active for agent in m.leveraged_traders)
                    else 0
                ),
                "AvgCapital": lambda m: (
                    np.mean(
                        [
                            agent.capital
                            for agent in m.leveraged_traders
                            if agent.active
                        ]
                    )
                    if any(agent.active for agent in m.leveraged_traders)
                    else 0
                ),
                "TotalPosition": lambda m: sum(
                    agent.position
                    for agent in m.leveraged_traders
                    if agent.active
                ),
                "UniqueMarginCalled": lambda m: sum(
                    agent.ever_margin_called
                    for agent in m.leveraged_traders
                ),
                "InitialTotalPosition": lambda m: m.initial_total_position,
            }
        )

    def step(self):
        """
        Ejecuta un paso temporal del modelo.

        Cada paso se divide en dos fases:
        1. Fase exógena: los noise traders generan órdenes aleatorias.
        2. Fase endógena: las margin calls provocan liquidaciones forzadas.

        Esta separación permite observar cómo un shock inicial puede
        amplificarse mediante ventas forzadas.
        """
        start_price = self.market.price

        self.market.reset_flows()

        # 1. Noise traders generan órdenes aleatorias.
        for agent in self.noise_traders:
            order = agent.generate_order()
            self.market.add_noise_order(order)

        # 2. El flujo de ruido impacta en el precio.
        self.market.use_noise_flow()
        self.market.update_price()

        # 3. Los traders apalancados actualizan su capital tras el movimiento.
        for agent in self.leveraged_traders:
            if agent.active:
                agent.update_capital()

        # 4. Se comprueban las condiciones de margin call.
        self.margin_calls_count = 0

        for agent in self.leveraged_traders:
            if agent.active:
                agent.check_margin_call()

                if agent.margin_call:
                    self.margin_calls_count += 1

        # 5. Los traders en margin call liquidan lo necesario para volver al margen.
        for agent in self.leveraged_traders:
            if agent.active and agent.margin_call:
                order = agent.liquidate()
                self.market.add_liquidation_order(order)

        # 6. Las liquidaciones tienen impacto adicional sobre el precio.
        self.market.use_liquidation_flow()
        self.market.update_price()

        # 7. Se actualiza de nuevo el capital tras el impacto de liquidaciones.
        for agent in self.leveraged_traders:
            if agent.active:
                agent.update_capital()

        # 8. Retorno total del paso temporal completo.
        self.step_return = (self.market.price - start_price) / start_price

        # 9. Se guardan las métricas del paso.
        self.datacollector.collect(self)