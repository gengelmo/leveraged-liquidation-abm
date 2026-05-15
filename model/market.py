import numpy as np


class Market:
    """
    Entorno de mercado donde se agregan las órdenes y se actualiza el precio.

    El precio se modifica mediante una función de impacto de mercado no lineal:

        delta_p = lambda * sign(order_flow) * |order_flow|^alpha

    y se actualiza de forma multiplicativa:

        P(t+1) = P(t) * exp(delta_p)

    Esto garantiza que el precio sea siempre positivo.
    """

    def __init__(self, lambda_=0.01, alpha=0.5, initial_price=100.0):
        self.price = initial_price
        self.lambda_ = lambda_
        self.alpha = alpha

        # Flujos de órdenes
        self.order_flow = 0.0
        self.order_flow_noise = 0.0
        self.order_flow_liq = 0.0

        # Históricos a nivel de actualización de precio
        self.price_history = [self.price]
        self.return_history = []
        self.flow_history = []

        # Métricas del último paso
        self.last_liquidations = 0.0

    def get_price(self):
        return self.price

    def reset_flows(self):
        """
        Reinicia los flujos de órdenes al comienzo de cada paso temporal.
        """
        self.order_flow = 0.0
        self.order_flow_noise = 0.0
        self.order_flow_liq = 0.0
        self.last_liquidations = 0.0

    def add_noise_order(self, order):
        """
        Añade una orden generada por un noise trader.
        """
        self.order_flow_noise += order

    def add_liquidation_order(self, order):
        """
        Añade una orden de liquidación forzada.

        Las liquidaciones se guardan en valor absoluto para medir
        el volumen total liquidado.
        """
        self.order_flow_liq += order
        self.last_liquidations += abs(order)

    def use_noise_flow(self):
        """
        Usa únicamente el flujo de órdenes de los noise traders.
        """
        self.order_flow = self.order_flow_noise

    def use_liquidation_flow(self):
        """
        Usa únicamente el flujo de órdenes procedente de liquidaciones.
        """
        self.order_flow = self.order_flow_liq

    def use_total_flow(self):
        """
        Usa el flujo agregado total.

        Este método queda disponible si se quiere procesar ruido y
        liquidaciones en una única fase.
        """
        self.order_flow = self.order_flow_noise + self.order_flow_liq

    def compute_price_change(self):
        """
        Calcula el impacto de mercado asociado al flujo de órdenes actual.
        """
        if self.order_flow == 0:
            return 0.0

        return (
            self.lambda_
            * np.sign(self.order_flow)
            * (abs(self.order_flow) ** self.alpha)
        )

    def update_price(self):
        """
        Actualiza el precio usando el flujo de órdenes activo.
        """
        old_price = self.price

        delta_p = self.compute_price_change()
        self.price *= np.exp(delta_p)

        simple_return = (self.price - old_price) / old_price

        self.price_history.append(self.price)
        self.return_history.append(simple_return)
        self.flow_history.append(self.order_flow)

        return simple_return