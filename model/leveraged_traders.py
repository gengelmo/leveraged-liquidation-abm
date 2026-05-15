from mesa import Agent
import numpy as np


class LeveragedTrader(Agent):
    """
    Agente que representa un trader apalancado.

    Cada trader mantiene una posición larga en el activo y dispone de capital
    propio. Cuando el precio cambia, su capital se actualiza según las pérdidas
    o ganancias generadas por su posición.

    Si el capital cae por debajo del margen de mantenimiento requerido, el
    trader recibe una margin call y liquida la cantidad necesaria para volver
    a cumplir el margen al precio actual.

    Si el trader se queda sin capital o sin posición, deja de estar activo.
    """

    def __init__(
        self,
        unique_id,
        model,
        margen_mantenimiento,
        capital_min,
        capital_max,
        position_min,
        position_max,
    ):
        """
        Inicializa un trader apalancado.

        Parameters
        ----------
        unique_id : int
            Identificador único del agente.
        model : MarketModel
            Modelo principal al que pertenece el agente.
        margen_mantenimiento : float
            Margen mínimo que debe mantener el trader respecto al valor de
            su posición.
        capital_min : float
            Capital inicial mínimo.
        capital_max : float
            Capital inicial máximo.
        position_min : float
            Posición inicial mínima.
        position_max : float
            Posición inicial máxima.
        """
        super().__init__(model)
        self.unique_id = unique_id

        self.capital = np.random.uniform(capital_min, capital_max)
        self.position = np.random.uniform(position_min, position_max)

        self.margin_call = False
        self.ever_margin_called = False

        self.margen_mantenimiento = margen_mantenimiento
        self.active = True

        self.prev_price = model.market.price

    @property
    def value(self):
        """
        Devuelve el valor absoluto de la posición del trader.

        Returns
        -------
        float
            Valor de mercado de la posición actual.
        """
        return abs(self.position * self.model.market.price)

    @property
    def leverage(self):
        """
        Calcula el ratio de apalancamiento del trader.

        El apalancamiento se define como:

            leverage = valor de la posición / capital

        Returns
        -------
        float
            Ratio de apalancamiento. Si el capital es cero, devuelve infinito.
        """
        if self.capital > 0:
            return self.value / self.capital
        return np.inf

    def update_capital(self):
        """
        Actualiza el capital del trader tras un cambio de precio.

        La variación del capital depende de la posición mantenida y del cambio
        en el precio del activo:

            capital(t+1) = capital(t) + position * [price(t+1) - price(t)]

        Si el capital resultante es negativo, se trunca a cero.
        """
        price_change = self.model.market.price - self.prev_price
        self.capital += self.position * price_change
        self.capital = max(self.capital, 0)
        self.prev_price = self.model.market.price

        if self.capital <= 1e-6:
            self.capital = 0
            self.position = 0
            self.active = False
            self.margin_call = False

    def check_margin_call(self):
        """
        Comprueba si el trader incumple el requisito de margen.

        La margin call se activa cuando:

            capital < margen_mantenimiento * valor_posicion

        Si el trader ya no tiene posición o no está activo, no puede recibir
        margin call.
        """
        if not self.active or self.position <= 1e-6:
            self.margin_call = False
            return

        self.margin_call = self.capital < self.margen_mantenimiento * self.value

        if self.margin_call:
            self.ever_margin_called = True

    def liquidate(self):
        """
        Liquida posición si el trader está en margin call.

        El trader calcula cuál es la posición máxima que puede mantener
        cumpliendo el margen de mantenimiento al precio actual:

            target_position = capital / (margen_mantenimiento * price)

        Si su posición actual es mayor que ese valor, vende la diferencia.
        Por tanto, la liquidación intenta devolver al trader al límite de
        margen al precio observado en ese momento.

        Returns
        -------
        float
            Orden enviada al mercado. Es negativa porque representa una venta.
        """
        if not self.margin_call:
            return 0.0

        price = self.model.market.price

        if price <= 0 or self.capital <= 0:
            sell_amount = self.position
            self.position = 0
            self.capital = max(self.capital, 0)
            self.active = False
            self.margin_call = False
            return -sell_amount

        target_position = self.capital / (
            self.margen_mantenimiento * price
        )
        target_position = max(0, target_position)

        required_sale = max(0, self.position - target_position)

        sell_amount = min(self.position, required_sale)

        self.position -= sell_amount

        if self.position <= 1e-6 or self.capital <= 1e-6:
            self.position = 0
            self.capital = max(self.capital, 0)
            self.active = False
            self.margin_call = False

        return -sell_amount