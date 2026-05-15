from mesa import Agent
import numpy as np


class NoiseTrader(Agent):
    """
    Agente que genera órdenes aleatorias de compra o venta.

    Representa ruido exógeno del mercado, como pequeños inversores,
    flujos de liquidez o reequilibrios aleatorios de cartera.
    """

    def __init__(self, unique_id, model, sigma_noise):
        super().__init__(model)
        self.unique_id = unique_id
        self.sigma_noise = sigma_noise

    def generate_order(self):
        """
        Genera una orden aleatoria.

        La dirección puede ser compra (+1) o venta (-1).
        El tamaño de la orden se extrae de una normal en valor absoluto.
        """
        direction = np.random.choice([-1, 1])
        size = abs(np.random.normal(0, self.sigma_noise))

        return direction * size