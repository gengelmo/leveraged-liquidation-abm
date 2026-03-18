from mesa import Model
import numpy as np

class MarketModel(Model):
    def __init__(self):
        super().__init__()

        self.price = 100.0
        self.order_flow = 0.0

        #parametros
        self.lambda_ = 0.01
        self.alpha = 0.5

    def compute_price_change(self):
        return self.lambda_ * np.sign(self.order_flow) * (abs(self.order_flow) ** self.alpha)

    def step(self):
        
        # Generar order flow aleatorio (simulacion de mercado)
        self.order_flow = np.random.normal(0,1) 

        # Calcular cambio de precio
        delta_p = self.compute_price_change()

        # Actualizar precio
        self.price = self.price * np.exp(delta_p)

