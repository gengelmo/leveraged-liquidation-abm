import numpy as np

class Market:
    def __init__(self, lambda_=0.01, alpha=0.5):
        self.price = 100.0
        self.lambda_ = lambda_
        self.alpha = alpha

        self.order_flow = 0.0
        self.order_flow_noise = 0.0
        self.order_flow_liq = 0.0

        self.price_history = []
        self.return_history = []
        self.flow_history = []

        # métricas
        self.last_liquidations = 0.0

    def get_price(self):
        return self.price
    
    def reset_flows(self):
        self.order_flow = 0.0
        self.order_flow_noise = 0.0
        self.order_flow_liq = 0.0
        self.last_liquidations = 0.0

    def add_noise_order(self, order):
        self.order_flow_noise += order

    def add_liquidation_order(self, order):
        self.order_flow_liq += order
        self.last_liquidations += abs(order)

    def compute_total_flow(self):
        self.order_flow = self.order_flow_noise + self.order_flow_liq

    def compute_price_change(self):
        return self.lambda_ * np.sign(self.order_flow) * (abs(self.order_flow) ** self.alpha)

    def update_price(self):
        old_price = self.price

        delta_p = self.compute_price_change()
        self.price *= np.exp(delta_p)

        self.price_history.append(self.price)

        r = (self.price - old_price) / old_price
        self.return_history.append(r)

        self.flow_history.append(self.order_flow)