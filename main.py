from model.model import MarketModel
from model.leveraged_traders import LeveragedTrader
model = MarketModel()

for i in range(5):
    model.step()
    print(f"\nStep {i}: Price = {model.price}")


    for i, agent in enumerate(model.leveraged_traders[:3]):
        print(
        f"Trader {i} | "
        f"cap={agent.capital:.2f} | "
        f"lev={agent.leverage:.2f} | "
        f"MC={agent.margin_call}"
    )