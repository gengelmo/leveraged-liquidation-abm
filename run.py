from model.model import MarketModel
from model.leveraged_traders import LeveragedTrader
model = MarketModel()

for i in range(5):
    model.step()
    print(f"\nStep {i}: Price = {model.price}")


    for i, agent in enumerate(model.leveraged_traders):
        print(f"Trader {i}: capital = {agent.capital:.2f}")