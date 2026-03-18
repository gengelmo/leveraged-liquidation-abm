from model.model import MarketModel

model = MarketModel()

for i in range(5):
    model.step()
    print(f"Step {i}: Price = {model.price}")