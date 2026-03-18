from model.model import MarketModel

model = MarketModel(N_noise=100)

for i in range(50):
    model.step()
    print(f"Step {i}: Price = {model.price}")