from model.model import MarketModel

print("\n" + "=" * 70)
print("TEST MICRO 1: update_capital()")
print("=" * 70)

model = MarketModel(N_noise=0, N_traders=1)
trader = model.leveraged_traders[0]

trader.capital = 100.0
trader.position = 10.0
trader.prev_price = 100.0
model.market.price = 95.0

print(f"Antes: capital={trader.capital:.2f}, prev_price={trader.prev_price:.2f}")
trader.update_capital()

print(f"Despues: capital={trader.capital:.2f}, prev_price={trader.prev_price:.2f}")
print("Esperado: capital = 100 + 10 * (95 - 100) = 50")
print("=" * 70 + "\n")
