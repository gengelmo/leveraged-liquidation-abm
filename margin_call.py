from model.model import MarketModel

print("\n" + "=" * 70)
print("TEST MICRO 2: check_margin_call()")
print("=" * 70)

model = MarketModel(N_noise=0, N_traders=1)
trader = model.leveraged_traders[0]

model.market.price = 100.0
trader.capital = 40.0
trader.position = 10.0
trader.prev_price = 100.0
trader.margen_mantenimiento = 0.25

print(f"Capital={trader.capital:.2f}")
print(f"Valor posicion={trader.value:.2f}")
print(f"Umbral margin call={trader.margen_mantenimiento * trader.value:.2f}")

trader.check_margin_call()

print(f"Margin call activado? {trader.margin_call}")
print("Esperado: True, porque 40 < 0.25 * 1000 = 250")
print("=" * 70 + "\n")
