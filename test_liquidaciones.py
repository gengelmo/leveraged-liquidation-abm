from model.model import MarketModel

print("\n" + "="*70)
print("🧪 TEST 2: EL CRASH MANUAL (Validación Unitaria del Margin Call)")
print("="*70)

# Instanciamos: Cero ruido y solo 1 trader apalancado
model = MarketModel(N_noise=0, N_traders=1)
trader = model.leveraged_traders[0]

# --- PARAMETRIZAMOS AL TRADER AL LÍMITE ---
# Le damos 50 de capital y 15 de posición (precio inicial es 100)
# Valor de la posición = 1500. Apalancamiento inicial = 1500/50 = 30x
trader.capital = 50.0
trader.position = 15.0
trader.prev_price = model.market.price 
model.margen_mantenimiento = 0.25 # Necesita mantener un 25% de la posición en capital

print("ESTADO INICIAL DEL TRADER:")
print(f"  Capital Propio: {trader.capital:.2f}")
print(f"  Apalancamiento: {trader.leverage:.2f}")

print("\n🔥 PROVOCANDO UN SHOCK EXÓGENO (-2% en el precio)...")
# Tiramos el precio ligeramente. Al estar apalancado x30, un 2% le dolerá mucho.
model.market.price *= 0.98 
print(f"  Precio de 100.00 -> alterado a {model.market.price:.2f}")

print("\n⏳ EJECUTANDO STEP (El mercado procesa el shock)...")
model.step()

print("\nESTADO TRAS EL STEP:")
print(f"  ¿Sufrió Margin Call?: {trader.margin_call}")
print(f"  Capital restante: {trader.capital:.2f}")
print(f"  Posición vendida forzosamente: {model.market.last_liquidations:.4f}")
print(f"  Nuevo Apalancamiento: {trader.leverage:.2f}")

print("\n💥 IMPACTO DE LA LIQUIDACIÓN:")
print(f"  Precio final del mercado: {model.market.price:.4f}")
print("="*70 + "\n")