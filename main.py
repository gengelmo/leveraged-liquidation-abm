from model.model import MarketModel
import numpy as np

model = MarketModel()

for step in range(10):
    model.step()

    print(f"\n===== Step {step} =====")
    print(f"Price: {model.price:.2f}")
    print(f"Order flow: {model.order_flow:.2f}")
    print(f"Noise flow: {model.order_flow_noise:.2f}")
    print(f"Liquidation flow: {model.order_flow_liq:.2f}")

    active_traders = sum(a.active for a in model.leveraged_traders)
    print(f"Active traders: {active_traders}")

    # margin calls
    mc_count = sum(a.margin_call for a in model.leveraged_traders if a.active)
    print(f"Margin Calls: {mc_count}")

    # liquidaciones (GLOBAL)
    print(f"Liquidations volume: {model.last_liquidations:.2f}")
    
    # algunos traders
    print("\nSample traders:")
    for i, agent in enumerate(model.leveraged_traders[:3]):
        print(
            f"Trader {i} | "
            f"cap={agent.capital:.2f} | "
            f"pos={agent.position:.2f} | "
            f"lev={agent.leverage:.2f} | "
            f"MC={agent.margin_call}"
        )