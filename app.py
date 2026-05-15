from mesa.visualization import SolaraViz, make_plot_component, Slider
from model.model import MarketModel

# Calibracion base elegida tras validar heterogeneidad y cascadas
CAPITAL_SCALE = 1.0
POSITION_SCALE = 0.25
BASE_CAPITAL_MIN = 50
BASE_CAPITAL_MAX = 150
BASE_POSITION_MIN = 5
BASE_POSITION_MAX = 15

model_params = {
    "N_noise": Slider("Noise traders", 50, 10, 1000, 10),
    "N_traders": Slider("Leveraged traders", 50, 10, 300, 10),

    "sigma_noise": Slider("Noise volatility", 0.5, 0.1, 2.0, 0.1),
    "lambda_": Slider("Market impact λ", 0.01, 0.001, 0.05, 0.001),
    "alpha": Slider("Impact nonlinearity α", 0.5, 0.1, 1.0, 0.05),

    "theta": Slider("Liquidation intensity", 0.5, 0.1, 1.0, 0.1),
    "theta_spread": Slider("Theta spread", 0.05, 0.0, 0.2, 0.01),
    "margen_mantenimiento": 0.25,
    "margen_mantenimiento_spread": Slider("Margin spread", 0.02, 0.0, 0.1, 0.01),

    "capital_min": BASE_CAPITAL_MIN,
    "capital_max": BASE_CAPITAL_MAX,
    "capital_scale": Slider("Capital scale", CAPITAL_SCALE, 0.5, 1.5, 0.05),

    "position_min": BASE_POSITION_MIN,
    "position_max": BASE_POSITION_MAX,
    "position_scale": Slider("Position scale", POSITION_SCALE, 0.1, 0.6, 0.05),
}

components = [
    make_plot_component(["Price"]),
    make_plot_component(["AvgLeverage"]),
    make_plot_component(["MarginCalls"]),
    make_plot_component(["Liquidations"]),
    make_plot_component(["ActiveTraders"]),
]

initial_model = MarketModel(
    capital_scale=CAPITAL_SCALE,
    capital_min=BASE_CAPITAL_MIN,
    capital_max=BASE_CAPITAL_MAX,
    position_scale=POSITION_SCALE,
    position_min=BASE_POSITION_MIN,
    position_max=BASE_POSITION_MAX,
    margen_mantenimiento=0.25,
    theta_spread=0.05,
    margen_mantenimiento_spread=0.02,
)

Page = SolaraViz(
    initial_model,
    components=components,
    model_params=model_params,
)
