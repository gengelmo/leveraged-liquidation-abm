from mesa.visualization import SolaraViz, make_plot_component, Slider
from model.model import MarketModel

# Calibracion base elegida tras validar heterogeneidad y cascadas
POSITION_SCALE = 0.25
BASE_CAPITAL_MIN = 50
BASE_CAPITAL_MAX = 150
BASE_POSITION_MIN = 5
BASE_POSITION_MAX = 15

model_params = {
    "N_noise": Slider("Noise traders", 50, 10, 300, 10),
    "N_traders": Slider("Leveraged traders", 50, 10, 300, 10),

    "sigma_noise": Slider("Noise volatility", 0.5, 0.1, 2.0, 0.1),
    "lambda_": Slider("Market impact λ", 0.01, 0.001, 0.05, 0.001),
    "alpha": Slider("Impact nonlinearity α", 0.5, 0.1, 1.0, 0.05),

    "theta": Slider("Liquidation intensity", 0.5, 0.1, 1.0, 0.1),
    "margen_mantenimiento": 0.25,

    "capital_min": BASE_CAPITAL_MIN,
    "capital_max": BASE_CAPITAL_MAX,

    "position_min": BASE_POSITION_MIN * POSITION_SCALE,
    "position_max": BASE_POSITION_MAX * POSITION_SCALE,
}

components = [
    make_plot_component(["Price"]),
    make_plot_component(["AvgLeverage"]),
    make_plot_component(["MarginCalls"]),
    make_plot_component(["Liquidations"]),
    make_plot_component(["ActiveTraders"]),
]

initial_model = MarketModel(
    capital_min=BASE_CAPITAL_MIN,
    capital_max=BASE_CAPITAL_MAX,
    position_min=BASE_POSITION_MIN * POSITION_SCALE,
    position_max=BASE_POSITION_MAX * POSITION_SCALE,
    margen_mantenimiento=0.25,
)

Page = SolaraViz(
    initial_model,
    components=components,
    model_params=model_params,
)
