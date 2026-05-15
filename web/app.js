"use strict";

const base = {
  capitalMin: 50,
  capitalMax: 150,
  positionMin: 5,
  positionMax: 15,
  maintenanceMargin: 0.25,
  maxHistory: 240
};

const sliders = {
  nNoise: { format: (v) => String(Math.round(v)) },
  sigmaNoise: { format: (v) => v.toFixed(1) },
  lambda: { format: (v) => v.toFixed(3) },
  alpha: { format: (v) => v.toFixed(2) },
  nTraders: { format: (v) => String(Math.round(v)) },
  marginSpread: { format: (v) => v.toFixed(2) },
  capitalScale: { format: (v) => v.toFixed(2) },
  positionScale: { format: (v) => v.toFixed(2) },
  speed: { format: (v) => String(Math.round(v)) }
};

const el = {};
let model;
let timer = null;
let running = true;

function byId(id) {
  return document.getElementById(id);
}

function uniform(min, max) {
  return min + Math.random() * (max - min);
}

function normal(mean = 0, sd = 1) {
  const u1 = Math.max(Math.random(), Number.EPSILON);
  const u2 = Math.random();
  return mean + sd * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function signedImpact(flow, lambda, alpha) {
  if (flow === 0) return 0;
  return lambda * Math.sign(flow) * Math.pow(Math.abs(flow), alpha);
}

function readParams() {
  return {
    nNoise: Number(el.nNoise.value),
    sigmaNoise: Number(el.sigmaNoise.value),
    lambda: Number(el.lambda.value),
    alpha: Number(el.alpha.value),
    nTraders: Number(el.nTraders.value),
    marginSpread: Number(el.marginSpread.value),
    capitalScale: Number(el.capitalScale.value),
    positionScale: Number(el.positionScale.value)
  };
}

function boundedDraw(mean, spread, min, max) {
  if (spread <= 0) return clamp(mean, min, max);
  return clamp(normal(mean, spread), min, max);
}

function createModel() {
  const params = readParams();
  const traders = Array.from({ length: params.nTraders }, () => {
    const capital = uniform(base.capitalMin * params.capitalScale, base.capitalMax * params.capitalScale);
    const position = uniform(base.positionMin * params.positionScale, base.positionMax * params.positionScale);
    return {
      capital,
      position,
      prevPrice: 100,
      margin: boundedDraw(base.maintenanceMargin, params.marginSpread, 0.01, 0.99),
      marginCall: false,
      active: true
    };
  });

  return {
    params,
    step: 0,
    price: 100,
    lastReturn: 0,
    marginCalls: 0,
    liquidations: 0,
    totalMarginCalls: 0,
    totalLiquidations: 0,
    noiseFlow: 0,
    liqFlow: 0,
    traders,
    history: {
      price: [100],
      returns: [0],
      liquidations: [0],
      marginCalls: [0],
      active: [params.nTraders],
      leverage: [averageLeverage(traders, 100)]
    }
  };
}

function traderValue(trader, price) {
  return Math.abs(trader.position * price);
}

function traderLeverage(trader, price) {
  return trader.capital > 0 ? traderValue(trader, price) / trader.capital : Infinity;
}

function updateCapital(trader, price) {
  const priceChange = price - trader.prevPrice;
  trader.capital = Math.max(0, trader.capital + trader.position * priceChange);
  trader.prevPrice = price;
}

function averageLeverage(traders, price) {
  const active = traders.filter((trader) => trader.active);
  if (!active.length) return 0;
  return active.reduce((sum, trader) => sum + traderLeverage(trader, price), 0) / active.length;
}

function activeCount(traders) {
  return traders.reduce((sum, trader) => sum + Number(trader.active), 0);
}

function pushHistory(key, value) {
  model.history[key].push(value);
  if (model.history[key].length > base.maxHistory) {
    model.history[key].shift();
  }
}

function tick() {
  const params = model.params;
  model.step += 1;

  let noiseFlow = 0;
  for (let i = 0; i < params.nNoise; i += 1) {
    const direction = Math.random() < 0.5 ? -1 : 1;
    noiseFlow += direction * Math.abs(normal(0, params.sigmaNoise));
  }

  let oldPrice = model.price;
  model.price *= Math.exp(signedImpact(noiseFlow, params.lambda, params.alpha));

  for (const trader of model.traders) {
    if (trader.active) updateCapital(trader, model.price);
  }

  let marginCalls = 0;
  for (const trader of model.traders) {
    if (!trader.active || trader.position <= 1e-6) {
      trader.marginCall = false;
      continue;
    }

    trader.marginCall = trader.capital < trader.margin * traderValue(trader, model.price);
    if (trader.marginCall) marginCalls += 1;
  }

  let liqFlow = 0;

  for (const trader of model.traders) {
    if (!trader.active || !trader.marginCall) continue;

    const price = model.price;
    let sellAmount = 0;

    if (price <= 0 || trader.capital <= 20) {
      sellAmount = trader.position;
      trader.position = 0;
      trader.capital = Math.max(0, trader.capital);
      trader.active = false;
      trader.marginCall = false;
    } else {
      const targetPosition = Math.max(
        0,
        trader.capital / (trader.margin * price)
      );

      const requiredSale = Math.max(0, trader.position - targetPosition);

      // Vende exactamente lo necesario para volver al margen al precio actual.
      sellAmount = Math.min(trader.position, requiredSale);

      trader.position -= sellAmount;

      if (trader.position <= 1e-6 || trader.capital <= 1e-6) {
        trader.position = 0;
        trader.capital = Math.max(0, trader.capital);
        trader.active = false;
        trader.marginCall = false;
      }
    }

    liqFlow -= sellAmount;
  }

  model.price *= Math.exp(signedImpact(liqFlow, params.lambda, params.alpha));

  for (const trader of model.traders) {
    if (trader.active) updateCapital(trader, model.price);
  }

  model.lastReturn = (model.price - oldPrice) / oldPrice;
  model.marginCalls = marginCalls;
  model.liquidations = Math.abs(liqFlow);

  model.totalMarginCalls += marginCalls;
  model.totalLiquidations += Math.abs(liqFlow);

  model.noiseFlow = noiseFlow;
  model.liqFlow = liqFlow;

  pushHistory("price", model.price);
  pushHistory("returns", model.lastReturn);
  pushHistory("liquidations", model.liquidations);
  pushHistory("marginCalls", marginCalls);
  pushHistory("active", activeCount(model.traders));
  pushHistory("leverage", averageLeverage(model.traders, model.price));

  render();
}

function drawAxes(ctx, width, height, pad) {
  ctx.strokeStyle = "rgba(170, 179, 194, 0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();
}

function drawLineChart(canvas, series, options) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { left: 54, right: 20, top: 28, bottom: 34 };
  const xs = width - pad.left - pad.right;
  const ys = height - pad.top - pad.bottom;
  const min = options.min ?? Math.min(...series);
  const max = options.max ?? Math.max(...series);
  const span = max - min || 1;

  ctx.clearRect(0, 0, width, height);
  drawAxes(ctx, width, height, pad);

  ctx.fillStyle = "rgba(170, 179, 194, 0.76)";
  ctx.font = "12px system-ui";
  ctx.fillText(max.toFixed(options.decimals ?? 1), 10, pad.top + 4);
  ctx.fillText(min.toFixed(options.decimals ?? 1), 10, height - pad.bottom);

  if (series.length < 2) return;

  ctx.lineWidth = 3;
  ctx.strokeStyle = options.color;
  ctx.beginPath();
  series.forEach((value, index) => {
    const x = pad.left + (index / (series.length - 1)) * xs;
    const y = pad.top + (1 - (value - min) / span) * ys;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  if (options.fill) {
    ctx.lineTo(width - pad.right, height - pad.bottom);
    ctx.lineTo(pad.left, height - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = options.fill;
    ctx.fill();
  }
}

function drawBarChart(canvas, liquidations, calls) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { left: 46, right: 18, top: 24, bottom: 30 };
  const xs = width - pad.left - pad.right;
  const ys = height - pad.top - pad.bottom;
  const maxLiq = Math.max(1, ...liquidations);
  const maxCalls = Math.max(1, ...calls);
  const count = liquidations.length;
  const barWidth = Math.max(1, xs / count);

  ctx.clearRect(0, 0, width, height);
  drawAxes(ctx, width, height, pad);

  liquidations.forEach((value, index) => {
    const x = pad.left + index * barWidth;
    const h = (value / maxLiq) * ys;
    ctx.fillStyle = value > 0 ? "rgba(251, 113, 133, 0.8)" : "rgba(96, 165, 250, 0.14)";
    ctx.fillRect(x, height - pad.bottom - h, Math.max(1, barWidth - 1), h);
  });

  ctx.strokeStyle = "rgba(251, 191, 36, 0.95)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  calls.forEach((value, index) => {
    const x = pad.left + (index / Math.max(1, count - 1)) * xs;
    const y = pad.top + (1 - value / maxCalls) * ys;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawTraderMap(canvas) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const pad = { left: 46, right: 18, top: 24, bottom: 34 };
  const xs = width - pad.left - pad.right;
  const ys = height - pad.top - pad.bottom;
  const active = model.traders.filter((trader) => trader.active);
  const maxCapital = Math.max(1, ...model.traders.map((trader) => trader.capital));
  const maxLeverage = Math.max(4.5, ...active.map((trader) => traderLeverage(trader, model.price)));

  ctx.clearRect(0, 0, width, height);
  drawAxes(ctx, width, height, pad);

  ctx.fillStyle = "rgba(170, 179, 194, 0.76)";
  ctx.font = "12px system-ui";
  ctx.fillText("capital", width - 68, height - 10);
  ctx.fillText("lev", 12, 22);

  for (const trader of model.traders) {
    const leverage = traderLeverage(trader, model.price);
    const x = pad.left + (trader.capital / maxCapital) * xs;
    const y = pad.top + (1 - Math.min(leverage, maxLeverage) / maxLeverage) * ys;
    const stressed = trader.marginCall;
    const inactive = !trader.active;

    ctx.beginPath();
    ctx.arc(x, y, stressed ? 5 : 3.4, 0, Math.PI * 2);
    ctx.fillStyle = inactive
      ? "rgba(170, 179, 194, 0.18)"
      : stressed
        ? "rgba(251, 113, 133, 0.95)"
        : "rgba(45, 212, 191, 0.78)";
    ctx.fill();
  }
}

function renderMetrics() {
  el.price.textContent = model.price.toFixed(2);
  el.lastReturn.textContent = `${(model.lastReturn * 100).toFixed(2)}% ultimo paso`;

  el.marginCalls.textContent = String(model.marginCalls);
  el.marginCallsTotal.textContent =
    `Acumulado: ${Math.round(model.totalMarginCalls).toLocaleString("es-ES")}`;

  el.liquidations.textContent = model.liquidations.toFixed(2);
  el.liquidationsTotal.textContent =
    `Acumulado: ${model.totalLiquidations.toFixed(2)}`;

  el.avgLeverage.textContent = `${averageLeverage(model.traders, model.price).toFixed(2)}x`;
  el.activeTraders.textContent = `${activeCount(model.traders)} traders activos`;
  el.stepCount.textContent = `Paso ${model.step}`;
}

function render() {
  renderMetrics();
  drawLineChart(el.priceChart, model.history.price, {
    color: "#2dd4bf",
    fill: "rgba(45, 212, 191, 0.10)",
    decimals: 2
  });
  drawBarChart(el.liqChart, model.history.liquidations, model.history.marginCalls);
  drawTraderMap(el.traderMap);
}

function reset() {
  model = createModel();
  render();
}

function scheduleLoop() {
  if (timer) clearInterval(timer);
  const delay = 620 - Number(el.speed.value) * 45;
  timer = setInterval(() => {
    if (running) tick();
  }, Math.max(60, delay));
}

function bindControls() {
  Object.keys(sliders).forEach((id) => {
    const input = byId(id);
    el[id] = input;
    const output = input.parentElement.querySelector("output");

    if (output) {
      const updateOutput = () => {
        output.textContent = sliders[id].format(Number(input.value));
      };
      input.addEventListener("input", updateOutput);
      updateOutput();
    }
  });

  Object.keys(sliders)
    .filter((id) => id !== "speed")
    .forEach((id) => {
      el[id].addEventListener("change", reset);
    });

  el.speed.addEventListener("input", scheduleLoop);

  el.playPause.addEventListener("click", () => {
    running = !running;
    el.playPause.textContent = running ? "Pausar" : "Reanudar";
  });

  el.stepOnce.addEventListener("click", tick);
  el.reset.addEventListener("click", reset);
}

function initElements() {
  [
    "playPause",
    "stepOnce",
    "reset",
    "price",
    "lastReturn",
    "marginCalls",
    "marginCallsTotal",
    "liquidations",
    "liquidationsTotal",
    "activeTraders",
    "avgLeverage",
    "stepCount",
    "priceChart",
    "liqChart",
    "traderMap"
  ].forEach((id) => {
    el[id] = byId(id);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  initElements();
  bindControls();
  reset();
  scheduleLoop();
});
