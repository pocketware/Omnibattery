"""Temperature-driven heat pump consumption model (shadow forecast).

The household consumption profile has no temperature input: for a home with a
heat pump it lags a cold snap by days and scales the whole day equally. This
module splits the heat pump off:

* a **pure model** (no Home Assistant import is needed to use it) learns daily
  heat pump energy from the effective outdoor temperature, per mode:

  - ``hot_water``  — domestic hot water only: a constant level;
  - ``heating``    — ``intercept + slope * max(0, T_stop - T_eff)``;
  - ``cooling``    — ``intercept + slope * max(0, T_eff - T_start)``;

  where ``T_eff`` blends 70 % of the day's mean temperature with 30 % of the
  previous day's. The fits are robust (Huber IRLS) and every mode carries an
  explicit maturity contract (minimum days and minimum temperature spread);
* a **tracker** owns its own Store with about 120 days of daily samples, the
  live capture of the current day, the hourly temperature history, and the
  cached weather forecast. It backfills from Recorder *long-term statistics*
  (raw history is purged after about ten days) and refreshes the forecast on a
  throttled background task.

A **base-load profile** (``BaseLoadProfileTracker``) learns home consumption
minus heat pump power with the existing profile mechanism, in its own Store
next to the combined profile, which is never altered.

Shadow mode: nothing here is read by the planner or by any control decision.
Every entry point swallows its own failures; learning is best effort.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from time import monotonic
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util

from ..const import (
    CONF_HEAT_PUMP_COOLING_ALLOWED_ENTITY,
    CONF_HEAT_PUMP_HEATING_ALLOWED_ENTITY,
    CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR,
    CONF_HEAT_PUMP_POWER_SENSORS,
    CONF_HEAT_PUMP_START_COOLING_TEMP_ENTITY,
    CONF_HEAT_PUMP_STOP_HEATING_TEMP_ENTITY,
    CONF_HEAT_PUMP_WEATHER_ENTITY,
    DOMAIN,
)
from .backfill import BackfillToken, RecorderBackfillCoordinator, local_day_bounds
from .consumption_profile import (
    INTERVAL_COUNT,
    INTERVAL_SECONDS,
    INTERVALS_PER_HOUR,
    MAX_SAMPLE_GAP_SECONDS,
    ConsumptionProfileTracker,
    split_sample_across_bins,
)

_LOGGER = logging.getLogger(__name__)

MODE_HOT_WATER = "hot_water"
MODE_HEATING = "heating"
MODE_COOLING = "cooling"
MODES = (MODE_HOT_WATER, MODE_HEATING, MODE_COOLING)

# Effective daily temperature: the building and its heat pump react to the
# previous day as well (thermal mass, buffer tanks).
TEMP_BLEND_SAME_DAY = 0.7
DEFAULT_STOP_HEATING_C = 17.0
DEFAULT_START_COOLING_C = 24.0

# About 13 months: the previous winter stays in the fit while the current
# heating season has only a few days. Hourly long-term statistics cover it.
SAMPLE_RETENTION_DAYS = 400
# Quarter-hour shapes are only kept for recent days (the shape uses the last
# few weeks); older days keep their daily total, which bounds the Store size.
SHAPE_RETENTION_DAYS = 60
# A day counts when at least this share of its hours was measured; the energy
# is then scaled to the whole day.
DAY_MIN_COVERAGE = 0.9
DAY_MIN_TEMPERATURE_HOURS = 12

# Inference from energy alone: a day is "active" (heating or cooling) when it
# used clearly more than the hot water level. A weekly legionella cycle or a
# bath-heavy day stays inside this margin.
ACTIVE_MARGIN_KWH = 2.5
ACTIVE_MARGIN_RATIO = 0.5
# An *inferred* active day must also be clearly in its season: this far past
# the threshold, and with other active days close by (heating is switched for
# a season, not for one mild autumn day: at least half of the six days
# around it are active too). Otherwise it is "uncertain" and
# left out of every fit.
MIN_INFERRED_HEATING_DEGREES_K = 3.0
MIN_INFERRED_COOLING_DEGREES_K = 1.5
SEASON_NEIGHBOUR_DAYS = 3
SEASON_MIN_ACTIVE_NEIGHBOURS = 3
MODE_UNCERTAIN = "uncertain"

# Older days weigh less in the degree-mode fits, but never below the floor:
# with half a year of half-life and a floor, last winter still carries the
# slope while the new season has only a handful of days.
AGE_HALF_LIFE_DAYS = 365.0
AGE_WEIGHT_FLOOR = 0.35

# Maturity contract.
MIN_HOT_WATER_DAYS = 5
MIN_HEATING_DAYS = 10
MIN_HEATING_SPREAD_K = 4.0
MIN_COOLING_DAYS = 20
MIN_COOLING_SPREAD_K = 4.0

# Heat pump recent-days correction (its own, separate from the base load's).
HP_LEVEL_DAYS = 3
HP_LEVEL_MIN = 0.75
HP_LEVEL_MAX = 1.35
HP_LEVEL_MIN_EXPECTED_KWH = 1.0
RECENT_MODE_DAYS = 5

# Quarter-hour shape.
SHAPE_DAYS = 21
COLD_TILT = 0.5

# Weather forecast refresh.
WEATHER_REFRESH_S = 30 * 60
WEATHER_RETRY_S = 10 * 60
WEATHER_TIMEOUT_S = 30.0
MEASURED_HISTORY_HOURS = 72

STORE_VERSION = 1
STORE_KEY = "heat_pump_profile"
CAPTURE_VERSION = 1
SAVE_INTERVAL_S = 300
STATISTICS_CHUNK_DAYS = 30
BACKFILL_RETRY_S = 6 * 3600

HUBER_K = 1.345


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def effective_temperature(
    same_day_c: float | None, previous_day_c: float | None
) -> float | None:
    """Blend 70 % of the day with 30 % of the previous day."""
    if same_day_c is None:
        return previous_day_c
    if previous_day_c is None:
        return same_day_c
    return TEMP_BLEND_SAME_DAY * same_day_c + (1.0 - TEMP_BLEND_SAME_DAY) * previous_day_c


def heating_degrees(t_eff: float, stop_heating_c: float) -> float:
    return max(0.0, stop_heating_c - t_eff)


def cooling_degrees(t_eff: float, start_cooling_c: float) -> float:
    return max(0.0, t_eff - start_cooling_c)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty")
    position = (len(ordered) - 1) * min(1.0, max(0.0, q))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def huber_regression(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    prior_weights: Sequence[float] | None = None,
    iterations: int = 30,
) -> tuple[float, float, float]:
    """Robust ``y = a + b x`` with Huber weights (IRLS).

    Returns ``(intercept, slope, robust_mae)``. A handful of odd days (a
    holiday, a defrost-heavy day, a sensor glitch) barely move the line.
    ``prior_weights`` (for example by age) multiply the Huber weights.
    """
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError("no data")
    prior = list(prior_weights) if prior_weights is not None else [1.0] * n
    if len(prior) != n:
        raise ValueError("weights do not match the data")
    weights = list(prior)
    intercept = statistics.median(ys)
    slope = 0.0
    for _ in range(iterations):
        sw = sum(weights)
        if sw <= 0.0:
            break
        mx = sum(w * x for w, x in zip(weights, xs)) / sw
        my = sum(w * y for w, y in zip(weights, ys)) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
        sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(weights, xs, ys))
        new_slope = sxy / sxx if sxx > 1e-9 else 0.0
        new_intercept = my - new_slope * mx
        residuals = [y - (new_intercept + new_slope * x) for x, y in zip(xs, ys)]
        mad = statistics.median(abs(r) for r in residuals) * 1.4826
        scale = max(mad, 0.1)
        weights = [
            base * (1.0 if abs(r) <= HUBER_K * scale else HUBER_K * scale / abs(r))
            for base, r in zip(prior, residuals)
        ]
        converged = (
            abs(new_slope - slope) < 1e-6 and abs(new_intercept - intercept) < 1e-6
        )
        slope, intercept = new_slope, new_intercept
        if converged:
            break
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    mae = statistics.median(abs(r) for r in residuals)
    return intercept, slope, mae


@dataclass
class HeatPumpDay:
    """One measured local day of heat pump energy."""

    local_date: date
    energy_kwh: float
    temp_c: float | None = None
    coverage: float = 1.0
    # Explicit mode from the "allowed" entities, or None (inferred later).
    mode: str | None = None
    source: str = "statistics"
    quarters_kwh: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.local_date.isoformat(),
            "energy_kwh": round(self.energy_kwh, 4),
            "temp_c": None if self.temp_c is None else round(self.temp_c, 2),
            "coverage": round(self.coverage, 3),
            "mode": self.mode,
            "source": self.source,
            "quarters_kwh": (
                None
                if self.quarters_kwh is None
                else [round(value, 4) for value in self.quarters_kwh]
            ),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "HeatPumpDay | None":
        if not isinstance(raw, Mapping):
            return None
        try:
            local_date = date.fromisoformat(str(raw["date"]))
        except (KeyError, TypeError, ValueError):
            return None
        energy = _finite(raw.get("energy_kwh"))
        if energy is None or energy < 0.0:
            return None
        mode = raw.get("mode")
        quarters = raw.get("quarters_kwh")
        parsed_quarters: list[float] | None = None
        if isinstance(quarters, list) and len(quarters) == INTERVAL_COUNT:
            values = [_finite(value) for value in quarters]
            if all(value is not None and value >= 0.0 for value in values):
                parsed_quarters = [float(value) for value in values]  # type: ignore[arg-type]
        coverage = _finite(raw.get("coverage"))
        return cls(
            local_date=local_date,
            energy_kwh=energy,
            temp_c=_finite(raw.get("temp_c")),
            coverage=1.0 if coverage is None else min(1.0, max(0.0, coverage)),
            mode=mode if mode in MODES else None,
            source=str(raw.get("source") or "statistics")[:16],
            quarters_kwh=parsed_quarters,
        )


@dataclass
class ModeFit:
    """A fitted mode with its maturity verdict."""

    mode: str
    days: int = 0
    intercept_kwh: float = 0.0
    slope_kwh_per_k: float = 0.0
    temperature_spread_k: float = 0.0
    robust_mae_kwh: float | None = None
    mature: bool = False
    reason: str | None = "no_days"

    def as_dict(self) -> dict[str, Any]:
        return {
            "days": self.days,
            "intercept_kwh": round(self.intercept_kwh, 3),
            "slope_kwh_per_k": round(self.slope_kwh_per_k, 4),
            "temperature_spread_k": round(self.temperature_spread_k, 2),
            "robust_mae_kwh": (
                None if self.robust_mae_kwh is None else round(self.robust_mae_kwh, 3)
            ),
            "mature": self.mature,
            "reason": self.reason,
        }


@dataclass
class ClassifiedDay:
    day: HeatPumpDay
    t_eff: float
    mode: str
    mode_source: str  # "explicit" | "inferred"


@dataclass
class HeatPumpModel:
    """Fitted coefficients and the classified training days."""

    stop_heating_c: float
    start_cooling_c: float
    threshold_source: str
    hot_water: ModeFit
    heating: ModeFit
    cooling: ModeFit
    classified: list[ClassifiedDay] = field(default_factory=list)
    sample_days: int = 0

    def fit_for(self, mode: str) -> ModeFit:
        return {
            MODE_HOT_WATER: self.hot_water,
            MODE_HEATING: self.heating,
            MODE_COOLING: self.cooling,
        }[mode]

    def predict(self, t_eff: float, mode: str) -> float | None:
        """Daily heat pump energy for one mode, or None while it is immature."""
        fit = self.fit_for(mode)
        if not fit.mature:
            return None
        if mode == MODE_HOT_WATER:
            return max(0.0, fit.intercept_kwh)
        if mode == MODE_HEATING:
            x = heating_degrees(t_eff, self.stop_heating_c)
        else:
            x = cooling_degrees(t_eff, self.start_cooling_c)
        return max(0.0, fit.intercept_kwh + fit.slope_kwh_per_k * x)

    def recent_modes(self, before: date, days: int = RECENT_MODE_DAYS) -> list[str]:
        recent = sorted(
            (item for item in self.classified if item.day.local_date < before),
            key=lambda item: item.day.local_date,
            reverse=True,
        )[:days]
        return [item.mode for item in recent]

    def as_dict(self) -> dict[str, Any]:
        counts = {mode: 0 for mode in (*MODES, MODE_UNCERTAIN)}
        explicit = 0
        for item in self.classified:
            counts[item.mode] = counts.get(item.mode, 0) + 1
            explicit += item.mode_source == "explicit"
        return {
            "stop_heating_c": round(self.stop_heating_c, 2),
            "start_cooling_c": round(self.start_cooling_c, 2),
            "threshold_source": self.threshold_source,
            "sample_days": self.sample_days,
            "classified_days": counts,
            "explicit_mode_days": explicit,
            "age_half_life_days": AGE_HALF_LIFE_DAYS,
            "age_weight_floor": AGE_WEIGHT_FLOOR,
            "hot_water": self.hot_water.as_dict(),
            "heating": self.heating.as_dict(),
            "cooling": self.cooling.as_dict(),
        }


def _with_effective_temperatures(
    days: Iterable[HeatPumpDay],
) -> list[tuple[HeatPumpDay, float]]:
    by_date = {day.local_date: day for day in days}
    result: list[tuple[HeatPumpDay, float]] = []
    for local_date in sorted(by_date):
        day = by_date[local_date]
        if day.temp_c is None or day.coverage < DAY_MIN_COVERAGE:
            continue
        previous = by_date.get(local_date - timedelta(days=1))
        t_eff = effective_temperature(
            day.temp_c, previous.temp_c if previous is not None else None
        )
        if t_eff is not None:
            result.append((day, t_eff))
    return result


def _is_active(energy: float, hot_water_kwh: float) -> bool:
    return energy - hot_water_kwh > max(
        ACTIVE_MARGIN_KWH, ACTIVE_MARGIN_RATIO * hot_water_kwh
    )


def classify_days(
    days: Sequence[tuple[HeatPumpDay, float]],
    hot_water_kwh: float | None,
    *,
    stop_heating_c: float,
    start_cooling_c: float,
) -> list[ClassifiedDay]:
    """Assign each day a mode: the explicit one, else inferred from energy.

    A day at the hot water level is hot water only, however cold it was:
    heating was off (switched by hand), and such a day must not flatten the
    heating slope. A clearly higher day is heating below the midpoint of the
    two thresholds and cooling above it. Without any known hot water level
    (a whole season of heating), every day colder than the heating stop
    temperature counts as heating.
    """
    midpoint = (stop_heating_c + start_cooling_c) / 2.0
    active_dates = {
        day.local_date
        for day, _ in days
        if day.mode in (MODE_HEATING, MODE_COOLING)
        or (
            day.mode is None
            and hot_water_kwh is not None
            and _is_active(day.energy_kwh, hot_water_kwh)
        )
    }

    def in_season(local_date: date) -> bool:
        neighbours = sum(
            1
            for offset in range(-SEASON_NEIGHBOUR_DAYS, SEASON_NEIGHBOUR_DAYS + 1)
            if offset and local_date + timedelta(days=offset) in active_dates
        )
        return neighbours >= SEASON_MIN_ACTIVE_NEIGHBOURS

    result: list[ClassifiedDay] = []
    for day, t_eff in days:
        if day.mode in MODES:
            result.append(ClassifiedDay(day, t_eff, day.mode, "explicit"))
            continue
        if hot_water_kwh is None:
            mode = MODE_HEATING if t_eff < stop_heating_c else MODE_HOT_WATER
        elif _is_active(day.energy_kwh, hot_water_kwh):
            if t_eff < midpoint:
                clear = (
                    heating_degrees(t_eff, stop_heating_c)
                    >= MIN_INFERRED_HEATING_DEGREES_K
                )
                mode = MODE_HEATING
            else:
                clear = (
                    cooling_degrees(t_eff, start_cooling_c)
                    >= MIN_INFERRED_COOLING_DEGREES_K
                )
                mode = MODE_COOLING
            if not (clear and in_season(day.local_date)):
                # A mild day a little above the hot water level, or a lone
                # active day: not evidence for either line.
                mode = MODE_UNCERTAIN
        else:
            mode = MODE_HOT_WATER
        result.append(ClassifiedDay(day, t_eff, mode, "inferred"))
    return result


def _hot_water_anchor(
    days: Sequence[tuple[HeatPumpDay, float]],
    stop_heating_c: float,
    start_cooling_c: float,
) -> float | None:
    """A first hot water level, or None when the data cannot tell one.

    In order: days known to be hot water only; days between the two
    thresholds (no heating, no cooling); cold days that sit clearly below the
    heating line (heating switched off). A season of heating only has none of
    these, and guessing a percentile there would call every day hot water.
    """
    explicit = [day.energy_kwh for day, _ in days if day.mode == MODE_HOT_WATER]
    if len(explicit) >= 3:
        return statistics.median(explicit)
    neutral = [
        day.energy_kwh
        for day, t_eff in days
        if day.mode is None and stop_heating_c <= t_eff <= start_cooling_c
    ]
    if len(neutral) >= 3:
        return statistics.median(neutral)
    cold = [(day, t_eff) for day, t_eff in days if day.mode is None and t_eff < stop_heating_c]
    if len(cold) < MIN_HEATING_DAYS:
        return None
    xs = [heating_degrees(t_eff, stop_heating_c) for _, t_eff in cold]
    if max(xs) - min(xs) < 1e-6:
        return None
    intercept, slope, mae = huber_regression(xs, [day.energy_kwh for day, _ in cold])
    margin = max(ACTIVE_MARGIN_KWH, 3.0 * mae)
    low = [
        day.energy_kwh
        for (day, _), x in zip(cold, xs)
        if day.energy_kwh < intercept + slope * x - margin
    ]
    return statistics.median(low) if len(low) >= 3 else None


def age_weight(age_days: float) -> float:
    """Weight of a day ``age_days`` before the newest sample."""
    return max(AGE_WEIGHT_FLOOR, 0.5 ** (max(0.0, age_days) / AGE_HALF_LIFE_DAYS))


def _fit_degree_mode(
    mode: str,
    items: Sequence[ClassifiedDay],
    degrees: Callable[[float], float],
    min_days: int,
    min_spread: float,
    newest: date | None = None,
) -> ModeFit:
    fit = ModeFit(mode=mode, days=len(items))
    if not items:
        return fit
    temps = [item.t_eff for item in items]
    fit.temperature_spread_k = (
        _quantile(temps, 0.9) - _quantile(temps, 0.1) if len(temps) >= 2 else 0.0
    )
    xs = [degrees(item.t_eff) for item in items]
    ys = [item.day.energy_kwh for item in items]
    reference = newest or max(item.day.local_date for item in items)
    prior = [age_weight((reference - item.day.local_date).days) for item in items]
    if len(items) >= 2 and max(xs) - min(xs) > 1e-6:
        intercept, slope, mae = huber_regression(xs, ys, prior_weights=prior)
    else:
        intercept, slope, mae = statistics.median(ys), 0.0, None
    if slope < 0.0:
        # Physically impossible: more heating on milder days means the mode or
        # threshold is wrong. Fall back to a flat level, never a negative slope.
        intercept, slope = statistics.median(ys), 0.0
        fit.reason = "nonpositive_slope"
    fit.intercept_kwh = max(0.0, intercept)
    fit.slope_kwh_per_k = max(0.0, slope)
    fit.robust_mae_kwh = mae
    if len(items) < min_days:
        fit.reason = "insufficient_days"
    elif fit.temperature_spread_k < min_spread:
        fit.reason = "insufficient_temperature_spread"
    elif fit.slope_kwh_per_k <= 0.0:
        fit.reason = fit.reason or "nonpositive_slope"
    else:
        fit.reason = None
        fit.mature = True
    return fit


def fit_heat_pump_model(
    days: Iterable[HeatPumpDay],
    *,
    stop_heating_c: float | None = None,
    start_cooling_c: float | None = None,
) -> HeatPumpModel:
    """Fit the three-mode model on daily samples. Pure and deterministic."""
    threshold_source = "configured"
    if stop_heating_c is None or start_cooling_c is None:
        threshold_source = "default" if (
            stop_heating_c is None and start_cooling_c is None
        ) else "partly_default"
    stop = DEFAULT_STOP_HEATING_C if stop_heating_c is None else float(stop_heating_c)
    start = DEFAULT_START_COOLING_C if start_cooling_c is None else float(start_cooling_c)
    if start <= stop:
        # A misconfigured pair would make the two modes overlap.
        start = stop + 1.0

    usable = _with_effective_temperatures(days)
    model = HeatPumpModel(
        stop_heating_c=stop,
        start_cooling_c=start,
        threshold_source=threshold_source,
        hot_water=ModeFit(MODE_HOT_WATER),
        heating=ModeFit(MODE_HEATING),
        cooling=ModeFit(MODE_COOLING),
        sample_days=len(usable),
    )
    if not usable:
        return model
    hot_water = _hot_water_anchor(usable, stop, start)
    classified = classify_days(
        usable, hot_water, stop_heating_c=stop, start_cooling_c=start
    )
    for _ in range(3 if hot_water is not None else 0):
        classified = classify_days(
            usable, hot_water, stop_heating_c=stop, start_cooling_c=start
        )
        hw_days = [item.day.energy_kwh for item in classified if item.mode == MODE_HOT_WATER]
        if not hw_days:
            break
        refined = statistics.median(hw_days)
        if abs(refined - hot_water) < 0.05:
            hot_water = refined
            break
        hot_water = refined
    model.classified = classified

    hw_items = [item for item in classified if item.mode == MODE_HOT_WATER]
    hw_fit = ModeFit(MODE_HOT_WATER, days=len(hw_items))
    if hw_items:
        energies = [item.day.energy_kwh for item in hw_items]
        hw_fit.intercept_kwh = statistics.median(energies)
        hw_fit.robust_mae_kwh = statistics.median(
            abs(value - hw_fit.intercept_kwh) for value in energies
        )
        if len(hw_items) >= MIN_HOT_WATER_DAYS:
            hw_fit.mature = True
            hw_fit.reason = None
        else:
            hw_fit.reason = "insufficient_days"
    model.hot_water = hw_fit
    newest = max(day.local_date for day, _ in usable)
    model.heating = _fit_degree_mode(
        MODE_HEATING,
        [item for item in classified if item.mode == MODE_HEATING],
        lambda t: heating_degrees(t, stop),
        MIN_HEATING_DAYS,
        MIN_HEATING_SPREAD_K,
        newest,
    )
    model.cooling = _fit_degree_mode(
        MODE_COOLING,
        [item for item in classified if item.mode == MODE_COOLING],
        lambda t: cooling_degrees(t, start),
        MIN_COOLING_DAYS,
        MIN_COOLING_SPREAD_K,
        newest,
    )
    return model


def choose_forecast_mode(
    model: HeatPumpModel,
    t_eff: float,
    *,
    heating_allowed: bool | None,
    cooling_allowed: bool | None,
    recent_modes: Sequence[str],
) -> tuple[str, str]:
    """Mode for a forecast day: ``(mode, basis)``.

    With the "allowed" entities the switches decide; otherwise the recent
    days do (a cold spell with only hot water use means heating is off).
    """
    if heating_allowed is not None or cooling_allowed is not None:
        heat_on = bool(heating_allowed)
        cool_on = bool(cooling_allowed)
        basis = "allowed_entities"
    elif recent_modes:
        heat_on = MODE_HEATING in recent_modes
        cool_on = MODE_COOLING in recent_modes
        basis = "recent_days"
    else:
        heat_on = t_eff < model.stop_heating_c
        cool_on = False
        basis = "temperature"
    if heat_on and t_eff < model.stop_heating_c:
        return MODE_HEATING, basis
    if cool_on and t_eff > model.start_cooling_c:
        return MODE_COOLING, basis
    return MODE_HOT_WATER, basis


def recent_correction(
    model: HeatPumpModel, before: date, days: int = HP_LEVEL_DAYS
) -> float:
    """Median actual/predicted ratio over the last few classified days."""
    recent = sorted(
        (item for item in model.classified if item.day.local_date < before),
        key=lambda item: item.day.local_date,
        reverse=True,
    )[:days]
    ratios: list[float] = []
    for item in recent:
        if item.mode not in MODES:
            continue
        predicted = model.predict(item.t_eff, item.mode)
        if predicted is None or predicted < HP_LEVEL_MIN_EXPECTED_KWH:
            continue
        ratios.append(item.day.energy_kwh / predicted)
    if len(ratios) < days:
        return 1.0
    return min(HP_LEVEL_MAX, max(HP_LEVEL_MIN, statistics.median(ratios)))


def learn_shape(
    days: Iterable[HeatPumpDay],
    modes: Mapping[date, str],
    mode: str,
    *,
    before: date,
    max_days: int = SHAPE_DAYS,
) -> tuple[list[float], str]:
    """Average normalized quarter-hour shape of recent days of ``mode``.

    Falls back to recent days of any mode, then to a flat shape.
    """
    candidates = sorted(
        (
            day
            for day in days
            if day.local_date < before
            and day.quarters_kwh is not None
            and day.energy_kwh > 0.1
        ),
        key=lambda day: day.local_date,
        reverse=True,
    )
    for label, pool in (
        (mode, [day for day in candidates if modes.get(day.local_date) == mode]),
        ("any", candidates),
    ):
        pool = pool[:max_days]
        if len(pool) < 2:
            continue
        totals = [0.0] * INTERVAL_COUNT
        for day in pool:
            day_total = sum(day.quarters_kwh or ())
            if day_total <= 0.0:
                continue
            for index, value in enumerate(day.quarters_kwh or ()):
                totals[index] += value / day_total
        grand = sum(totals)
        if grand > 0.0:
            return [value / grand for value in totals], label
    return [1.0 / INTERVAL_COUNT] * INTERVAL_COUNT, "flat"


def distribute_day(
    energy_kwh: float,
    shape: Sequence[float],
    hourly_temps: Sequence[float | None],
    mode: str,
    *,
    stop_heating_c: float,
    start_cooling_c: float,
    tilt: float = COLD_TILT,
) -> list[float]:
    """Spread a daily total over 96 quarters, tilted to colder (hotter) hours."""
    energy = max(0.0, float(energy_kwh))
    weights = list(shape) if len(shape) == INTERVAL_COUNT else [1.0] * INTERVAL_COUNT
    degrees: list[float | None] = []
    if mode in (MODE_HEATING, MODE_COOLING) and len(hourly_temps) == 24:
        for temp in hourly_temps:
            if temp is None:
                degrees.append(None)
            elif mode == MODE_HEATING:
                degrees.append(heating_degrees(temp, stop_heating_c))
            else:
                degrees.append(cooling_degrees(temp, start_cooling_c))
        known = [value for value in degrees if value is not None]
        mean = sum(known) / len(known) if known else 0.0
        if mean > 0.0:
            weights = [
                weight
                * (
                    1.0
                    if degrees[index // INTERVALS_PER_HOUR] is None
                    else (1.0 - tilt) + tilt * degrees[index // INTERVALS_PER_HOUR] / mean
                )
                for index, weight in enumerate(weights)
            ]
    total = sum(weights)
    if total <= 0.0:
        return [energy / INTERVAL_COUNT] * INTERVAL_COUNT
    return [energy * weight / total for weight in weights]


def _row_start(row: Mapping[str, Any]) -> float | None:
    value = row.get("start")
    if isinstance(value, datetime):
        return value.timestamp()
    return _finite(value)


def aggregate_hourly_statistics(
    rows_by_id: Mapping[str, Sequence[Mapping[str, Any]]],
    power_ids: Sequence[str],
    temperature_id: str | None,
) -> tuple[dict[float, float], dict[float, float]]:
    """Hourly heat pump energy (kWh) and temperature (°C) by hour-start timestamp.

    Power statistics are hourly means in kW, so the mean *is* the hour's
    energy in kWh. An hour counts only when every configured power sensor has
    a value: a partial sum would read as a low-energy hour.
    """
    per_sensor: list[dict[float, float]] = []
    for entity_id in power_ids:
        values: dict[float, float] = {}
        for row in rows_by_id.get(entity_id, ()) or ():
            start = _row_start(row)
            mean = _finite(row.get("mean"))
            if start is None or mean is None:
                continue
            values[start] = max(0.0, mean)
        per_sensor.append(values)
    energy: dict[float, float] = {}
    if per_sensor:
        common = set(per_sensor[0])
        for values in per_sensor[1:]:
            common &= set(values)
        for start in common:
            energy[start] = sum(values[start] for values in per_sensor)
    temperature: dict[float, float] = {}
    if temperature_id:
        for row in rows_by_id.get(temperature_id, ()) or ():
            start = _row_start(row)
            mean = _finite(row.get("mean"))
            if start is not None and mean is not None:
                temperature[start] = mean
    return energy, temperature


def days_from_hourly(
    energy_by_hour: Mapping[float, float],
    temperature_by_hour: Mapping[float, float],
    local_dates: Iterable[date],
    tz: Any,
) -> dict[date, HeatPumpDay]:
    """Build complete daily samples from hourly energy and temperature."""
    result: dict[date, HeatPumpDay] = {}
    for local_date in local_dates:
        start, end = local_day_bounds(local_date, tz)
        hour = start.timestamp()
        end_ts = end.timestamp()
        expected = 0
        covered = 0
        energy = 0.0
        temps: list[float] = []
        quarters = [0.0] * INTERVAL_COUNT
        while hour < end_ts - 1e-6:
            expected += 1
            local = datetime.fromtimestamp(hour, tz)
            value = energy_by_hour.get(hour)
            if value is not None:
                covered += 1
                energy += value
                base = local.hour * INTERVALS_PER_HOUR
                for offset in range(INTERVALS_PER_HOUR):
                    quarters[base + offset] += value / INTERVALS_PER_HOUR
            temp = temperature_by_hour.get(hour)
            if temp is not None:
                temps.append(temp)
            hour += 3600.0
        if expected == 0 or covered / expected < DAY_MIN_COVERAGE:
            continue
        scale = expected / covered
        result[local_date] = HeatPumpDay(
            local_date=local_date,
            energy_kwh=energy * scale,
            temp_c=(
                sum(temps) / len(temps) if len(temps) >= DAY_MIN_TEMPERATURE_HOURS else None
            ),
            coverage=covered / expected,
            mode=None,
            source="statistics",
            quarters_kwh=[value * scale for value in quarters],
        )
    return result


def parse_allowed_state(state: Any) -> bool | None:
    """Interpret an "allowed" entity: on/true/1/heating → True, off/0 → False."""
    if state is None:
        return None
    raw = getattr(state, "state", state)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ("unknown", "unavailable", ""):
        return None
    if text in ("on", "true", "yes", "heating", "cooling", "auto", "enabled"):
        return True
    if text in ("off", "false", "no", "disabled"):
        return False
    number = _finite(text)
    if number is None:
        return None
    return number > 0.0


def parse_forecast_response(
    response: Any, entity_id: str
) -> list[tuple[float, float]]:
    """``weather.get_forecasts`` response → ``[(timestamp, temperature)]``."""
    if not isinstance(response, Mapping):
        return []
    block = response.get(entity_id)
    forecast = block.get("forecast") if isinstance(block, Mapping) else None
    result: list[tuple[float, float]] = []
    for item in forecast if isinstance(forecast, list) else []:
        if not isinstance(item, Mapping):
            continue
        temp = _finite(item.get("temperature"))
        when = item.get("datetime")
        if temp is None or when is None:
            continue
        if isinstance(when, datetime):
            moment = when
        else:
            try:
                moment = datetime.fromisoformat(str(when))
            except ValueError:
                continue
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        result.append((moment.timestamp(), temp))
    result.sort()
    return result


def _to_celsius(value: float, unit: Any) -> float:
    text = str(unit or "").strip().upper().replace("°", "")
    if text == "F":
        return (value - 32.0) * 5.0 / 9.0
    if text == "K":
        return value - 273.15
    return value


@dataclass
class HeatPumpDayForecast:
    local_date: date
    intervals_kwh: list[float]
    energy_kwh: float
    mode: str | None
    mode_basis: str | None
    t_eff: float | None
    day_temp_c: float | None
    temperature_source: str
    correction: float
    shape_source: str
    mature: bool
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.local_date.isoformat(),
            "energy_kwh": round(self.energy_kwh, 3),
            "mode": self.mode,
            "mode_basis": self.mode_basis,
            "effective_temperature_c": None if self.t_eff is None else round(self.t_eff, 2),
            "day_temperature_c": None if self.day_temp_c is None else round(self.day_temp_c, 2),
            "temperature_source": self.temperature_source,
            "correction": round(self.correction, 3),
            "shape_source": self.shape_source,
            "mature": self.mature,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Base-load profile (home minus heat pump)
# ---------------------------------------------------------------------------


def heat_pump_power_sensors(config_data: Mapping[str, Any]) -> list[str]:
    raw = config_data.get(CONF_HEAT_PUMP_POWER_SENSORS) or []
    if isinstance(raw, str):
        raw = [raw]
    result: list[str] = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        entity_id = str(item or "").strip()
        if entity_id and entity_id not in result:
            result.append(entity_id)
    return result


class BaseLoadProfileTracker(ConsumptionProfileTracker):
    """The 28-day profile of home consumption minus heat pump power.

    Same capture, training and Recorder backfill as the combined profile —
    only the input differs (the heat pump is subtracted for learning; the
    battery still covers it) and it lives in its own Store. The combined
    profile and its Store are untouched.
    """

    STORE_KEY = "consumption_profile_base"
    BACKFILL_JOB_NAME = "consumption_profile_base"

    def configuration_fingerprint(self) -> str:
        data = getattr(self._config_entry, "data", {}) or {}
        payload = {
            "combined": super().configuration_fingerprint(),
            "heat_pump_power_sensors": sorted(heat_pump_power_sensors(data)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _backfill_external_loads(
        config_data: dict[str, Any],
    ) -> list[tuple[str, float]]:
        loads = ConsumptionProfileTracker._backfill_external_loads(config_data)
        loads.extend(
            (entity_id, -1.0) for entity_id in heat_pump_power_sensors(config_data)
        )
        return loads


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class HeatPumpProfileTracker:
    """Capture, persist and query the heat pump model for one config entry."""

    def __init__(self, hass: Any, config_entry: Any, controller: Any = None) -> None:
        self._hass = hass
        self._config_entry = config_entry
        self._controller = controller
        data = getattr(config_entry, "data", {}) or {}
        self.power_sensors = heat_pump_power_sensors(data)
        self.temperature_sensor: str | None = data.get(CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR) or None
        self.weather_entity: str | None = data.get(CONF_HEAT_PUMP_WEATHER_ENTITY) or None
        self.heating_allowed_entity: str | None = (
            data.get(CONF_HEAT_PUMP_HEATING_ALLOWED_ENTITY) or None
        )
        self.cooling_allowed_entity: str | None = (
            data.get(CONF_HEAT_PUMP_COOLING_ALLOWED_ENTITY) or None
        )
        self.stop_heating_entity: str | None = (
            data.get(CONF_HEAT_PUMP_STOP_HEATING_TEMP_ENTITY) or None
        )
        self.start_cooling_entity: str | None = (
            data.get(CONF_HEAT_PUMP_START_COOLING_TEMP_ENTITY) or None
        )
        entry_id = getattr(config_entry, "entry_id", "unknown")
        self._store: Any = Store(hass, STORE_VERSION, f"{DOMAIN}.{entry_id}.{STORE_KEY}")
        self._days: dict[date, HeatPumpDay] = {}
        self._needs_backfill: set[date] = set()
        # Explicit modes observed live for a day the statistics still fill in.
        self._pending_modes: dict[date, str] = {}
        # Live capture of the current local day.
        self._live_date: date | None = None
        self._live_quarters = [0.0] * INTERVAL_COUNT
        self._live_coverage = [0.0] * INTERVAL_COUNT
        self._live_temp_sum = 0.0
        self._live_temp_seconds = 0.0
        self._live_flag_seconds = {"heating": 0.0, "cooling": 0.0, "observed": 0.0}
        self._last_sample_time: datetime | None = None
        self._last_sample_mono: float | None = None
        self._last_power_kw: float | None = None
        self._last_temp_c: float | None = None
        self._last_flags: tuple[bool | None, bool | None] = (None, None)
        # Measured hourly temperature: hour-start timestamp -> [sum, seconds].
        self._hourly_temp: dict[float, list[float]] = {}
        # Weather forecast cache.
        self._forecast: list[tuple[float, float]] = []
        self._forecast_fetched_mono: float | None = None
        self._forecast_fetched_at: str | None = None
        self._forecast_error: str | None = None
        self._weather_task: asyncio.Task | None = None
        self._last_weather_attempt_mono: float | None = None
        # Thresholds: last valid entity readings survive an unavailable entity.
        self._last_stop_heating_c: float | None = None
        self._last_start_cooling_c: float | None = None
        # Model cache.
        self._model: HeatPumpModel | None = None
        self._model_key: tuple[Any, ...] | None = None
        # Lifecycle.
        self._loaded = False
        self._last_error: str | None = None
        self._save_task: asyncio.Task | None = None
        self._last_save_mono = 0.0
        self._backfill_coordinator: RecorderBackfillCoordinator | None = None
        self._backfill_task: asyncio.Task | None = None
        self._backfill_generation = 1
        self._backfill_status = "not_started"
        self._backfill_days = 0
        self._last_backfill_mono: float | None = None
        self._background_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Configuration and clock
    # ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        return bool(self.power_sensors)

    def sensors_fingerprint(self) -> str:
        payload = {
            "power": sorted(self.power_sensors),
            "temperature": self.temperature_sensor,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def set_backfill_coordinator(self, coordinator: RecorderBackfillCoordinator) -> None:
        self._backfill_coordinator = coordinator

    def _timezone(self) -> Any:
        configured = getattr(getattr(self._hass, "config", None), "time_zone", None)
        if configured:
            try:
                return dt_util.get_time_zone(configured) or ZoneInfo(configured)
            except Exception:  # noqa: BLE001
                pass
        return timezone.utc

    def _now(self) -> datetime:
        try:
            current = dt_util.now()
        except Exception:  # noqa: BLE001
            current = datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=self._timezone())
        return current.astimezone(self._timezone())

    def _local(self, value: datetime | None) -> datetime:
        if value is None:
            return self._now()
        if value.tzinfo is None:
            return value.replace(tzinfo=self._timezone())
        return value.astimezone(self._timezone())

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    def _state(self, entity_id: str | None) -> Any:
        if not entity_id:
            return None
        states = getattr(self._hass, "states", None)
        get = getattr(states, "get", None)
        if not callable(get):
            return None
        try:
            return get(entity_id)
        except Exception:  # noqa: BLE001
            return None

    def read_power_kw(self) -> float | None:
        """Sum of every configured heat pump power sensor, or None.

        One unavailable sensor makes the sum unusable: half a heat pump is
        not a measurement.
        """
        if not self.power_sensors:
            return None
        total = 0.0
        for entity_id in self.power_sensors:
            state = self._state(entity_id)
            if state is None or getattr(state, "state", None) in ("unknown", "unavailable"):
                return None
            value = _finite(getattr(state, "state", None))
            if value is None:
                return None
            unit = str(
                (getattr(state, "attributes", {}) or {}).get("unit_of_measurement", "W")
            ).strip().lower()
            if unit == "kw":
                kw = value
            elif unit == "w":
                kw = value / 1000.0
            else:
                return None
            total += max(0.0, kw)
        return total

    def read_temperature_c(self) -> float | None:
        state = self._state(self.temperature_sensor)
        if state is None:
            return None
        value = _finite(getattr(state, "state", None))
        if value is None:
            return None
        unit = (getattr(state, "attributes", {}) or {}).get("unit_of_measurement")
        return _to_celsius(value, unit)

    def _read_threshold(self, entity_id: str | None) -> float | None:
        state = self._state(entity_id)
        if state is None:
            return None
        value = _finite(getattr(state, "state", None))
        if value is None:
            return None
        unit = (getattr(state, "attributes", {}) or {}).get("unit_of_measurement")
        return _to_celsius(value, unit)

    def thresholds(self) -> tuple[float | None, float | None]:
        stop = self._read_threshold(self.stop_heating_entity)
        if stop is not None:
            self._last_stop_heating_c = stop
        start = self._read_threshold(self.start_cooling_entity)
        if start is not None:
            self._last_start_cooling_c = start
        return (
            stop if stop is not None else self._last_stop_heating_c,
            start if start is not None else self._last_start_cooling_c,
        )

    def read_flags(self) -> tuple[bool | None, bool | None]:
        heating = (
            parse_allowed_state(self._state(self.heating_allowed_entity))
            if self.heating_allowed_entity
            else None
        )
        cooling = (
            parse_allowed_state(self._state(self.cooling_allowed_entity))
            if self.cooling_allowed_entity
            else None
        )
        return heating, cooling

    @property
    def has_allowed_entities(self) -> bool:
        return bool(self.heating_allowed_entity or self.cooling_allowed_entity)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def async_load(self) -> bool:
        try:
            data = await self._store.async_load()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"load: {exc}"
            self._loaded = True
            return False
        self._loaded = True
        if not isinstance(data, Mapping):
            return False
        if data.get("capture_version") != CAPTURE_VERSION:
            return False
        if data.get("sensors_fingerprint") != self.sensors_fingerprint():
            # Other sensors measure something else: relearn from statistics.
            _LOGGER.info("Heat pump profile: sensors changed; relearning")
            return False
        for raw in data.get("days", []) or []:
            day = HeatPumpDay.from_dict(raw)
            if day is not None:
                self._days[day.local_date] = day
        for raw in data.get("needs_backfill", []) or []:
            try:
                self._needs_backfill.add(date.fromisoformat(str(raw)))
            except ValueError:
                continue
        for raw in data.get("hourly_temperatures", []) or []:
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                start, value = _finite(raw[0]), _finite(raw[1])
                if start is not None and value is not None:
                    self._hourly_temp[start] = [value * 3600.0, 3600.0]
        thresholds = data.get("thresholds") if isinstance(data.get("thresholds"), Mapping) else {}
        self._last_stop_heating_c = _finite(thresholds.get("stop_heating_c"))
        self._last_start_cooling_c = _finite(thresholds.get("start_cooling_c"))
        live = data.get("live") if isinstance(data.get("live"), Mapping) else None
        if live is not None:
            self._restore_live(live)
        self._prune()
        self._model = None
        return bool(self._days)

    def _restore_live(self, live: Mapping[str, Any]) -> None:
        try:
            live_date = date.fromisoformat(str(live.get("date")))
        except ValueError:
            return
        quarters = live.get("quarters_kwh")
        coverage = live.get("coverage_s")
        if not (
            isinstance(quarters, list)
            and isinstance(coverage, list)
            and len(quarters) == INTERVAL_COUNT
            and len(coverage) == INTERVAL_COUNT
        ):
            return
        self._live_date = live_date
        self._live_quarters = [max(0.0, _finite(v) or 0.0) for v in quarters]
        self._live_coverage = [max(0.0, _finite(v) or 0.0) for v in coverage]
        self._live_temp_sum = _finite(live.get("temp_sum")) or 0.0
        self._live_temp_seconds = max(0.0, _finite(live.get("temp_seconds")) or 0.0)
        flags = live.get("flag_seconds") if isinstance(live.get("flag_seconds"), Mapping) else {}
        for key in self._live_flag_seconds:
            self._live_flag_seconds[key] = max(0.0, _finite(flags.get(key)) or 0.0)

    def _store_payload(self) -> dict[str, Any]:
        self._prune()
        horizon = self._now().timestamp() - MEASURED_HISTORY_HOURS * 3600.0
        return {
            "capture_version": CAPTURE_VERSION,
            "sensors_fingerprint": self.sensors_fingerprint(),
            "timezone": getattr(getattr(self._hass, "config", None), "time_zone", None),
            "days": [self._days[key].as_dict() for key in sorted(self._days)],
            "needs_backfill": sorted(value.isoformat() for value in self._needs_backfill),
            "hourly_temperatures": [
                [start, round(total / seconds, 3)]
                for start, (total, seconds) in sorted(self._hourly_temp.items())
                if seconds > 0.0 and start >= horizon
            ],
            "thresholds": {
                "stop_heating_c": self._last_stop_heating_c,
                "start_cooling_c": self._last_start_cooling_c,
            },
            "live": (
                None
                if self._live_date is None
                else {
                    "date": self._live_date.isoformat(),
                    "quarters_kwh": [round(v, 6) for v in self._live_quarters],
                    "coverage_s": [round(v, 2) for v in self._live_coverage],
                    "temp_sum": round(self._live_temp_sum, 3),
                    "temp_seconds": round(self._live_temp_seconds, 2),
                    "flag_seconds": {
                        key: round(value, 1) for key, value in self._live_flag_seconds.items()
                    },
                }
            ),
        }

    async def async_save(self) -> None:
        try:
            await self._store.async_save(self._store_payload())
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"save: {exc}"
            _LOGGER.warning("Heat pump profile: failed to save Store: %s", exc)

    def request_save(self) -> None:
        if self._save_task is not None and not self._save_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._save_task = loop.create_task(self.async_save())

    def _prune(self) -> None:
        today = self._now().date()
        floor = today - timedelta(days=SAMPLE_RETENTION_DAYS)
        self._days = {key: day for key, day in self._days.items() if key >= floor}
        shape_floor = today - timedelta(days=SHAPE_RETENTION_DAYS)
        for key, day in self._days.items():
            if key < shape_floor:
                day.quarters_kwh = None
        self._needs_backfill = {key for key in self._needs_backfill if key >= floor}
        horizon = self._now().timestamp() - MEASURED_HISTORY_HOURS * 3600.0
        self._hourly_temp = {
            start: value for start, value in self._hourly_temp.items() if start >= horizon
        }

    def _create_task(self, coroutine: Any, name: str) -> asyncio.Task | None:
        create = getattr(self._controller, "_create_entry_background_task", None)
        task = None
        try:
            if callable(create):
                task = create(coroutine, name)
            else:
                task = asyncio.get_running_loop().create_task(coroutine, name=name)
        except Exception:  # noqa: BLE001
            close = getattr(coroutine, "close", None)
            if callable(close):
                close()
            return None
        if isinstance(task, asyncio.Task):
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return task
        close = getattr(coroutine, "close", None)
        if task is None and callable(close):
            close()
        return None

    async def async_stop(self) -> None:
        """Invalidate backfill and stop background tasks (unload)."""
        self._backfill_generation += 1
        tasks = [
            task
            for task in (*self._background_tasks, self._backfill_task, self._weather_task)
            if task is not None and not task.done() and task is not asyncio.current_task()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    async def async_save_all(self) -> None:
        await self.async_stop()
        if self._save_task is not None and not self._save_task.done():
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
        await self.async_save()

    # ------------------------------------------------------------------
    # Live capture
    # ------------------------------------------------------------------

    def _days_changed(self) -> None:
        self._model = None

    def record_sample(
        self,
        *,
        local_time: datetime | None = None,
        monotonic_time: float | None = None,
    ) -> float | None:
        """Integrate one reading; return the summed heat pump power (kW).

        The return value lets the consumption tracker subtract the same
        reading from the home power for the base-load profile.
        """
        now = self._local(local_time)
        mono = monotonic() if monotonic_time is None else monotonic_time
        current_date = now.date()
        if self._live_date is None:
            self._live_date = current_date
        elif current_date != self._live_date:
            self._close_live_day(current_date)

        power_kw = self.read_power_kw()
        temp_c = self.read_temperature_c()
        flags = self.read_flags()

        previous_time = self._last_sample_time
        previous_mono = self._last_sample_mono
        elapsed: float | None = None
        if previous_time is not None and previous_mono is not None:
            elapsed = mono - previous_mono
        if elapsed is not None and 0.0 < elapsed <= MAX_SAMPLE_GAP_SECONDS:
            if power_kw is not None and self._last_power_kw is not None:
                for contribution in split_sample_across_bins(
                    previous_time, now, self._last_power_kw, power_kw
                ):
                    if contribution.local_date != self._live_date:
                        continue
                    index = contribution.interval_index
                    self._live_quarters[index] += contribution.energy_kwh
                    self._live_coverage[index] += contribution.coverage_s
            if temp_c is not None and self._last_temp_c is not None:
                average = (temp_c + self._last_temp_c) / 2.0
                self._live_temp_sum += average * elapsed
                self._live_temp_seconds += elapsed
                hour_start = now.replace(minute=0, second=0, microsecond=0).timestamp()
                slot = self._hourly_temp.setdefault(hour_start, [0.0, 0.0])
                slot[0] += average * elapsed
                slot[1] += elapsed
            heating, cooling = self._last_flags
            if heating is not None or cooling is not None:
                self._live_flag_seconds["observed"] += elapsed
                if heating:
                    self._live_flag_seconds["heating"] += elapsed
                if cooling:
                    self._live_flag_seconds["cooling"] += elapsed

        self._last_sample_time = now
        self._last_sample_mono = mono
        self._last_power_kw = power_kw
        self._last_temp_c = temp_c
        self._last_flags = flags
        if self._last_save_mono == 0.0 or mono - self._last_save_mono >= SAVE_INTERVAL_S:
            self._last_save_mono = mono
            self.request_save()
        return power_kw

    def _close_live_day(self, new_date: date) -> None:
        """Turn the finished live capture into a daily sample."""
        closed = self._live_date
        if closed is not None:
            start, end = local_day_bounds(closed, self._timezone())
            expected = max(1.0, end.timestamp() - start.timestamp())
            covered = sum(self._live_coverage)
            coverage = min(1.0, covered / expected)
            temp = (
                self._live_temp_sum / self._live_temp_seconds
                if self._live_temp_seconds >= DAY_MIN_TEMPERATURE_HOURS * 3600.0
                else None
            )
            mode = None
            observed = self._live_flag_seconds["observed"]
            if self.has_allowed_entities and observed >= 0.5 * expected:
                heating_share = self._live_flag_seconds["heating"] / observed
                cooling_share = self._live_flag_seconds["cooling"] / observed
                if cooling_share >= 0.5 and cooling_share >= heating_share:
                    mode = MODE_COOLING
                elif heating_share >= 0.5:
                    mode = MODE_HEATING
                else:
                    mode = MODE_HOT_WATER
            if coverage >= DAY_MIN_COVERAGE:
                energy = sum(self._live_quarters) / coverage
                existing = self._days.get(closed)
                if existing is None or existing.coverage <= coverage:
                    self._days[closed] = HeatPumpDay(
                        local_date=closed,
                        energy_kwh=energy,
                        temp_c=temp if temp is not None else (existing.temp_c if existing else None),
                        coverage=coverage,
                        mode=mode,
                        source="live",
                        quarters_kwh=[value / coverage for value in self._live_quarters],
                    )
                elif mode is not None:
                    existing.mode = mode
                self._needs_backfill.discard(closed)
            else:
                # Too many gaps (a restart): the statistics fill this day in;
                # the explicit mode observed live is kept.
                self._needs_backfill.add(closed)
                existing = self._days.get(closed)
                if existing is not None and mode is not None:
                    existing.mode = mode
                elif mode is not None:
                    self._pending_modes[closed] = mode
            self._days_changed()
            self.request_save()
        self._live_date = new_date
        self._live_quarters = [0.0] * INTERVAL_COUNT
        self._live_coverage = [0.0] * INTERVAL_COUNT
        self._live_temp_sum = 0.0
        self._live_temp_seconds = 0.0
        self._live_flag_seconds = {"heating": 0.0, "cooling": 0.0, "observed": 0.0}
        self._prune()

    # ------------------------------------------------------------------
    # Statistics backfill
    # ------------------------------------------------------------------

    def _dates_needing_backfill(self, today: date) -> list[date]:
        floor = today - timedelta(days=SAMPLE_RETENTION_DAYS)
        result: list[date] = []
        for offset in range((today - floor).days):
            local_date = floor + timedelta(days=offset)
            day = self._days.get(local_date)
            if (
                day is None
                or local_date in self._needs_backfill
                or day.temp_c is None
                or day.coverage < DAY_MIN_COVERAGE
            ):
                result.append(local_date)
        return result

    def merge_backfilled_days(self, days: Mapping[date, HeatPumpDay]) -> int:
        """Fold statistics days in without losing an explicit live mode."""
        changed = 0
        for local_date, day in days.items():
            existing = self._days.get(local_date)
            pending_mode = self._pending_modes.pop(local_date, None)
            if existing is not None:
                if existing.source == "live" and existing.coverage >= day.coverage and (
                    local_date not in self._needs_backfill
                ):
                    if existing.temp_c is None and day.temp_c is not None:
                        existing.temp_c = day.temp_c
                        changed += 1
                    continue
                day.mode = existing.mode or pending_mode
            else:
                day.mode = pending_mode
            self._days[local_date] = day
            self._needs_backfill.discard(local_date)
            changed += 1
        if changed:
            self._days_changed()
        return changed

    async def async_backfill_from_statistics(
        self, token: BackfillToken | None = None
    ) -> bool:
        """Fill missing days from hourly long-term statistics."""
        generation = self._backfill_generation
        self._backfill_status = "running"
        self._last_backfill_mono = monotonic()
        try:
            coordinator = self._backfill_coordinator
            if coordinator is None:
                self._backfill_status = "coordinator_unavailable"
                return False
            if token is None:
                token = coordinator.new_token()
            today = self._now().date()
            missing = self._dates_needing_backfill(today)
            if not missing:
                self._backfill_status = "complete"
                return False
            tz = self._timezone()
            ids = set(self.power_sensors)
            if self.temperature_sensor:
                ids.add(self.temperature_sensor)
            changed = 0
            index = 0
            while index < len(missing):
                chunk_start = missing[index]
                chunk_end = chunk_start + timedelta(days=STATISTICS_CHUNK_DAYS)
                chunk = [d for d in missing[index:] if d < chunk_end]
                index += len(chunk)
                if generation != self._backfill_generation or not token.is_valid():
                    self._backfill_status = "cancelled"
                    return False
                start, _ = local_day_bounds(chunk[0], tz)
                _, end = local_day_bounds(chunk[-1], tz)
                rows = await coordinator.async_query_statistics(
                    token,
                    ids,
                    start,
                    end,
                    period="hour",
                    types={"mean"},
                    units={"power": "kW", "temperature": "°C"},
                    block=f"{chunk[0].isoformat()}..{chunk[-1].isoformat()}",
                )
                if rows is None or generation != self._backfill_generation:
                    self._backfill_status = "cancelled"
                    return False
                energy, temperature = aggregate_hourly_statistics(
                    rows, self.power_sensors, self.temperature_sensor
                )
                del rows
                self._absorb_hourly_temperatures(temperature)
                days = days_from_hourly(energy, temperature, chunk, tz)
                changed += self.merge_backfilled_days(days)
                self._backfill_days += len(days)
                await asyncio.sleep(0)
            # A day the statistics could not fill within a few days never will;
            # stop retrying it (the startup backfill still looks once more).
            self._needs_backfill = {
                local_date
                for local_date in self._needs_backfill
                if local_date >= today - timedelta(days=3)
            }
            if changed:
                await self.async_save()
            self._backfill_status = "complete"
            return bool(changed)
        except asyncio.CancelledError:
            self._backfill_status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - learning is best effort
            self._last_error = f"backfill: {exc}"
            self._backfill_status = "failed"
            _LOGGER.warning("Heat pump profile: statistics backfill failed: %s", exc)
            return False

    def _absorb_hourly_temperatures(self, temperature: Mapping[float, float]) -> None:
        horizon = self._now().timestamp() - MEASURED_HISTORY_HOURS * 3600.0
        for start, value in temperature.items():
            if start >= horizon and start not in self._hourly_temp:
                self._hourly_temp[start] = [value * 3600.0, 3600.0]

    def start_backfill(self, coordinator: RecorderBackfillCoordinator | None = None) -> None:
        if not self.configured:
            return
        if coordinator is not None:
            self._backfill_coordinator = coordinator
        if self._backfill_task is not None and not self._backfill_task.done():
            return
        if self._backfill_coordinator is None:
            self._backfill_status = "coordinator_unavailable"
            return
        self._backfill_task = self._backfill_coordinator.submit(
            "heat_pump_profile", self.async_backfill_from_statistics
        )

    # ------------------------------------------------------------------
    # Weather forecast
    # ------------------------------------------------------------------

    def maybe_refresh(self, *, monotonic_time: float | None = None) -> None:
        """Throttled background work: weather forecast, late statistics."""
        if not self.configured:
            return
        mono = monotonic() if monotonic_time is None else monotonic_time
        if self.weather_entity and (
            self._weather_task is None or self._weather_task.done()
        ):
            due = (
                self._forecast_fetched_mono is None
                or mono - self._forecast_fetched_mono >= WEATHER_REFRESH_S
            )
            retry_ok = (
                self._last_weather_attempt_mono is None
                or mono - self._last_weather_attempt_mono >= WEATHER_RETRY_S
            )
            if due and retry_ok:
                self._last_weather_attempt_mono = mono
                self._weather_task = self._create_task(
                    self.async_refresh_weather(), "omnibattery_heat_pump_weather"
                )
        # Yesterday's statistics land shortly after midnight; a day with gaps
        # is retried a few times a day instead of only at the next restart.
        # Only after the startup backfill ran: before Home Assistant has
        # started, the Recorder is not ready.
        if (
            self._needs_backfill
            and self._last_backfill_mono is not None
            and mono - self._last_backfill_mono >= BACKFILL_RETRY_S
        ):
            self._last_backfill_mono = mono
            self.start_backfill()

    async def async_refresh_weather(self) -> bool:
        """Fetch the hourly forecast through ``weather.get_forecasts``."""
        entity_id = self.weather_entity
        if not entity_id:
            return False
        try:
            services = getattr(self._hass, "services", None)
            call = getattr(services, "async_call", None)
            if not callable(call):
                raise RuntimeError("services unavailable")
            response = await asyncio.wait_for(
                call(
                    "weather",
                    "get_forecasts",
                    {"entity_id": entity_id, "type": "hourly"},
                    blocking=True,
                    return_response=True,
                ),
                WEATHER_TIMEOUT_S,
            )
            entries = parse_forecast_response(response, entity_id)
            state = self._state(entity_id)
            unit = (getattr(state, "attributes", {}) or {}).get("temperature_unit")
            entries = [(ts, _to_celsius(value, unit)) for ts, value in entries]
            if not entries:
                raise ValueError("empty hourly forecast")
            self._forecast = entries
            self._forecast_fetched_mono = monotonic()
            self._forecast_fetched_at = self._now().isoformat(timespec="minutes")
            self._forecast_error = None
            return True
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the measured fallback covers it
            self._forecast_error = f"{type(exc).__name__}: {exc}"[:200]
            _LOGGER.debug("Heat pump profile: weather forecast failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Temperatures for a forecast day
    # ------------------------------------------------------------------

    def _measured_hour(self, start: float) -> float | None:
        slot = self._hourly_temp.get(start)
        if slot is None or slot[1] < 600.0:
            return None
        return slot[0] / slot[1]

    def _forecast_hour(self, start: float) -> float | None:
        for ts, value in self._forecast:
            if start - 1.0 <= ts < start + 3600.0 - 1.0:
                return value
        return None

    def hourly_temperatures(
        self, local_date: date, now: datetime | None = None
    ) -> tuple[list[float | None], str]:
        """24 hourly temperatures for a day and where they came from.

        Past hours use the measurement; future hours the weather forecast,
        else the same hour in the last 24 h of measurement.
        """
        now = self._local(now)
        now_ts = now.timestamp()
        tz = self._timezone()
        recent_by_hour: dict[int, float] = {}
        for start in sorted(self._hourly_temp):
            if now_ts - 24 * 3600.0 - 3600.0 <= start < now_ts:
                value = self._measured_hour(start)
                if value is not None:
                    recent_by_hour[datetime.fromtimestamp(start, tz).hour] = value
        current = self.read_temperature_c()
        temps: list[float | None] = [None] * 24
        used: set[str] = set()
        start_dt, end_dt = local_day_bounds(local_date, tz)
        hour = start_dt.timestamp()
        while hour < end_dt.timestamp() - 1e-6:
            local_hour = datetime.fromtimestamp(hour, tz).hour
            value: float | None = None
            if hour + 3600.0 <= now_ts:
                value = self._measured_hour(hour)
                if value is not None:
                    used.add("measured")
            if value is None:
                value = self._forecast_hour(hour)
                if value is not None:
                    used.add("weather_forecast")
            if value is None:
                value = recent_by_hour.get(local_hour)
                if value is not None:
                    used.add("measured_last_24h")
            if value is None and current is not None:
                value = current
                used.add("current_reading")
            if value is not None:
                temps[local_hour] = value
            hour += 3600.0
        if "weather_forecast" in used:
            source = "weather_forecast"
        elif "measured_last_24h" in used or "current_reading" in used:
            source = "measured_last_24h"
        elif used:
            source = "measured"
        else:
            source = "unavailable"
        return temps, source

    def day_temperature(
        self, local_date: date, now: datetime | None = None
    ) -> tuple[float | None, str]:
        stored = self._days.get(local_date)
        if stored is not None and stored.temp_c is not None:
            return stored.temp_c, "measured"
        temps, source = self.hourly_temperatures(local_date, now)
        known = [value for value in temps if value is not None]
        if len(known) < DAY_MIN_TEMPERATURE_HOURS:
            return None, "unavailable"
        return sum(known) / len(known), source

    # ------------------------------------------------------------------
    # Model and forecast
    # ------------------------------------------------------------------

    def model(self) -> HeatPumpModel:
        stop, start = self.thresholds()
        key = (stop, start, len(self._days))
        if self._model is None or self._model_key != key:
            self._model = fit_heat_pump_model(
                self._days.values(), stop_heating_c=stop, start_cooling_c=start
            )
            if self.stop_heating_entity or self.start_cooling_entity:
                self._model.threshold_source = (
                    "entity"
                    if stop is not None and start is not None
                    else "entity_partial"
                )
            self._model_key = key
        return self._model

    def forecast_day(
        self, local_date: date, now: datetime | None = None
    ) -> HeatPumpDayForecast:
        """Daily heat pump forecast; ``mature`` False means "use the fallback"."""
        now = self._local(now)
        model = self.model()
        day_temp, source = self.day_temperature(local_date, now)
        previous_temp, _ = self.day_temperature(local_date - timedelta(days=1), now)
        t_eff = effective_temperature(day_temp, previous_temp)

        def immature(reason: str, mode: str | None = None, basis: str | None = None):
            return HeatPumpDayForecast(
                local_date, [0.0] * INTERVAL_COUNT, 0.0, mode, basis, t_eff,
                day_temp, source, 1.0, "none", False, reason,
            )

        if not self.configured:
            return immature("heat_pump_not_configured")
        if t_eff is None or day_temp is None:
            return immature("temperature_unavailable")
        if model.sample_days == 0:
            return immature("no_heat_pump_days")
        # Hand-switched modes are assumed to stay as they are now.
        if self.has_allowed_entities:
            heating, cooling = self.read_flags()
        else:
            heating = cooling = None
        mode, basis = choose_forecast_mode(
            model,
            t_eff,
            heating_allowed=heating,
            cooling_allowed=cooling,
            recent_modes=model.recent_modes(min(local_date, now.date())),
        )
        energy = model.predict(t_eff, mode)
        if energy is None:
            return immature(f"heat_pump_model_immature:{mode}", mode, basis)
        correction = recent_correction(model, min(local_date, now.date()))
        energy *= correction
        modes = {item.day.local_date: item.mode for item in model.classified}
        shape, shape_source = learn_shape(
            self._days.values(), modes, mode, before=min(local_date, now.date())
        )
        hourly, _ = self.hourly_temperatures(local_date, now)
        intervals = distribute_day(
            energy,
            shape,
            hourly,
            mode,
            stop_heating_c=model.stop_heating_c,
            start_cooling_c=model.start_cooling_c,
        )
        return HeatPumpDayForecast(
            local_date, intervals, energy, mode, basis, t_eff, day_temp, source,
            correction, shape_source, True, None,
        )

    def temperature_source(self) -> str:
        if self.weather_entity and self._forecast:
            return "weather_forecast"
        if self._hourly_temp:
            return "measured_last_24h"
        return "unavailable"

    def diagnostics(self) -> dict[str, Any]:
        model = self.model()
        today = self._now().date()
        forecasts = {}
        for label, local_date in (("today", today), ("tomorrow", today + timedelta(days=1))):
            try:
                forecasts[label] = self.forecast_day(local_date).as_dict()
            except Exception as exc:  # noqa: BLE001
                forecasts[label] = {"error": str(exc)}
        sources: dict[str, int] = {}
        for day in self._days.values():
            sources[day.source] = sources.get(day.source, 0) + 1
        return {
            "configured": self.configured,
            "power_sensors": list(self.power_sensors),
            "temperature_sensor": self.temperature_sensor,
            "weather_entity": self.weather_entity,
            "allowed_entities": {
                "heating": self.heating_allowed_entity,
                "cooling": self.cooling_allowed_entity,
            },
            "threshold_entities": {
                "stop_heating": self.stop_heating_entity,
                "start_cooling": self.start_cooling_entity,
            },
            "model": model.as_dict(),
            "sample_days": len(self._days),
            "sample_sources": sources,
            "days_awaiting_statistics": len(self._needs_backfill),
            "temperature_source": self.temperature_source(),
            "weather_forecast": {
                "hours": len(self._forecast),
                "fetched_at": self._forecast_fetched_at,
                "last_error": self._forecast_error,
            },
            "measured_temperature_hours": len(self._hourly_temp),
            "live_capture": {
                "date": self._live_date.isoformat() if self._live_date else None,
                "energy_kwh": round(sum(self._live_quarters), 3),
                "coverage_ratio": round(
                    sum(self._live_coverage) / (INTERVAL_COUNT * INTERVAL_SECONDS), 3
                ),
            },
            "forecast": forecasts,
            "backfill_status": self._backfill_status,
            "backfill_days": self._backfill_days,
            "last_error": self._last_error,
        }


__all__ = [
    "BaseLoadProfileTracker",
    "ClassifiedDay",
    "HeatPumpDay",
    "HeatPumpDayForecast",
    "HeatPumpModel",
    "HeatPumpProfileTracker",
    "MODE_COOLING",
    "MODE_HEATING",
    "MODE_HOT_WATER",
    "ModeFit",
    "aggregate_hourly_statistics",
    "choose_forecast_mode",
    "classify_days",
    "days_from_hourly",
    "distribute_day",
    "effective_temperature",
    "fit_heat_pump_model",
    "heat_pump_power_sensors",
    "huber_regression",
    "learn_shape",
    "parse_allowed_state",
    "parse_forecast_response",
    "recent_correction",
]
