"""Heat pump shadow forecast: pure model, capture, backfill and isolation.

The entity ids below only mirror a typical two-meter installation (an outdoor
and an indoor unit); nothing in the integration depends on them.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from homeassistant.util import dt as dt_util

from custom_components.omnibattery.const import (
    CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR,
    CONF_HEAT_PUMP_POWER_SENSORS,
    CONF_HEAT_PUMP_WEATHER_ENTITY,
    CONF_HEAT_PUMP_HEATING_ALLOWED_ENTITY,
)
from custom_components.omnibattery.tracking import (
    consumption_profile as profile_module,
    consumption_tracker as tracker_module,
    heat_pump_profile as hp_module,
    solar_profile as solar_module,
)
from custom_components.omnibattery.tracking.consumption_profile import (
    INTERVAL_COUNT,
    INTERVAL_SECONDS,
    ConsumptionProfileTracker,
    ProfileDay,
)
from custom_components.omnibattery.tracking.consumption_tracker import ConsumptionTracker
from custom_components.omnibattery.tracking.forecast_history import (
    ForecastHistoryManager,
)
from custom_components.omnibattery.tracking.heat_pump_profile import (
    MODE_COOLING,
    MODE_HEATING,
    MODE_HOT_WATER,
    BaseLoadProfileTracker,
    HeatPumpDay,
    HeatPumpProfileTracker,
    aggregate_hourly_statistics,
    choose_forecast_mode,
    days_from_hourly,
    distribute_day,
    effective_temperature,
    fit_heat_pump_model,
    huber_regression,
    parse_allowed_state,
    parse_forecast_response,
)

AMSTERDAM = ZoneInfo("Europe/Amsterdam")
OUTDOOR_UNIT = "sensor.hp_outdoor_unit_power"
INDOOR_UNIT = "sensor.hp_indoor_unit_power"
OUTDOOR_TEMP = "sensor.outdoor_temperature"
WEATHER = "weather.forecast_home"
HEATING_FLAG = "sensor.heating_allowed"
NOW = datetime(2026, 11, 15, 12, 0, tzinfo=AMSTERDAM)


# ---------------------------------------------------------------------------
# Synthetic daily samples
# ---------------------------------------------------------------------------


def _series(temps, energy_fn, *, start=date(2026, 1, 1), mode=None):
    """Daily samples whose energy is a function of the blended temperature."""
    days = []
    previous = None
    for offset, temp in enumerate(temps):
        t_eff = effective_temperature(temp, previous)
        days.append(
            HeatPumpDay(
                local_date=start + timedelta(days=offset),
                energy_kwh=max(0.0, energy_fn(offset, t_eff)),
                temp_c=temp,
                mode=mode(offset) if callable(mode) else mode,
            )
        )
        previous = temp
    return days


def _heating_season(seed=1, *, heating_days=60, hot_water_days=20):
    rng = random.Random(seed)
    temps = [rng.uniform(18.5, 21.5) for _ in range(hot_water_days)]
    temps += [rng.uniform(-5.0, 14.0) for _ in range(heating_days)]

    def energy(offset, t_eff):
        if offset < hot_water_days or t_eff >= 17.0:
            return 5.0 + rng.gauss(0.0, 0.3)
        return 6.0 + 1.25 * (17.0 - t_eff) + rng.gauss(0.0, 1.0)

    return _series(temps, energy)


# ---------------------------------------------------------------------------
# Pure model
# ---------------------------------------------------------------------------


def test_effective_temperature_blends_seventy_thirty():
    assert effective_temperature(10.0, 0.0) == pytest.approx(7.0)
    assert effective_temperature(10.0, None) == 10.0
    assert effective_temperature(None, 4.0) == 4.0


def test_huber_regression_resists_outliers():
    xs = [float(x) for x in range(20)]
    ys = [2.0 + 1.5 * x for x in xs]
    ys[3] += 40.0
    ys[11] -= 30.0
    intercept, slope, _ = huber_regression(xs, ys)
    assert slope == pytest.approx(1.5, abs=0.05)
    assert intercept == pytest.approx(2.0, abs=0.5)


def test_heating_slope_is_recovered():
    model = fit_heat_pump_model(_heating_season(), stop_heating_c=17.0)
    assert model.heating.mature
    assert model.heating.slope_kwh_per_k == pytest.approx(1.25, abs=0.12)
    assert model.heating.intercept_kwh == pytest.approx(6.0, abs=1.2)
    assert model.hot_water.mature
    assert model.hot_water.intercept_kwh == pytest.approx(5.0, abs=0.3)
    assert model.predict(2.0, MODE_HEATING) == pytest.approx(6.0 + 1.25 * 15, rel=0.1)


def test_cold_hot_water_only_days_do_not_flatten_the_heating_slope():
    days = _heating_season()
    rng = random.Random(7)
    start = days[-1].local_date + timedelta(days=1)
    # Heating switched off by hand during a cold week: hot water use only.
    off_days = [
        HeatPumpDay(start + timedelta(days=i), 5.0 + rng.gauss(0, 0.3), rng.uniform(-2.0, 6.0))
        for i in range(15)
    ]
    with_off = fit_heat_pump_model(days + off_days, stop_heating_c=17.0)
    without_off = fit_heat_pump_model(days, stop_heating_c=17.0)
    assert with_off.heating.slope_kwh_per_k == pytest.approx(
        without_off.heating.slope_kwh_per_k, abs=0.1
    )
    modes = {item.day.local_date: item.mode for item in with_off.classified}
    assert all(modes[day.local_date] == MODE_HOT_WATER for day in off_days)
    # A plain least-squares line over every cold day would be flattened.
    cold = [item for item in with_off.classified if item.t_eff < 15]
    xs = [17.0 - item.t_eff for item in cold]
    ys = [item.day.energy_kwh for item in cold]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    naive_slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )
    assert naive_slope < with_off.heating.slope_kwh_per_k - 0.15


def test_a_season_of_heating_only_still_learns_the_heating_line():
    rng = random.Random(11)
    temps = [rng.uniform(-6.0, 12.0) for _ in range(40)]
    days = _series(temps, lambda offset, t: 6.0 + 1.25 * (17.0 - t))
    model = fit_heat_pump_model(days, stop_heating_c=17.0)
    assert model.heating.mature
    assert model.heating.slope_kwh_per_k == pytest.approx(1.25, abs=0.01)
    assert not model.hot_water.mature


def test_heating_off_days_are_found_without_mild_days():
    rng = random.Random(12)
    temps = [rng.uniform(-6.0, 12.0) for _ in range(40)]

    def energy(offset, t):
        return 5.0 if offset % 5 == 0 else 6.0 + 1.25 * (17.0 - t)

    model = fit_heat_pump_model(_series(temps, energy), stop_heating_c=17.0)
    assert model.heating.slope_kwh_per_k == pytest.approx(1.25, abs=0.05)
    off = [item for item in model.classified if item.day.energy_kwh == 5.0]
    assert off and all(item.mode == MODE_HOT_WATER for item in off)


def test_explicit_mode_wins_over_inference():
    days = _heating_season(heating_days=12, hot_water_days=6)
    # Explicitly "heating" but at hot-water level: it stays heating.
    days[-1].mode = MODE_HEATING
    days[-1].energy_kwh = 5.0
    days[0].mode = MODE_HOT_WATER
    model = fit_heat_pump_model(days, stop_heating_c=17.0)
    by_date = {item.day.local_date: item for item in model.classified}
    assert by_date[days[-1].local_date].mode == MODE_HEATING
    assert by_date[days[-1].local_date].mode_source == "explicit"
    assert by_date[days[0].local_date].mode_source == "explicit"


def test_cooling_needs_twenty_days_and_four_kelvin():
    rng = random.Random(3)
    temps = [rng.uniform(18.5, 21.5) for _ in range(10)]
    temps += [rng.uniform(26.0, 33.0) for _ in range(15)]

    def energy(offset, t_eff):
        return 5.0 if offset < 10 else 5.0 + 1.5 * max(0.0, t_eff - 24.0)

    model = fit_heat_pump_model(
        _series(temps, energy), stop_heating_c=17.0, start_cooling_c=24.0
    )
    assert model.cooling.days == 15
    assert not model.cooling.mature
    assert model.cooling.reason == "insufficient_days"


def test_cooling_branch():
    rng = random.Random(3)
    temps = [rng.uniform(18.5, 21.5) for _ in range(10)]
    temps += [rng.uniform(26.0, 35.0) for _ in range(25)]

    def energy(offset, t_eff):
        if offset < 10:
            return 5.0
        return 5.0 + 1.5 * max(0.0, t_eff - 24.0)

    model = fit_heat_pump_model(
        _series(temps, energy), stop_heating_c=17.0, start_cooling_c=24.0
    )
    assert model.cooling.mature
    assert model.cooling.slope_kwh_per_k == pytest.approx(1.5, abs=0.1)
    assert model.predict(30.0, MODE_COOLING) == pytest.approx(5.0 + 1.5 * 6, rel=0.08)
    assert not model.heating.mature


def test_immature_model_predicts_nothing():
    days = _heating_season(heating_days=5, hot_water_days=2)
    model = fit_heat_pump_model(days, stop_heating_c=17.0)
    assert not model.heating.mature
    assert model.heating.reason == "insufficient_days"
    assert model.predict(0.0, MODE_HEATING) is None


def test_too_little_temperature_spread_is_immature():
    temps = [20.0] * 6 + [5.0 + 0.1 * i for i in range(20)]
    days = _series(temps, lambda offset, t: 5.0 if offset < 6 else 6.0 + 1.2 * (17 - t))
    model = fit_heat_pump_model(days, stop_heating_c=17.0)
    assert not model.heating.mature
    assert model.heating.reason == "insufficient_temperature_spread"


def test_previous_day_temperature_enters_the_fit():
    days = _series([10.0, 0.0, 0.0], lambda offset, t: 5.0)
    model = fit_heat_pump_model(days, stop_heating_c=17.0)
    t_eff = {item.day.local_date: item.t_eff for item in model.classified}
    assert t_eff[days[0].local_date] == pytest.approx(10.0)
    assert t_eff[days[1].local_date] == pytest.approx(0.7 * 0.0 + 0.3 * 10.0)
    assert t_eff[days[2].local_date] == pytest.approx(0.0)


def test_forecast_mode_follows_switches_then_recent_days():
    model = fit_heat_pump_model(_heating_season(), stop_heating_c=17.0)
    assert choose_forecast_mode(
        model, 3.0, heating_allowed=False, cooling_allowed=False, recent_modes=[]
    ) == (MODE_HOT_WATER, "allowed_entities")
    assert choose_forecast_mode(
        model, 3.0, heating_allowed=True, cooling_allowed=None, recent_modes=[]
    ) == (MODE_HEATING, "allowed_entities")
    # Cold but recent days were hot water only: heating is off.
    assert choose_forecast_mode(
        model, 3.0, heating_allowed=None, cooling_allowed=None,
        recent_modes=[MODE_HOT_WATER] * 5,
    ) == (MODE_HOT_WATER, "recent_days")
    assert choose_forecast_mode(
        model, 20.0, heating_allowed=True, cooling_allowed=False, recent_modes=[]
    )[0] == MODE_HOT_WATER


def test_distribution_keeps_the_total_and_leans_to_cold_hours():
    shape = [1.0 / INTERVAL_COUNT] * INTERVAL_COUNT
    temps = [0.0] * 6 + [10.0] * 18
    intervals = distribute_day(
        24.0, shape, temps, MODE_HEATING, stop_heating_c=17.0, start_cooling_c=24.0
    )
    assert sum(intervals) == pytest.approx(24.0)
    assert intervals[0] > intervals[60]
    flat = distribute_day(
        24.0, shape, temps, MODE_HOT_WATER, stop_heating_c=17.0, start_cooling_c=24.0
    )
    assert flat[0] == pytest.approx(flat[60])


def test_allowed_state_and_forecast_parsing():
    assert parse_allowed_state(SimpleNamespace(state="1")) is True
    assert parse_allowed_state(SimpleNamespace(state="0")) is False
    assert parse_allowed_state(SimpleNamespace(state="on")) is True
    assert parse_allowed_state(SimpleNamespace(state="unavailable")) is None
    response = {
        WEATHER: {
            "forecast": [
                {"datetime": "2026-11-16T01:00:00+00:00", "temperature": 3.5},
                {"datetime": "2026-11-16T00:00:00+00:00", "temperature": 4.0},
                {"datetime": "bad", "temperature": 1.0},
            ]
        }
    }
    parsed = parse_forecast_response(response, WEATHER)
    assert [value for _, value in parsed] == [4.0, 3.5]


# ---------------------------------------------------------------------------
# Multi-sensor summing and statistics
# ---------------------------------------------------------------------------


def _hourly_rows(day_count, *, first=date(2026, 11, 10), value=0.3, skip=()):
    rows = []
    for day in range(day_count):
        local = datetime.combine(first + timedelta(days=day), time.min, tzinfo=AMSTERDAM)
        for hour in range(24):
            start = (local + timedelta(hours=hour)).timestamp()
            if (day, hour) in skip:
                continue
            rows.append({"start": start, "end": start + 3600.0, "mean": value})
    return rows


def test_hourly_statistics_sum_all_sensors_and_drop_partial_hours():
    rows = {
        OUTDOOR_UNIT: _hourly_rows(1, value=0.3),
        INDOOR_UNIT: _hourly_rows(1, value=0.2, skip={(0, 5)}),
        OUTDOOR_TEMP: _hourly_rows(1, value=4.0),
    }
    energy, temperature = aggregate_hourly_statistics(
        rows, [OUTDOOR_UNIT, INDOOR_UNIT], OUTDOOR_TEMP
    )
    assert len(energy) == 23  # the hour one sensor missed is not half a value
    assert all(value == pytest.approx(0.5) for value in energy.values())
    days = days_from_hourly(energy, temperature, [date(2026, 11, 10)], AMSTERDAM)
    day = days[date(2026, 11, 10)]
    assert day.energy_kwh == pytest.approx(12.0)  # scaled for the missing hour
    assert day.temp_c == pytest.approx(4.0)
    assert sum(day.quarters_kwh) == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Fake Home Assistant
# ---------------------------------------------------------------------------


class FakeStore:
    saved: dict = {}

    def __init__(self, hass, version, key, **_kwargs):
        self.key = key

    async def async_load(self):
        return FakeStore.saved.get(self.key)

    async def async_save(self, data):
        FakeStore.saved[self.key] = json.loads(json.dumps(data, default=str))

    def async_delay_save(self, data_func, delay=0):
        FakeStore.saved[self.key] = data_func()


def _state(value, unit=None, **attributes):
    if unit is not None:
        attributes["unit_of_measurement"] = unit
    return SimpleNamespace(state=str(value), attributes=attributes)


class FakeServices:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def async_call(self, domain, service, data, **kwargs):
        self.calls.append((domain, service, data, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def fake_env(monkeypatch):
    FakeStore.saved = {}
    for module in (profile_module, tracker_module, hp_module, solar_module):
        monkeypatch.setattr(module, "Store", FakeStore)
    clock = {"now": NOW}
    monkeypatch.setattr(dt_util, "now", lambda *args, **kwargs: clock["now"])
    states: dict = {}
    hass = SimpleNamespace(
        config=SimpleNamespace(
            time_zone="Europe/Amsterdam", latitude=52.0, longitude=5.0
        ),
        states=SimpleNamespace(get=states.get),
        services=FakeServices(),
        data={},
    )
    return SimpleNamespace(hass=hass, states=states, clock=clock)


def _controller():
    return SimpleNamespace(
        consumption_sensor=None,
        solar_production_sensor=None,
        _external_loads=None,
        charging_time_slots=[],
        get_avg_daily_consumption=lambda: 10.0,
        get_adjusted_home_power_kw=lambda: 0.4,
        vacation_mode_enabled=False,
        _daily_consumption_history=[],
        coordinators=[],
    )


def _entry(data=None):
    return SimpleNamespace(entry_id="entry1", data=dict(data or {}))


HEAT_PUMP_DATA = {
    CONF_HEAT_PUMP_POWER_SENSORS: [OUTDOOR_UNIT, INDOOR_UNIT],
    CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR: OUTDOOR_TEMP,
    CONF_HEAT_PUMP_WEATHER_ENTITY: WEATHER,
}


def test_power_sensors_are_summed_and_one_gap_voids_the_sum(fake_env):
    tracker = HeatPumpProfileTracker(fake_env.hass, _entry(HEAT_PUMP_DATA))
    fake_env.states[OUTDOOR_UNIT] = _state(1200, "W")
    fake_env.states[INDOOR_UNIT] = _state(0.3, "kW")
    assert tracker.read_power_kw() == pytest.approx(1.5)
    fake_env.states[INDOOR_UNIT] = _state("unavailable", "kW")
    assert tracker.read_power_kw() is None


def test_live_capture_closes_a_day_with_its_explicit_mode(fake_env):
    data = dict(HEAT_PUMP_DATA)
    data[CONF_HEAT_PUMP_HEATING_ALLOWED_ENTITY] = HEATING_FLAG
    tracker = HeatPumpProfileTracker(fake_env.hass, _entry(data))
    fake_env.states.update(
        {
            OUTDOOR_UNIT: _state(600, "W"),
            INDOOR_UNIT: _state(400, "W"),
            OUTDOOR_TEMP: _state(5.0, "°C"),
            HEATING_FLAG: _state(1),
        }
    )
    start = datetime(2026, 11, 14, 0, 0, tzinfo=AMSTERDAM)
    for minute in range(24 * 60 + 2):
        tracker.record_sample(
            local_time=start + timedelta(minutes=minute), monotonic_time=minute * 60.0
        )
    day = tracker._days[date(2026, 11, 14)]
    assert day.energy_kwh == pytest.approx(24.0, rel=0.01)
    assert day.temp_c == pytest.approx(5.0)
    assert day.mode == MODE_HEATING
    assert day.source == "live"


def test_statistics_backfill_with_fakes(fake_env):
    tracker = HeatPumpProfileTracker(fake_env.hass, _entry(HEAT_PUMP_DATA))
    tracker._pending_modes[date(2026, 11, 11)] = MODE_HEATING
    rows = {
        OUTDOOR_UNIT: _hourly_rows(4, value=0.3),
        INDOOR_UNIT: _hourly_rows(4, value=0.2, skip={(3, h) for h in range(6)}),
        OUTDOOR_TEMP: _hourly_rows(4, value=3.0),
    }

    class Coordinator:
        def __init__(self):
            self.calls = []

        def new_token(self):
            return SimpleNamespace(is_valid=lambda: True)

        async def async_query_statistics(self, token, ids, start, end, **kwargs):
            self.calls.append((set(ids), kwargs))
            return {
                key: [r for r in value if start.timestamp() <= r["start"] < end.timestamp()]
                for key, value in rows.items()
            }

    coordinator = Coordinator()
    tracker.set_backfill_coordinator(coordinator)
    assert asyncio.run(tracker.async_backfill_from_statistics()) is True
    assert coordinator.calls
    ids, kwargs = coordinator.calls[0]
    assert ids == {OUTDOOR_UNIT, INDOOR_UNIT, OUTDOOR_TEMP}
    assert kwargs["period"] == "hour"
    assert kwargs["types"] == {"mean"}
    assert kwargs["units"] == {"power": "kW", "temperature": "°C"}
    assert set(tracker._days) == {date(2026, 11, d) for d in (10, 11, 12)}
    assert tracker._days[date(2026, 11, 10)].energy_kwh == pytest.approx(12.0)
    assert tracker._days[date(2026, 11, 10)].temp_c == pytest.approx(3.0)
    # A mode seen live while the day had gaps survives the statistics fill.
    assert tracker._days[date(2026, 11, 11)].mode == MODE_HEATING
    assert tracker._days[date(2026, 11, 12)].mode is None
    saved = FakeStore.saved["omnibattery.entry1.heat_pump_profile"]
    assert len(saved["days"]) == 3


def test_missing_weather_forecast_falls_back_to_measured_last_24h(fake_env):
    fake_env.hass.services = FakeServices(error=RuntimeError("weather down"))
    tracker = HeatPumpProfileTracker(fake_env.hass, _entry(HEAT_PUMP_DATA))
    now = NOW
    for hours_ago in range(1, 25):
        start = (now - timedelta(hours=hours_ago)).replace(minute=0)
        tracker._hourly_temp[start.timestamp()] = [float(start.hour) * 3600.0, 3600.0]
    assert asyncio.run(tracker.async_refresh_weather()) is False
    assert "weather down" in tracker._forecast_error
    temps, source = tracker.hourly_temperatures(date(2026, 11, 16), now)
    assert source == "measured_last_24h"
    assert temps == [float(hour) for hour in range(24)]

    fake_env.hass.services = FakeServices(
        response={
            WEATHER: {
                "forecast": [
                    {
                        "datetime": (
                            datetime(2026, 11, 16, 0, tzinfo=AMSTERDAM) + timedelta(hours=h)
                        ).isoformat(),
                        "temperature": -1.0,
                    }
                    for h in range(24)
                ]
            }
        }
    )
    assert asyncio.run(tracker.async_refresh_weather()) is True
    assert fake_env.hass.services.calls[0][:3] == (
        "weather", "get_forecasts", {"entity_id": WEATHER, "type": "hourly"}
    )
    temps, source = tracker.hourly_temperatures(date(2026, 11, 16), now)
    assert source == "weather_forecast"
    assert temps == [-1.0] * 24


# ---------------------------------------------------------------------------
# Shadow forecast and isolation from the existing forecast
# ---------------------------------------------------------------------------


def _profile_day(local_date, base_kwh_per_quarter):
    energy = [
        base_kwh_per_quarter * (1.0 + 0.5 * ((index // 4) % 6 == 0))
        for index in range(INTERVAL_COUNT)
    ]
    return ProfileDay(local_date, energy, [INTERVAL_SECONDS] * INTERVAL_COUNT, complete=True)


def _fill(profile, value):
    for offset in range(1, 15):
        local_date = NOW.date() - timedelta(days=offset)
        profile._days[local_date] = _profile_day(local_date, value)
    profile._invalidate_forecast_cache()


def _ranges():
    midnight = datetime.combine(NOW.date(), time.min, tzinfo=AMSTERDAM)
    return [
        (midnight, midnight + timedelta(days=1)),
        (midnight + timedelta(days=1), midnight + timedelta(days=2)),
        (NOW, NOW + timedelta(hours=20)),
        (NOW + timedelta(hours=3, minutes=7), NOW + timedelta(hours=5)),
    ]


def _serialize(forecast):
    # repr keeps every float exactly, and the field order is the dataclass's.
    return repr(dataclasses.asdict(forecast))


def _all_current_forecasts(tracker):
    out = []
    for start, end in _ranges():
        for fallback in ("legacy_daily", "current_rate"):
            out.append(_serialize(tracker.forecast_consumption_between(start, end, fallback=fallback)))
            out.append(
                _serialize(
                    tracker.consumption_profile.forecast_energy_between(
                        start, end, exclude_charging_windows=True, fallback=fallback
                    )
                )
            )
    out.append(_serialize(tracker.forecast_consumption_for_date(NOW.date())))
    return out


def _mature_heat_pump(tracker):
    temps = [2.0 + (i % 9) for i in range(30)]
    days = _series(
        temps,
        lambda offset, t: 6.0 + 1.25 * (17.0 - t),
        start=NOW.date() - timedelta(days=30),
    )
    for day in days:
        day.quarters_kwh = [day.energy_kwh / INTERVAL_COUNT] * INTERVAL_COUNT
        tracker._days[day.local_date] = day
    tracker._forecast = [
        ((datetime.combine(NOW.date(), time.min, tzinfo=AMSTERDAM) + timedelta(hours=h)).timestamp(), 2.0)
        for h in range(48)
    ]
    tracker._model = None


def test_no_heat_pump_configured_builds_nothing(fake_env):
    tracker = ConsumptionTracker(fake_env.hass, _entry(), _controller())
    assert tracker.heat_pump_profile is None
    assert tracker.base_consumption_profile is None
    assert tracker.heat_pump_diagnostics() == {"configured": False}
    _fill(tracker.consumption_profile, 0.1)
    forecast = tracker.forecast_consumption_with_heat_pump_between(*_ranges()[0])
    current = tracker.forecast_consumption_between(*_ranges()[0])
    assert _serialize(forecast) == _serialize(current)
    # Recording a sample touches no heat pump state.
    asyncio.run(tracker.accumulate_household_consumption())
    assert not any("heat_pump" in key or "base" in key for key in FakeStore.saved)


def test_existing_forecasts_are_byte_for_byte_unchanged(fake_env):
    plain = ConsumptionTracker(fake_env.hass, _entry(), _controller())
    _fill(plain.consumption_profile, 0.1)
    baseline = _all_current_forecasts(plain)

    shadow = ConsumptionTracker(fake_env.hass, _entry(HEAT_PUMP_DATA), _controller())
    assert shadow.heat_pump_profile is not None
    _fill(shadow.consumption_profile, 0.1)
    _fill(shadow.base_consumption_profile, 0.05)
    _mature_heat_pump(shadow.heat_pump_profile)
    before = _all_current_forecasts(shadow)
    # Exercise every shadow path, then read the current forecasts again.
    summary = shadow.heat_pump_shadow_summary(NOW)
    assert summary["tomorrow"]["source"] == "heat_pump_model"
    for start, end in _ranges():
        shadow.forecast_consumption_with_heat_pump_between(start, end)
    shadow.heat_pump_diagnostics()
    after = _all_current_forecasts(shadow)

    assert before == baseline
    assert after == baseline
    # The combined profile keeps its own Store key and backfill job.
    assert shadow.consumption_profile.STORE_KEY == "consumption_profile"
    assert shadow.consumption_profile.BACKFILL_JOB_NAME == "consumption_profile"
    assert shadow.base_consumption_profile.STORE_KEY == "consumption_profile_base"


def test_shadow_forecast_adds_the_heat_pump_model_to_the_base_load(fake_env):
    tracker = ConsumptionTracker(fake_env.hass, _entry(HEAT_PUMP_DATA), _controller())
    _fill(tracker.consumption_profile, 0.1)
    _fill(tracker.base_consumption_profile, 0.05)
    _mature_heat_pump(tracker.heat_pump_profile)
    start, end = _ranges()[1]
    shadow, details = tracker.heat_pump_shadow_forecast_between(start, end)
    base = tracker.base_consumption_profile.forecast_energy_between(
        start, end, exclude_charging_windows=False, fallback="legacy_daily"
    )
    expected_heat_pump = 6.0 + 1.25 * (17.0 - 2.0)
    assert shadow.source == "heat_pump_model"
    assert shadow.mature
    assert details["heat_pump_kwh"] == pytest.approx(expected_heat_pump, rel=0.02)
    assert shadow.energy_kwh == pytest.approx(base.energy_kwh + expected_heat_pump, rel=0.01)
    assert sum(shadow.intervals_kwh) == pytest.approx(shadow.energy_kwh)
    tomorrow = start.date()
    day = details["days"][tomorrow.isoformat()]
    assert day["mode"] == MODE_HEATING
    assert day["temperature_source"] == "weather_forecast"


def test_immature_heat_pump_model_falls_back_to_the_current_forecast(fake_env):
    tracker = ConsumptionTracker(fake_env.hass, _entry(HEAT_PUMP_DATA), _controller())
    _fill(tracker.consumption_profile, 0.1)
    _fill(tracker.base_consumption_profile, 0.05)
    _mature_heat_pump(tracker.heat_pump_profile)
    hp = tracker.heat_pump_profile
    for key in sorted(hp._days)[:-5]:
        del hp._days[key]
    hp._model = None
    start, end = _ranges()[1]
    shadow, details = tracker.heat_pump_shadow_forecast_between(start, end)
    current = tracker.forecast_consumption_between(start, end)
    assert _serialize(shadow) == _serialize(current)
    assert details["fallback_reason"].startswith("heat_pump_model_immature")


def test_immature_base_profile_falls_back(fake_env):
    tracker = ConsumptionTracker(fake_env.hass, _entry(HEAT_PUMP_DATA), _controller())
    _fill(tracker.consumption_profile, 0.1)
    _mature_heat_pump(tracker.heat_pump_profile)
    shadow, details = tracker.heat_pump_shadow_forecast_between(*_ranges()[1])
    assert shadow.source != "heat_pump_model"
    assert details["fallback_reason"].startswith("base_profile_immature")


def test_base_load_learns_home_minus_heat_pump(fake_env):
    tracker = ConsumptionTracker(fake_env.hass, _entry(HEAT_PUMP_DATA), _controller())
    fake_env.states.update(
        {
            OUTDOOR_UNIT: _state(1000, "W"),
            INDOOR_UNIT: _state(500, "W"),
            OUTDOOR_TEMP: _state(4.0, "°C"),
        }
    )
    start = NOW.replace(minute=0)
    tracker._record_heat_pump_samples(2.0, start, 1000.0)
    tracker._record_heat_pump_samples(2.0, start + timedelta(minutes=1), 1060.0)
    base_capture = tracker.base_consumption_profile.current_day_capture(NOW.date())
    assert base_capture["energy_kwh"] == pytest.approx(0.5 / 60.0, rel=1e-3)
    assert tracker.consumption_profile.current_day_capture(NOW.date()) is None
    # The Recorder backfill subtracts every heat pump sensor at full weight.
    loads = BaseLoadProfileTracker._backfill_external_loads(
        {**HEAT_PUMP_DATA, "excluded_devices": [{"power_sensor": "sensor.ev", "enabled": True}]}
    )
    assert loads == [("sensor.ev", -1.0), (OUTDOOR_UNIT, -1.0), (INDOOR_UNIT, -1.0)]
    assert ConsumptionProfileTracker._backfill_external_loads(
        {**HEAT_PUMP_DATA, "excluded_devices": [{"power_sensor": "sensor.ev", "enabled": True}]}
    ) == [("sensor.ev", -1.0)]


def test_forecast_history_carries_the_shadow_series_separately():
    class Store:
        def async_delay_save(self, *_args):
            return None

    history = ForecastHistoryManager(
        SimpleNamespace(), SimpleNamespace(entry_id="x"), store=Store(),
        now_provider=lambda: NOW,
    )
    before = json.dumps(history._payload(), sort_keys=True)
    shadow = {
        "source": "heat_pump_model",
        "shadow_kwh": 30.0,
        "current_kwh": 25.0,
        "base_kwh": 9.0,
        "heat_pump_kwh": 21.0,
        "intervals_kwh": [30.0 / INTERVAL_COUNT] * INTERVAL_COUNT,
        "heat_pump_intervals_kwh": [21.0 / INTERVAL_COUNT] * INTERVAL_COUNT,
        "current_intervals_kwh": [25.0 / INTERVAL_COUNT] * INTERVAL_COUNT,
        "days": {"2026-11-16": {"mode": "heating", "effective_temperature_c": 2.0,
                                "temperature_source": "weather_forecast"}},
    }
    assert history.record_shadow_consumption(date(2026, 11, 16), shadow=shadow, captured_at=NOW)
    record = history._days["2026-11-16"]
    assert set(record) == {"date", "revisions", "shadow_consumption"}
    entry = record["shadow_consumption"]["day_ahead"]
    assert entry["shadow_kwh"] == 30.0 and entry["heat_pump_mode"] == "heating"
    assert sum(entry["heat_pump_series"]) == pytest.approx(21.0, abs=0.01)
    assert sum(entry["current_series"]) == pytest.approx(25.0, abs=0.01)
    # A start-of-day capture is kept once (a fallback captured while the model
    # was loading gives way to the model); the summary is untouched by it.
    fallback = {**shadow, "source": "profile", "fallback_reason": "no_heat_pump_days"}
    assert history.record_shadow_consumption(date(2026, 11, 15), shadow=fallback, captured_at=NOW)
    assert not history.record_shadow_consumption(date(2026, 11, 15), shadow=fallback, captured_at=NOW)
    assert history.record_shadow_consumption(date(2026, 11, 15), shadow=shadow, captured_at=NOW)
    assert not history.record_shadow_consumption(date(2026, 11, 15), shadow=shadow, captured_at=NOW)
    # The list carries a compact block (no series) only for days with a
    # shadow record; a day without one keeps the old summary shape.
    summary = history._summary("2026-11-16", None)
    compact = summary["shadow_consumption"]["day_ahead"]
    assert "series" not in compact and "heat_pump_series" not in compact
    assert compact["heat_pump_mode"] == "heating" and compact["compare"] is None
    plain = history._summary("2026-11-10", None)
    assert "shadow_consumption" not in plain
    detail = history.day_detail("2026-11-16")["shadow_consumption"]["day_ahead"]
    assert detail["shadow_kwh"] == 30.0 and len(detail["heat_pump_series"]) == INTERVAL_COUNT
    assert before != json.dumps(history._payload(), sort_keys=True)

    # Once the day is measured both forecasts are compared over the same quarters.
    history._days["2026-11-16"]["actual"] = {
        "series": {"consumption": [28.0 / INTERVAL_COUNT] * INTERVAL_COUNT},
        "totals": {"consumption": 28.0},
    }
    finished = history._summary("2026-11-16", None)["shadow_consumption"]["day_ahead"]
    assert finished["compare"]["error"] == pytest.approx(2.0, abs=0.01)
    assert finished["current_compare"]["error"] == pytest.approx(-3.0, abs=0.01)


def test_control_and_planner_code_never_reads_the_shadow_forecast():
    package = Path(__file__).resolve().parents[1] / "custom_components" / "omnibattery"
    tokens = (
        "forecast_consumption_with_heat_pump_between",
        "heat_pump_shadow",
        "base_consumption_profile",
        "heat_pump_profile",
        "HeatPumpProfileTracker",
    )
    allowed = {
        "tracking/consumption_tracker.py",
        "tracking/heat_pump_profile.py",
        "sensor.py",
        "diagnostics.py",
        "__init__.py",
    }
    offenders = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(package).as_posix()
        if relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in tokens):
            offenders.append(relative)
    assert offenders == []
    init_text = (package / "__init__.py").read_text(encoding="utf-8")
    for token in ("forecast_consumption_with_heat_pump_between", "heat_pump_shadow"):
        assert token not in init_text


def test_options_flow_validation():
    from custom_components.omnibattery.config_flow import _validate_heat_pump_input

    states = {
        OUTDOOR_UNIT: _state(100, "W"),
        INDOOR_UNIT: _state(0.1, "kW"),
        OUTDOOR_TEMP: _state(4.0, "°C"),
        "sensor.energy": _state(10, "kWh"),
    }
    hass = SimpleNamespace(states=SimpleNamespace(get=states.get))
    assert _validate_heat_pump_input(hass, {}, {}) == {}
    assert _validate_heat_pump_input(hass, HEAT_PUMP_DATA, {}) == {}
    assert _validate_heat_pump_input(
        hass, {CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR: OUTDOOR_TEMP}, {}
    ) == {CONF_HEAT_PUMP_POWER_SENSORS: "heat_pump_power_required"}
    assert _validate_heat_pump_input(
        hass, {CONF_HEAT_PUMP_POWER_SENSORS: ["sensor.energy"], CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR: OUTDOOR_TEMP}, {}
    ) == {CONF_HEAT_PUMP_POWER_SENSORS: "heat_pump_power_invalid_unit"}
    assert _validate_heat_pump_input(
        hass, {CONF_HEAT_PUMP_POWER_SENSORS: [OUTDOOR_UNIT]}, {}
    ) == {CONF_HEAT_PUMP_OUTDOOR_TEMP_SENSOR: "heat_pump_temperature_required"}


def test_shadow_sensor_and_diagnostics_surface_both_forecasts(fake_env):
    from custom_components.omnibattery.diagnostics import _json_safe
    from custom_components.omnibattery.sensor import HeatPumpShadowForecastSensor

    tracker = ConsumptionTracker(fake_env.hass, _entry(HEAT_PUMP_DATA), _controller())
    _fill(tracker.consumption_profile, 0.1)
    _fill(tracker.base_consumption_profile, 0.05)
    _mature_heat_pump(tracker.heat_pump_profile)
    sensor = HeatPumpShadowForecastSensor(SimpleNamespace(_consumption_tracker=tracker))
    attributes = sensor.extra_state_attributes
    assert sensor.native_value == attributes["today_shadow_kwh"]
    assert attributes["tomorrow_source"] == "heat_pump_model"
    assert attributes["tomorrow_current_kwh"] == pytest.approx(
        tracker.forecast_consumption_for_date(NOW.date() + timedelta(days=1)).energy_kwh,
        abs=1e-3,
    )
    assert len(attributes["tomorrow_hourly_shadow_kwh"]) == 24
    diagnostics = _json_safe(tracker.heat_pump_diagnostics())
    json.dumps(diagnostics)
    assert diagnostics["mode"] == "shadow"
    assert diagnostics["model"]["model"]["heating"]["mature"] is True
    assert diagnostics["model"]["temperature_source"] == "weather_forecast"
    assert diagnostics["shadow_tomorrow"]["source"] == "heat_pump_model"


# ---------------------------------------------------------------------------
# Long horizon: last winter in the fit
# ---------------------------------------------------------------------------


def test_age_weight_halves_per_year_but_keeps_a_floor():
    assert hp_module.age_weight(0) == 1.0
    assert hp_module.age_weight(365) == pytest.approx(0.5)
    assert hp_module.age_weight(5000) == hp_module.AGE_WEIGHT_FLOOR
    assert hp_module.SAMPLE_RETENTION_DAYS >= 400


def test_lone_or_mild_active_days_are_uncertain_not_heating():
    rng = random.Random(21)
    # Mild autumn: hot water ~5 kWh, every fourth day a little higher (a
    # legionella cycle, a bath) at 12-15 degC, plus one lone cold active day.
    temps = [rng.uniform(12.0, 15.0) for _ in range(28)] + [19.0] * 6
    temps[10] = 4.0

    def energy(offset, t_eff):
        if offset == 10:
            return 25.0
        return 8.5 if offset % 4 == 0 else 5.0

    model = fit_heat_pump_model(_series(temps, energy), stop_heating_c=17.0)
    modes = [item.mode for item in model.classified]
    assert hp_module.MODE_HEATING not in modes
    assert modes.count(hp_module.MODE_UNCERTAIN) >= 1
    assert model.as_dict()["classified_days"]["heating"] == 0


def test_last_winter_dominates_the_heating_slope_over_mild_inferred_days():
    rng = random.Random(5)
    today = date(2026, 11, 15)
    start = today - timedelta(days=399)
    days = []
    previous = None
    for offset in range(400):
        local_date = start + timedelta(days=offset)
        month = local_date.month
        if month in (12, 1, 2):
            temp = rng.uniform(-5.0, 8.0)
        elif month in (3, 11):
            temp = rng.uniform(2.0, 12.0)
        elif month in (4, 10):
            temp = rng.uniform(9.0, 15.0)
        else:
            temp = rng.uniform(17.5, 23.0)
        t_eff = effective_temperature(temp, previous)
        previous = temp
        heating_on = local_date < date(2026, 3, 31) or local_date >= date(2026, 11, 5)
        if heating_on and t_eff < 17.0:
            energy = 6.0 + 1.25 * (17.0 - t_eff) + rng.gauss(0.0, 1.5)
        elif month in (9, 10) and offset % 3 == 0:
            # Mild autumn days a bit over the hot water level, heating off.
            energy = 8.0 + rng.uniform(0.0, 1.0)
        else:
            energy = 5.0 + rng.gauss(0.0, 0.4)
        mode = MODE_HEATING if local_date >= date(2026, 11, 5) else None
        days.append(HeatPumpDay(local_date, energy, temp, mode=mode))

    model = fit_heat_pump_model(days, stop_heating_c=17.0)
    assert model.heating.mature
    assert model.heating.slope_kwh_per_k == pytest.approx(1.25, abs=0.15)
    # About 20 kWh at -3..0 degC, as measured on real winter days.
    assert model.predict(-1.5, MODE_HEATING) == pytest.approx(6.0 + 1.25 * 18.5, rel=0.12)
    autumn = [
        item
        for item in model.classified
        if item.day.local_date.year == 2026
        and item.day.local_date.month in (9, 10)
    ]
    assert autumn and not any(item.mode == MODE_HEATING for item in autumn)

    # The 120-day window the first build kept has only the new season and
    # the autumn: it cannot learn the winter slope.
    recent = [day for day in days if day.local_date >= today - timedelta(days=120)]
    short = fit_heat_pump_model(recent, stop_heating_c=17.0)
    assert short.heating.days < model.heating.days


def test_an_existing_120_day_store_is_extended_by_the_next_backfill(fake_env):
    tracker = HeatPumpProfileTracker(fake_env.hass, _entry(HEAT_PUMP_DATA))
    today = NOW.date()
    stored_days = []
    for offset in range(1, 121):
        local_date = today - timedelta(days=offset)
        stored_days.append(
            HeatPumpDay(
                local_date, 5.0, 15.0,
                quarters_kwh=[5.0 / INTERVAL_COUNT] * INTERVAL_COUNT,
            ).as_dict()
        )
    FakeStore.saved["omnibattery.entry1.heat_pump_profile"] = {
        "capture_version": hp_module.CAPTURE_VERSION,
        "sensors_fingerprint": tracker.sensors_fingerprint(),
        "days": stored_days,
    }
    assert asyncio.run(tracker.async_load()) is True
    assert len(tracker._days) == 120
    missing = tracker._dates_needing_backfill(today)
    assert min(missing) == today - timedelta(days=hp_module.SAMPLE_RETENTION_DAYS)
    assert max(missing) < today - timedelta(days=120)
    assert len(missing) == hp_module.SAMPLE_RETENTION_DAYS - 120

    first = today - timedelta(days=hp_module.SAMPLE_RETENTION_DAYS)
    count = hp_module.SAMPLE_RETENTION_DAYS - 120
    rows = {
        OUTDOOR_UNIT: _hourly_rows(count, first=first, value=0.6),
        INDOOR_UNIT: _hourly_rows(count, first=first, value=0.2),
        OUTDOOR_TEMP: _hourly_rows(count, first=first, value=-1.0),
    }

    class Coordinator:
        def __init__(self):
            self.calls = 0

        def new_token(self):
            return SimpleNamespace(is_valid=lambda: True)

        async def async_query_statistics(self, token, ids, start, end, **kwargs):
            self.calls += 1
            assert (end - start) <= timedelta(days=hp_module.STATISTICS_CHUNK_DAYS + 1)
            return {
                key: [r for r in value if start.timestamp() <= r["start"] < end.timestamp()]
                for key, value in rows.items()
            }

    coordinator = Coordinator()
    tracker.set_backfill_coordinator(coordinator)
    assert asyncio.run(tracker.async_backfill_from_statistics()) is True
    assert len(tracker._days) == hp_module.SAMPLE_RETENTION_DAYS
    assert coordinator.calls == -(-count // hp_module.STATISTICS_CHUNK_DAYS)
    assert tracker._dates_needing_backfill(today) == []
    assert tracker._days[first].energy_kwh == pytest.approx(19.2)
    # Old days keep their total but drop the quarter-hour shape.
    saved = FakeStore.saved["omnibattery.entry1.heat_pump_profile"]
    with_shape = [day for day in saved["days"] if day["quarters_kwh"] is not None]
    assert len(with_shape) <= hp_module.SHAPE_RETENTION_DAYS
    assert len(json.dumps(saved)) < 400_000
