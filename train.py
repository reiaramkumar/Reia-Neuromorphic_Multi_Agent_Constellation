import sys
sys.path.insert(0, ".")
import os
import csv
import json
import time as _time
import numpy as np
import torch
import yaml
from pipeline_classes import *
try:
    from global_land_mask import globe
    HAVE_LAND_MASK = True
except ImportError:
    HAVE_LAND_MASK = False

DATA_DIR = "pipeline_outputs/data"

# ......................................................................................................................
#                                                   DATASETS
# ......................................................................................................................

def _is_ocean(lat, lon):
    if not HAVE_LAND_MASK:
        return True
    return not globe.is_land(lat, lon)

def _perturb_near_real_case(base_case, rng, max_retries = 20):
    for attempt in range(max_retries):
        lat = base_case["lat"] +  rng.uniform(-3.0, 3.0)
        lon = base_case["lon"] + rng.uniform(-3.0, 3.0)
        if _is_ocean(lat, lon):
            area = base_case["area_km2"] *  rng.uniform(0.7, 1.3)
            return {"lat": float(lat), "lon": float(lon), "area_km2": float(area), "anomaly": True, "based_on": base_case["name"]}
    return {"lat": base_case["lat"], "lon": base_case["lon"], "area_km2": base_case["area_km2"], "anomaly": True, "based_on": base_case["name"]}

def _sample_normal_case(rng, max_retries = 20):
    for attempt in range(max_retries):
        case = sample_training_case(rng)
        if _is_ocean(case["lat"], case["lon"]):
            return {"lat": float(case["lat"]), "lon": float(case["lon"]),
                    "area_km2": float(case["area_km2"]), "anomaly": False,
                    "region": case["region"], "date": case["date"]}
    return {"lat": float(case["lat"]), "lon": float(case["lon"]),
                    "area_km2": float(case["area_km2"]), "anomaly": False,
                    "region": case["region"], "date": case["date"]}

def generate_training_set(seed = 0, num_normal = 35, num_anomalous = 25):
    rng = np.random.default_rng(seed)
    cases = {}
    for i in range(num_normal):
        case = _sample_normal_case(rng)
        cases[f"case_{i + 1:03d}"] = case
    base_events = rng.choice(TABLE_1_ALL_CASES, size = num_anomalous, replace = (num_anomalous > len(TABLE_1_ALL_CASES)))
    for i, base in enumerate(base_events):
        case = _perturb_near_real_case(base, rng)
        cases[f"case_{num_normal + i + 1:03d}"] = case
    n_anom = sum(1 for c in cases.values() if c["anomaly"])
    print(f"Generated {len(cases)} training cases ({len(cases) - n_anom} normal, {n_anom} anomalous)"
          f"{' [land mask disabled & ocean-only NOT enforced]' if not HAVE_LAND_MASK else ''}")
    return cases


def save_all_cases_yaml(training_cases, path = f"{DATA_DIR}/all_cases.yaml"):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    doc = {
        "train": training_cases,
        "test": {c["name"]: {"lat": c["lat"], "lon": c["lon"], "area_km2": c["area_km2"],
                              "start_date": c["start"], "anomaly": False} for c in TEST_CASES},
        "validation": {c["name"]: {"lat": c["lat"], "lon": c["lon"], "area_km2": c["area_km2"],
                                    "start_date": c["start"], "anomaly": False} for c in VALIDATION_CASES},
    }

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style = False, sort_keys = False)
        print(f"Saved {path}")
        return doc

# ......................................................................................................................
#                                                   SHARED RUN CASE
# ......................................................................................................................

def get_real_or_fallback_field(case_lat, case_lon, case_date, use_real_currents=True,
                                max_retries=3, retry_backoff_sec=5.0):
    coords = np.linspace(-GRID_LIMIT_KM, GRID_LIMIT_KM, 200)
    X, Y = np.meshgrid(coords, coords)

    if use_real_currents:
        last_error = None
        for attempt in range(max_retries):
            try:
                u_field, v_field = fetch_hycom_current_field(X, Y, ref_lat=case_lat, ref_lon=case_lon,
                                                             date=case_date, timeout_note=False)
                return u_field, v_field
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    _time.sleep(retry_backoff_sec)
                    print(f"[WARNING] HYCOM fetch failed for ({case_lat:.2f},{case_lon:.2f}) after {max_retries} attempts ({last_error}) hence using synthetic eddy fallback.")

    rng_local = np.random.default_rng(abs(int(case_lat * 1000 + case_lon * 1000)) % 10000)
    dx = X[0, 1] - X[0, 0]
    psi = np.zeros_like(X)
    for _ in range(4):
        cx, cy = rng_local.uniform(-300, 300, size=2)
        strength = rng_local.uniform(-4000, 4000)
        sigma = rng_local.uniform(80, 160)
        psi += strength * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    u_field = np.gradient(psi, dx, axis=0)
    v_field = -np.gradient(psi, dx, axis=1)
    return u_field, v_field

def _sat_name(idx):
    return "A" if idx ==0 else("B" if idx ==1 else idx)

def run_case(case_lat, case_lon, case_date, area_km2, snn, encoder, trainer, buffer,
             num_steps, is_training, kepler_log, sac_loss_log, snapshot_log,
             u_field=None, v_field=None, take_snapshots=False, use_real_currents=True,
             num_satellites=2, fuel_weight=0.5, visibility_bonus=0.1,
             altitude_km=824.0, raan_clustered_spread_deg=None, swath_half_width_km=400.0):
    if u_field is None:
        u_field, v_field = get_real_or_fallback_field(case_lat, case_lon, case_date, use_real_currents)

    custom_params = bloom_params(max_area_spread_km2 = area_km2)
    env = TwoSatelliteBloomEnvironment( u_field, v_field, params=custom_params, num_satellites=num_satellites,
                                        altitude_km=altitude_km, raan_clustered_spread_deg=raan_clustered_spread_deg,
                                        swath_half_width_km=swath_half_width_km, origin_ref_lat=case_lat,
                                        origin_ref_lon=case_lon, origin_date=case_date, refetch_on_recenter=False)
    imap =InterestMap(env.X, env.Y)
    names = [_sat_name(i) for i in range(num_satellites)]
    total_reward = 0.0
    visible_steps = {name: 0 for name in names}
    prediction_errors = []
    frame_history = {name: [] for name in names}

    for step in range(num_steps):
        env._step(dt_step_day= 1.0/ 24.0)
        visibility = env.visible_satellites()
        true_state = np.array([env.state.bloom_center_km[0], env.state.bloom_center_km[1], env.state.bloom_area_km2])

        obs_list, actions_list = [],[]
        for name in names:
            elements = env.satellites[env._resolve_index(name)]
            obs = build_observation(elements, imap, case_lat, case_lon)
            obs_list.append(obs)

            if visibility[name]:
                visible_steps[name] += 1
                frame = env.get_satellite_view(name, visibility)
                ref_frame = pick_reference_frame(frame_history[name], env.state.current_bloom_day)
                pred_norm = snn_predict(snn, encoder, frame, env, prev_frame=ref_frame)
                frame_history[name].append((env.state.current_bloom_day, frame))

                if pred_norm is not None:
                    pred_x, pred_y = pred_norm[0] * GRID_LIMIT_KM, pred_norm[1] * GRID_LIMIT_KM
                    imap.deposit(pred_x, pred_y, amount=1.0)
                    prediction_errors.append(float(np.linalg.norm(np.array([pred_x, pred_y]) - env.state.bloom_center_km)))

            obs_tensor = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = trainer.actor.sample(obs_tensor)
            actions_list.append(action.squeeze(0).numpy())
            env.apply_satellite_action(name, action.squeeze(0).numpy())


        imap.evaporate()
        is_visible_any = any(visibility[name] for name in names)
        reward = compute_reward(predicted_state, true_state, np.mean(actions_list, axis=0), is_visible_any,
                                fuel_weight=fuel_weight, visibility_bonus=visibility_bonus,
                                state_scale=STATE_SCALE)
        reward += env.collision_penalty()
        total_reward += reward

        if is_training:
            joint_action = np.concatenate(actions_list)
            next_obs_list = [build_observation(env.satellites[env._resolve_index(n)], imap, case_lat, case_lon)
                             for n in names]
            buffer.push(obs_list, joint_action, reward, next_obs_list, done = env._is_dead(env.state.current_bloom_day))
            if len(buffer) >= 16:
                losses = trainer.update(buffer.sample(16, num_satellites))
                sac_loss_log["critic"].append(losses["critic_loss"])
                sac_loss_log["actor"].append(losses["actor_loss"])

        for name in names:
            elem = env.satellites[env._resolve_index(name)]
            kepler_log[name].append({"a_km": elem.a_km, "raan_deg": elem.raan_deg,
                              "M_deg": elem.mean_anomaly_deg, "step": step})

        snapshot_interval = max(1, num_steps//10)
        if take_snapshots and step % snapshot_interval == 0:
            snapshot_log.append({
                "step": step, "bloom_field": env._rendering().copy(),
                "interest_grid": imap.grid.copy(),
                "bloom_center_km": env.state.bloom_center_km.copy(),
            })

    result = {
        "total_reward": float(total_reward), "reward_per_step": float(total_reward / num_steps),
        "num_steps": num_steps, "mean_prediction_error_km": float(np.mean(prediction_errors)) if prediction_errors else None,
        "num_predictions_made": len(prediction_errors),
        "final_true_area_km2": float(env.state.bloom_area_km2),
        "num_satellites": num_satellites,
    }
    for name in names:
        result[f"visible_steps_{name}"] = visible_steps[name]
    return result




if __name__ == "__main__":
    training_cases = generate_training_set(seed=0)
    save_all_cases_yaml(training_cases)
