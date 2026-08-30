# IMPORTS
import sys
sys.path.insert(0, ".")
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spikingjelly.activation_based import functional
from pipeline_classes import *
from train import run_case, get_real_or_fallback_field


# DIRECTORIES
SENSITIVITY_DIR = "pipeline_outputs/sensitivity"
DATA_DIR = "pipeline_outputs/data"
BEST_CONFIG_DIR = "pipeline_outputs/best_config"
CASES_DIR = "pipeline_outputs/cases"
os.makedirs(BEST_CONFIG_DIR, exist_ok=True)

# CONSTANTS
NUM_TRAIN_STEPS_PER_CASE = 1080
NUM_EVAL_STEPS = 1080
NUM_SNAPSHOTS = 10
USE_REAL_CURRENTS = True

EXPECTED_FILES = [
    "bloom_filmstrip.png", "interest_map_filmstrip.png", "snn_spike_activity.png",
    "sac_reward_trace.png", "sac_tracking_overlay.png", "sac_critic_q_trace.png",
    "sac_actor_entropy.png", "snn_activity_context.png",
]

# DEFAULTS
# SNN & SAC
BASE_SNN_DEFAULTS = {"v_threshold": 0.3, "tau": 1.5, "weight_scale": 2.0}
BASE_SAC_DEFAULTS = {"lr": 3e-4, "gamma": 0.99, "tau": 0.005, "alpha": 0.2}
BASE_FUEL_WEIGHT = 0.5  # must match compute_reward's own default exactly
SNN_FOLDABLE_PARAMS = {"v_threshold": "v_threshold", "tau": "tau", "weight_scale": "weight_scale"}
SAC_FOLDABLE_PARAMS = {"learning_rate": "lr", "target_tau": "tau", "alpha": "alpha", "gamma": "gamma"}


# ......................................................................................................................
#                                       BUILD THE BEST CONFIG
# ......................................................................................................................
def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

def _best_value_per_parameter(sensitivity_data):
    best = {}
    for parameter, param_data in sensitivity_data.items():
        best_val, best_score = None, -np.inf
        for value, case_data in param_data.items():
            rewards = [c.get("reward_per_step") for c in case_data.values() if c.get("reward_per_step") is not None]
            if not rewards:
                continue
            mean_reward = float(np.mean(rewards))
            if mean_reward > best_score:
                best_score, best_val = mean_reward, value
        best[parameter] = {"value": best_val, "mean_reward_per_step": best_score}
    return best

def build_best_config(snn_sensitivity, sac_sensitivity):
    snn_best = _best_value_per_parameter(snn_sensitivity)
    sac_best = _best_value_per_parameter(sac_sensitivity)

    snn_kwargs = dict(BASE_SNN_DEFAULTS)
    snn_notes = []
    for param, sweep_name in SNN_FOLDABLE_PARAMS.items():
        if param in snn_best:
            snn_kwargs[sweep_name] = snn_best[param]["value"]

    for param in ("network_depth", "surrogate_function", "optimizer", "dropout_rate"):
        if param in snn_best:
            snn_notes.append(f"[PARAM EXCLUDED] {param} best value was '{snn_best[param]['value']}, varies arch so nah :(")

    sac_kwargs = dict(BASE_SAC_DEFAULTS)
    sac_notes = []
    for param, sweep_name in SAC_FOLDABLE_PARAMS.items():
        if param in sac_best:
            sac_kwargs[sweep_name] = sac_best[param]["value"]
    for param in ("activation_function", "optimizer", "num_agents"):
        if param in sac_best:
            sac_notes.append(f"[PARAM EXCLUDED] {param} best value was '{sac_best[param]['value']} , varies arch so nah :(")

    best_fuel_weight = BASE_FUEL_WEIGHT
    if "fuel_weight" in sac_best and sac_best["fuel_weight"]["value"] is not None:
        best_fuel_weight = float(sac_best["fuel_weight"]["value"])

    return {
        "snn_kwargs": snn_kwargs, "sac_kwargs": sac_kwargs,
        "snn_best_raw": snn_best, "sac_best_raw": sac_best,
        "snn_notes": snn_notes, "sac_notes": sac_notes,
        "fuel_weight": best_fuel_weight,
    }

# ......................................................................................................................
#                                       TRAINING
# ......................................................................................................................

def train_snn_on_training_set(snn_kwargs, training_cases, seed = 0, samples_per_case = 10):
    torch.manual_seed(seed)
    snn = SNN(**snn_kwargs)
    optimizer = torch.optim.Adam(snn.parameters(), lr = 3e-5)
    encoder = DVS_Encoder(rng = np.random.default_rng(seed))
    losses = []
    n_cases = len(training_cases)

    for i, (case_id, case) in enumerate(training_cases.items()):
        u,v_ = get_real_or_fallback_field(case["lat"], case["lon"], case.get("date", "2020-01-01"),
                                         use_real_currents = USE_REAL_CURRENTS)
        env = HycomBloomEnvironment(u, v_, params=bloom_params(max_area_spread_km2=case["area_km2"]),
                                     origin_ref_lat=case["lat"], origin_ref_lon=case["lon"],
                                     origin_date=case.get("date", "2020-01-01"), refetch_on_recenter=False)
        for _ in range(samples_per_case):
            env._step(dt_step_day= 1.0)
            events = encoder.encode(env, dt_days = 1.0)

            if events.shape[0] == 0:
                continue
            events_tensor = torch.tensor(events, dtype = torch.float32 )
            target = torch.tensor([env.state.bloom_center_km[0] / 400.0,
                                        env.state.bloom_center_km[1] / 400.0,
                                        env.state.bloom_area_km2 / 45000.0], dtype=torch.float32)

            functional.reset_net(snn)
            optimizer.zero_grad()
            pred = snn(events_tensor)
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 0.5)
            optimizer.step()
            losses.append(loss.item())

        if (i + 1) % 10 == 0 or i + 1 == n_cases:
            print(f"    SNN training: {i+1}/{n_cases} cases, running mean loss={np.mean(losses[-samples_per_case:]):.4f}")

    return snn, losses



def train_sac_on_training_set(snn, sac_kwargs, training_cases, seed=0, num_steps=NUM_TRAIN_STEPS_PER_CASE,
                               fuel_weight=BASE_FUEL_WEIGHT):
    torch.manual_seed(seed)
    trainer = SAC_Trainer(num_agents=2, **sac_kwargs)
    encoder = DVS_Encoder()
    buffer = ReplayBuffer(capacity = 5000)
    n_cases = len(training_cases)

    for i, (case_id, case) in enumerate(training_cases.items()):
        kepler_log = {"A": [], "B": []}
        metrics = run_case(case_lat = case["lat"], case_lon = case["lon"],
                           case_date = case.get("date", "2020-01-01"), area_km2 = case["area_km2"],
                           snn = snn, encoder=encoder, trainer = trainer, buffer = buffer,
                           num_steps = num_steps, is_training = True, kepler_log = kepler_log,
                           sac_loss_log = {"critic": [], "actor": []}, snapshot_log=[],
                           use_real_currents=USE_REAL_CURRENTS, fuel_weight = fuel_weight)

        if (i + 1) % 10 == 0 or i + 1 == n_cases:
            print(f"    SAC training: {i + 1}/{n_cases} cases, reward/step={metrics['reward_per_step']:.4f}")

    return trainer

def evaluate_on_cases(snn, trainer, cases, label, fuel_weight=BASE_FUEL_WEIGHT):
    encoder = DVS_Encoder()
    results = {}
    for case in cases:
        buffer = ReplayBuffer(capacity = 100)
        kepler_log = {"A": [], "B": []}
        metrics = run_case(case_lat = case["lat"], case_lon = case["lon"], case_date=case["start"],
                           area_km2 = case["area_km2"], snn = snn, encoder = encoder, trainer = trainer,
                           buffer = buffer, num_steps = NUM_EVAL_STEPS, is_training = False,
                           kepler_log = kepler_log, sac_loss_log = {"critic": [], "actor": []},
                           snapshot_log=[], use_real_currents = USE_REAL_CURRENTS, fuel_weight = fuel_weight)
        results[case["name"]] = metrics
        print(f"    [{label}] {case['name']}: reward/step={metrics['reward_per_step']:.4f}, "
              f"pred_error_km={metrics['mean_prediction_error_km']}")
    return results



def _mean_reward(results_dict):
    vals = [r["reward_per_step"] for r in results_dict.values() if r.get("reward_per_step") is not None]
    return float(np.mean(vals)) if vals else None

def run_stage_a():
    print("\n/// BUILD + TRAIN + VALIDATE + TEST -- BEST CONFIG ///")
    snn_sensitivity = _load_yaml(f"{SENSITIVITY_DIR}/snn_sensitivity.yaml")
    sac_sensitivity = _load_yaml(f"{SENSITIVITY_DIR}/sac_sensitivity.yaml")
    config = build_best_config(snn_sensitivity, sac_sensitivity)

    print(f"\n Best Config - SNN - {config['snn_kwargs']}")
    print(f"\n Best Config - SAC - {config['sac_kwargs']}, fuel_weight={config['fuel_weight']}")

    for note in config["snn_notes"] + config["sac_notes"]:
        print(f"\n [NOTE]: {note}")

    all_cases = _load_yaml(f"{DATA_DIR}/all_cases.yaml")
    training_cases  = all_cases["train"]

    print("\nTraining Base Config Framework")
    base_snn, _ = train_snn_on_training_set(BASE_SNN_DEFAULTS, training_cases, seed =0)
    base_trainer = train_sac_on_training_set(base_snn, BASE_SAC_DEFAULTS, training_cases, seed =0, fuel_weight=BASE_FUEL_WEIGHT)

    print("\nTraining Best Config Framework")
    best_snn, _ = train_snn_on_training_set(config["snn_kwargs"], training_cases, seed=1)
    best_trainer = train_sac_on_training_set(best_snn, config["sac_kwargs"], training_cases, seed=1, fuel_weight=config["fuel_weight"])

    print("\nValidation Set(Banda & Socotra)")
    base_val = evaluate_on_cases(base_snn, base_trainer, VALIDATION_CASES, "BASE/val", fuel_weight = BASE_FUEL_WEIGHT)
    best_val = evaluate_on_cases(best_snn, best_trainer, VALIDATION_CASES, "BEST/val", fuel_weight = config["fuel_weight"])
    base_val_mean, best_val_mean = _mean_reward(base_val), _mean_reward(best_val)
    print (f" mean reward/step: base = {base_val_mean:.4f}, best = {best_val_mean:.4f}")

    proceed = best_val_mean is not None and base_val_mean is not None and best_val_mean >= base_val_mean
    deployed_label = "best" if proceed else "base"
    deployed_snn = best_snn if proceed else base_snn
    deployed_trainer = best_trainer if proceed else base_trainer
    torch.save(deployed_snn.state_dict(), PRETRAINED_SNN_PATH)
    deployed_trainer.save_checkpoint(PRETRAINED_TRAINER_PATH)
    print(f"[DEPLOY] '{deployed_label}' config -> {PRETRAINED_SNN_PATH}, {PRETRAINED_TRAINER_PATH}")

    if not proceed:
        print("[WARNING] best config performed worse than - one at a time sensitivity analysis")


    print(f"\n Test Set (Java + Somalia)")
    base_test = evaluate_on_cases(base_snn, base_trainer, TEST_CASES, "BASE/test", fuel_weight = BASE_FUEL_WEIGHT)
    best_test = evaluate_on_cases(best_snn, best_trainer, TEST_CASES, "BEST/test", fuel_weight = config["fuel_weight"])
    base_test_mean, best_test_mean = _mean_reward(base_test), _mean_reward(best_test)

    summary = {
        "best_config": {"snn_kwargs": config["snn_kwargs"], "sac_kwargs": config["sac_kwargs"],
                        "fuel_weight": config["fuel_weight"],
                        "unfolded_parameters": config["snn_notes"] + config["sac_notes"]},
        "base_config": {"snn_kwargs": BASE_SNN_DEFAULTS, "sac_kwargs": BASE_SAC_DEFAULTS,
                        "fuel_weight": BASE_FUEL_WEIGHT},
        "snn_best_raw_selection": config["snn_best_raw"], "sac_best_raw_selection": config["sac_best_raw"],
        "stage4_validation": {"base": {"per_case": base_val, "mean_reward_per_step": base_val_mean},
                              "best": {"per_case": best_val, "mean_reward_per_step": best_val_mean},
                              "best_outperforms_base": proceed},
        "stage5_final_test": {"base": {"per_case": base_test, "mean_reward_per_step": base_test_mean},
                              "best": {"per_case": best_test, "mean_reward_per_step": best_test_mean},
                              "best_outperforms_base": (best_test_mean is not None and base_test_mean is not None
                                                        and best_test_mean >= base_test_mean)},
        "limitation_note": ("Best-config values were selected via one-at-a-time (OAT) sensitivity "
                            "sweeps -- interaction effects between parameters are not captured."),
    }

    with open(f"{BEST_CONFIG_DIR}/best_config_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False, default_flow_style=False)
        torch.save(base_snn.state_dict(), f"{BEST_CONFIG_DIR}/base_snn.pt")
        torch.save(best_snn.state_dict(), f"{BEST_CONFIG_DIR}/best_snn.pt")
        print(f"[SAVED] 3 files saved to {BEST_CONFIG_DIR} - best_config_summary.yaml, base_snn.pt, best_snn.pt"
            f"Final test mean reward/step - base: {base_test_mean:.4f}, best: {best_test_mean:.4f}")


# ......................................................................................................................
#                                            SAVING CASE FOLDER DATA + PLOTS
# ......................................................................................................................

def _case_already_done(case_dir):
    return all(os.path.exists(os.path.join(case_dir, f)) for f in EXPECTED_FILES)

def run_case_instrumented(case_lat, case_lon, case_date, area_km2, snn, encoder, trainer, num_steps=None):
    u_field, v_field = get_real_or_fallback_field(case_lat, case_lon, case_date, USE_REAL_CURRENTS)
    custom_params = bloom_params(max_area_spread_km2 = area_km2)
    lifetime_days = custom_params.bloom_lifetime_days
    if num_steps is None:
        num_steps = int(lifetime_days * 24) + 6
    env = TwoSatelliteBloomEnvironment(u_field, v_field, params=custom_params, num_satellites=2,
                                        origin_ref_lat=case_lat, origin_ref_lon=case_lon,
                                        origin_date=case_date, refetch_on_recenter=False)
    imap = InterestMap(env.X, env.Y)
    frame_history = {"A": [], "B": []}

    snapshot_days  = np.linspace(0, lifetime_days, NUM_SNAPSHOTS)
    snapshots_taken = [False] * NUM_SNAPSHOTS
    snapshots = []
    SNAPSHOT_EPS = 1e-3

    rewards, pred_positions, true_positions = [], [], []
    critic_q_trace, actor_entropy_trace = [], []
    spike_counts = {"lif1": [], "lif2": [], "lif3": []}
    visibility_trace = {"A": [], "B": []}
    deltav_trace = {"A": [], "B": []}

    spike_accum = {"lif1": 0.0, "lif2": 0.0, "lif3": 0.0}
    hooks = []

    def _make_hook(name):
        def hook(module, inp, output):
            spike_accum[name] += output.sum().item()
        return hook

    for layer in {"lif1", "lif2", "lif3"}:
        if hasattr(snn, layer):
            hooks.append(getattr(snn, layer).register_forward_hook(_make_hook(layer)))

    for step in range(num_steps):
        env._step(dt_step_day = 1.0 / 24.0)
        visibility = env.visible_satellites()
        bloom_center = env.state.bloom_center_km.copy()
        true_state = np.array([bloom_center[0], bloom_center[1], env.state.bloom_area_km2])
        true_positions.append(bloom_center)

        spike_accum["lif1"] = spike_accum["lif2"] = spike_accum["lif3"] = 0.0

        for name in ("A", "B"):
            elements = env.satellites[env._resolve_index(name)]
            obs = build_observation(elements, imap,case_lat, case_lon)
            visibility_trace[name].append(bool(visibility[name]))

            pred_xy = None
            if visibility[name]:
                frame = env.get_satellite_view(name, visibility)
                ref_frame = pick_reference_frame(frame_history[name], env.state.current_bloom_day)
                pred_norm = snn_predict(snn, encoder, frame, env, prev_frame=ref_frame)
                frame_history[name].append((env.state.current_bloom_day, frame))
                if pred_norm is not None:
                    pred_xy = np.array([pred_norm[0] * GRID_LIMIT_KM, pred_norm[1] * GRID_LIMIT_KM])
                    imap.deposit(pred_xy[0], pred_xy[1], amount=1.0)

            obs_tensor = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)

            with torch.no_grad():
                mean, log_std = trainer.actor.forward(obs_tensor)
                std = log_std.exp()
                entropy = float((0.5 * torch.log(2 * np.pi * np.e * std.pow(2))).sum().item())
                action, _ = trainer.actor.sample(obs_tensor)

            actor_entropy_trace.append(entropy)
            deltav_trace[name].append(float(np.linalg.norm(action.squeeze(0).numpy())))
            if pred_xy is not None:
                pred_positions.append(pred_xy)
            env.apply_satellite_action(name, action.squeeze(0).numpy())

        imap.evaporate()
        is_visible_any = any(visibility[n] for n in ("A", "B"))
        reward = compute_reward(predicted_state, true_state, action.squeeze(0).numpy(), is_visible_any, state_scale= STATE_SCALE)
        rewards.append(reward)

        with torch.no_grad():
            obs_a = build_observation(env.satellites[0], imap, case_lat, case_lon)
            obs_b = build_observation(env.satellites[1], imap, case_lat, case_lon)
            joint_obs = torch.tensor(np.concatenate([obs_a, obs_b]), dtype=torch.float32).unsqueeze(0)
            joint_action = torch.tensor(action.squeeze(0).numpy().tolist() * 2, dtype=torch.float32).unsqueeze(0)
            q1, q2 = trainer.critic(joint_obs, joint_action)
            critic_q_trace.append(float(torch.min(q1, q2).item()))

        for i, sd in enumerate(snapshot_days):
            if not snapshots_taken[i] and env.state.current_bloom_day >= sd - SNAPSHOT_EPS:
                snapshots.append({
                    "day": env.state.current_bloom_day, "fraction": i / (NUM_SNAPSHOTS - 1),
                    "area_km2": env.state.bloom_area_km2, "bloom_field": env._rendering().copy(),
                    "bloom_density_raw": env.C.copy(), "interest_grid": imap.grid.copy(),
                })

                snapshots_taken[i] = True

        spike_counts["lif1"].append(spike_accum["lif1"])
        spike_counts["lif2"].append(spike_accum["lif2"])
        spike_counts["lif3"].append(spike_accum["lif3"])

    for h in hooks:
        h.remove()

    while len(snapshots) < NUM_SNAPSHOTS and snapshots:
        snapshots.append(snapshots[-1])

    return {
    "rewards": rewards, "pred_positions": pred_positions, "true_positions": true_positions,
    "critic_q_trace": critic_q_trace, "actor_entropy_trace": actor_entropy_trace,
    "spike_counts": spike_counts, "snapshots": snapshots,
    "visibility_trace": visibility_trace, "deltav_trace": deltav_trace,
}

def save_filmstrip(snapshots, key, title, out_path, cmap = "turbo", show_area = False, shared_scale = False, colorbar_label = None):
    n = len(snapshots)
    if n == 0:
        return
    ncols, nrows = 5,2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()
    extent = [-GRID_LIMIT_KM, GRID_LIMIT_KM, -GRID_LIMIT_KM, GRID_LIMIT_KM]

    vmin, vmax = None, None

    if shared_scale:
        vmin = 0.0
        vmax = max(snapshots[i][key].max() for i in range(n))

    im = None
    for i, ax in enumerate(axes):
        if i < n:
            intended_pct = i / max(n - 1, 1) * 100
            im = ax.imshow(snapshots[i][key], extent=extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            label = f"{intended_pct:.0f}% (day {snapshots[i]['day']:.1f})"
            if show_area:
                label += f"\narea={snapshots[i].get('area_km2', float('nan')):.0f} km$^2$"
            ax.set_title(label, fontsize=9)
        else:
            ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.92, 1] if shared_scale else [0, 0, 1, 1])

    if shared_scale and im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax = cbar_ax)
        cbar.set_label(colorbar_label or "value (relative units, shared scale)")

    fig.savefig(out_path, dpi = 120)
    plt.close(fig)

def save_spike_activity(spike_counts, out_path):
    fig, ax = plt.subplots(figsize = (8,4))
    for name, counts  in spike_counts.items():
        ax.plot(counts, label = name, alpha = 0.8)

    ax.set_xlabel("step")
    ax.set_ylabel("total spikes (summed over layer)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title("SNN spiking activity per layer over the episode (symlog scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_activity_context(spike_counts, visibility_trace, deltav_trace, out_path):
    n_steps = len(spike_counts["lif1"])
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_spike, ax_vis, ax_dv = axes

    for name, counts in spike_counts.items():
        ax_spike.plot(steps, counts, label = name, alpha = 0.8)
    ax_spike.set_ylabel("total spikes")
    ax_spike.set_yscale("symlog", linthresh=0.1)
    ax_spike.set_title("SNN spiking activity (symlog scale)")
    ax_spike.legend(fontsize = 8)
    ax_spike.grid(True, alpha = 0.3)

    for name, color, y_offset in (("A", "tab:blue", 1.0), ("B", "tab:orange", 0.0)):
        vis = np.array(visibility_trace[name], dtype = float)
        ax_vis.fill_between(steps, y_offset, y_offset + vis * 0.9, step = "pre", color = color, alpha = 0.6, label = f"Sat {name} visible" )
    ax_vis.set_ylabel("visibility")
    ax_vis.set_yticks([0.45, 1.45]);
    ax_vis.set_yticklabels(["Sat B", "Sat A"])
    ax_vis.set_title("Satellite visibility windows (bloom within swath)")
    ax_vis.grid(True, alpha=0.3)

    ax_dv.plot(steps, deltav_trace["A"], label="Sat A |$\\Delta V$|", alpha=0.8, color="tab:blue")
    ax_dv.plot(steps, deltav_trace["B"], label="Sat B |$\\Delta V$|", alpha=0.8, color="tab:orange")
    ax_dv.set_xlabel("step");
    ax_dv.set_ylabel("|$\\Delta V$| (km/s)")
    ax_dv.set_title("Maneuver magnitude per satellite")
    ax_dv.legend(fontsize=8);
    ax_dv.grid(True, alpha=0.3)

    fig.suptitle("SNN activity, satellite visibility, and maneuvers over the episode")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def save_reward_trace(rewards, out_path):
    fig, axes = plt.subplots(1, 2, figsize = (12,4))
    axes[0].plot(rewards, alpha = 0.8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("reward")
    axes[1].plot(np.cumsum(rewards))
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("Cumulative reward")
    axes[1].set_title("Cumulative reward")
    axes[1].grid(True, alpha = 0.3)
    fig.suptitle("SAC: reward over the episode")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)

def save_tracking_overlay(pred_positions, true_positions, out_path):
    true_arr = np.array(true_positions)
    x_range = true_arr[:,0].max() - true_arr[:,0].min()
    y_range = true_arr[:, 1].max() - true_arr[:, 1].min()
    aspect_ratio = np.clip((y_range +1e-6)/ (x_range + 1e-6), 0.2, 2.0)
    fig_w = 8.0
    fig_h = np.clip(fig_w * aspect_ratio, 3.5, 8.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.plot(true_arr[:, 0], true_arr[:, 1], "-", color="black", label="true bloom center", linewidth=2)
    if pred_positions:
        pred_arr = np.array(pred_positions)
        ax.scatter(pred_arr[:, 0], pred_arr[:, 1], color="red", s=20, alpha=0.7, label = f"SNN predictions (n={len(pred_positions)})", zorder = 5)
    ax.set_xlabel("km");
    ax.set_ylabel("km")
    ax.legend(loc="best", fontsize=8);
    ax.grid(True, alpha=0.3)
    fig.suptitle("SAC: predicted vs. true bloom-center trajectory")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)

def save_critic_q_trace(critic_q_trace, out_path):
    fig,ax = plt.subplots(figsize = (8,4))
    ax.plot(critic_q_trace, color = "purple")
    ax.set_xlabel("step")
    ax.set_ylabel("Critic Q-estimate (min of 2 critics)")
    ax.set_title("SAC: critic's value estimate over the episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def save_actor_entropy(actor_entropy_trace, out_path):
    fig,ax = plt.subplots(figsize = (8,4))
    ax.plot(actor_entropy_trace, color = "darkorange")
    ax.set_xlabel("step (per-satellite action)")
    ax.set_ylabel("action distribution entropy (nats)")
    ax.set_title("SAC: actor entropy over the episode")
    fig.tight_layout()
    fig.savefig(out_path, dpi = 120)
    plt.close(fig)


def process_case(case_id, case_lat, case_lon, case_date, area_km2, out_dir, snn, encoder, trainer):
    os.makedirs(out_dir, exist_ok=True)
    if _case_already_done(out_dir):
        return
    result = run_case_instrumented(case_lat, case_lon, case_date, area_km2, snn, encoder, trainer)
    save_filmstrip(result["snapshots"], "bloom_density_raw", f"{case_id}: bloom density over lifecycle",
                    os.path.join(out_dir, "bloom_filmstrip.png"), cmap="turbo", show_area=True, shared_scale=True,
                    colorbar_label="bloom concentration density (relative units, shared scale)")


    save_filmstrip(result["snapshots"], "interest_grid", f"{case_id}: interest map over lifecycle",
                    os.path.join(out_dir, "interest_map_filmstrip.png"), cmap = "hot", shared_scale = True,
                    colorbar_label="interest map intensity (stigmergy deposits, shared scale)")


    save_spike_activity(result["spike_counts"], os.path.join(out_dir, "snn_spike_activity.png"))

    save_activity_context(result["spike_counts"], result["visibility_trace"], result["deltav_trace"],
                           os.path.join(out_dir, "snn_activity_context.png"))

    save_reward_trace(result["rewards"], os.path.join(out_dir, "sac_reward_trace.png"))

    save_tracking_overlay(result["pred_positions"], result["true_positions"], os.path.join(out_dir, "sac_tracking_overlay.png"))

    save_critic_q_trace(result["critic_q_trace"], os.path.join(out_dir, "sac_critic_q_trace.png"))

    save_actor_entropy(result["actor_entropy_trace"], os.path.join(out_dir, "sac_actor_entropy.png"))

    arr = np.array(result["spike_counts"]["lif1"])
    print(f" {case_id}: lif1 nonzero_steps = {int((arr>0).sum())}/{len(arr)}, "
          f"predictions={len(result['pred_positions'])}")


def run_stage_b():
    print("///DIAGNOSTIC CASE FOLDERS///")
    with open(f"{DATA_DIR}/all_cases.yaml") as f:
        all_cases = yaml.safe_load(f)

        # snn
        torch.manual_seed(0)
        snn = SNN()
        if os.path.exists(PRETRAINED_SNN_PATH):
            snn.load_state_dict(torch.load(PRETRAINED_SNN_PATH))
        else:
            print(f"[WARNING] {PRETRAINED_SNN_PATH} not found")

        # sac
        encoder = DVS_Encoder()
        trainer = SAC_Trainer(num_agents=2)
        if  os.path.exists(PRETRAINED_TRAINER_PATH):
            trainer.load_checkpoint(PRETRAINED_TRAINER_PATH)
        else:
            print("[WARNING] {PRETRAINED_SNN_PATH} not found")
        for split_name, split_key, date_key in [("train", "train", "date"), ("test", "test", "start_date"),
                                              ("validation", "validation", "start_date")]:
            print(f"\n{split_name} cases ({len(all_cases[split_key])}):")

            for name, case in all_cases[split_key].items():
                safe_name = name.replace(" ", "_").lower() if split_key != "train" else name
                out_dir = os.path.join(CASES_DIR, split_name, safe_name)
                process_case(safe_name, case["lat"], case["lon"], case.get(date_key, "2020-01-01"),
                             case["area_km2"], out_dir, snn, encoder, trainer)

    print("\nAll cases processed.")




if __name__ == "__main__":
    run_stage_a()
    run_stage_b()