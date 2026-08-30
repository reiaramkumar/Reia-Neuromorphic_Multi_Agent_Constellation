import sys
sys.path.insert(0, ".")

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, functional, surrogate
from pipeline_classes import *
from train import run_case, get_real_or_fallback_field, _sat_name

OUTPUT_DIR = "pipeline_outputs/sensitivity"
os.makedirs(OUTPUT_DIR, exist_ok = True)

PLOTS_DIR = f"{OUTPUT_DIR}/plots"
NUM_STEPS_PER_EVAL = 1080
NUM_SNN_TRAIN_SAMPLES = 23
USE_REAL_CURRENTS = True


# for the sensitivity analysis we will be varying 8 params each in SAC n SNN


# ......................................................................................................................
#                                                   SNN VARIANT CLASSES (3 & 2 layers)
# ......................................................................................................................

_SURROGATE_FNS = {"atan": surrogate.ATan,
                  "sigmoid": surrogate.Sigmoid,
                  "piecewise_linear": surrogate.PiecewiseLeakyReLU}

_OPTIMIZERS = {"adam": torch.optim.Adam,
               "sgd": torch.optim.SGD,
               "rmsprop": torch.optim.RMSprop}

class _Variant_SNN_3(nn.Module):
    def __init__(self, v_threshold = 0.3, tau = 1.5, weight_scale = 2.0, surrogate_type = "atan", dropout_rate = 0.0):
        super().__init__()
        surr_cls = _SURROGATE_FNS.get(surrogate_type, surrogate.ATan)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)

        with torch.no_grad():
            self.conv1.weight *= weight_scale
            self.conv2.weight *= weight_scale
            self.conv3.weight *= weight_scale


        self.lif1 = neuron.LIFNode(tau = tau, v_threshold = v_threshold, surrogate_function = surr_cls())
        self.lif2 = neuron.LIFNode(tau = tau, v_threshold = v_threshold,surrogate_function = surr_cls())
        self.lif3 = neuron.LIFNode(tau = tau, v_threshold = v_threshold,surrogate_function = surr_cls())

        # the o/p must be flattened out to a single array
        flattened_size = 64 * 25 * 25
        self.readout = nn.Linear(flattened_size, 3) # -> o/p (center_x_km, center_y_km, area_km2)
        self.dropout = nn.Dropout(p = dropout_rate) if dropout_rate > 0 else nn.Identity()
    def forward(self, event_sequence: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self) # clear the prev sequence
        T = event_sequence.shape[0]
        readout_accumulated = torch.zeros(3)

        for t in range(T):
            x = event_sequence[t].unsqueeze(0)
            x = self.lif1(self.conv1(x))
            x = self.lif2(self.conv2(x))
            x = self.lif3(self.conv3(x))
            x = x.flatten(start_dim = 1)
            x = self.dropout(x)
            readout_accumulated = readout_accumulated + self.readout(x).squeeze(0)
        return readout_accumulated / T # avg readout over all time steps

# the 2 layer variant
class _Variant_SNN_2(nn.Module):
    def __init__(self, v_threshold = 0.3, tau = 1.5, weight_scale = 2.0, surrogate_type = "atan", dropout_rate = 0.0):
        super().__init__()
        surr_cls = _SURROGATE_FNS.get(surrogate_type, surrogate.ATan)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2)

        with torch.no_grad():
            self.conv1.weight *= weight_scale
            self.conv2.weight *= weight_scale

        self.lif1 = neuron.LIFNode(tau = tau, v_threshold = v_threshold, surrogate_function = surr_cls())
        self.lif2 = neuron.LIFNode(tau = tau, v_threshold = v_threshold,surrogate_function = surr_cls())

        # the o/p must be flattened out to a single array
        flattened_size = 32 * 50 * 50
        self.readout = nn.Linear(flattened_size, 3) # -> o/p (center_x_km, center_y_km, area_km2)

    def forward(self, event_sequence: torch.Tensor) -> torch.Tensor:
        functional.reset_net(self) # clear the prev sequence
        T = event_sequence.shape[0]
        readout_accumulated = torch.zeros(3)

        for t in range(T):
            x = event_sequence[t].unsqueeze(0)
            x = self.lif1(self.conv1(x))
            x = self.lif2(self.conv2(x))
            x = x.flatten(start_dim = 1)
            readout_accumulated = readout_accumulated + self.readout(x).squeeze(0)
        return readout_accumulated / T # avg readout over all time steps


# ......................................................................................................................
#                                                   SAC VARIANT CLASSES
# ......................................................................................................................

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}

class _Variant_SAC_Actor(nn.Module):
    def __init__ (self, obs_dim: int= OBS_DIM, action_dim: int = ACTION_DIM,  hidden_dim: int = 128,
                  activation = "relu"):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), act_cls(),
            nn.Linear(hidden_dim, hidden_dim), act_cls(),
        )
        self.mean_head = nn.Linear (hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor):
        x = self.net(obs)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs:torch.Tensor):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * MAX_ACTION_KMS

        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(1-squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

class _Variant_SAC_Critic(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128,
                 activation = "relu"):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
        joint_dim = num_agents * (obs_dim + action_dim)

        self.q1 = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), act_cls(),
            nn.Linear(hidden_dim, hidden_dim), act_cls(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
                    nn.Linear(joint_dim, hidden_dim), act_cls(),
                    nn.Linear(hidden_dim, hidden_dim), act_cls(),
                    nn.Linear(hidden_dim, 1),
                )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor):
        x = torch.cat([joint_obs, joint_actions], dim=1)
        return self.q1(x), self.q2(x)


class _Variant_SAC_Trainer:
    def __init__(self, num_agents: int, lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, activation = "relu", optimizer_name = "adam"):

        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = _Variant_SAC_Actor(activation = activation)
        self.critic = _Variant_SAC_Critic(num_agents=num_agents, activation = activation)
        import copy
        self.critic_target = copy.deepcopy(self.critic)

        opt_cls = _OPTIMIZERS.get(optimizer_name, torch.optim.Adam)
        self.actor_optimizer = opt_cls(self.actor.parameters(), lr=lr)
        self.critic_optimizer = opt_cls(self.critic.parameters(), lr = lr)

    def act(self, per_agent_obs):
        with torch.no_grad():
            actions = [self.actor.sample(o.unsqueeze(0))[0].squeeze(0) for o in per_agent_obs]
            return torch.cat(actions).numpy()

    def _jointsample(self, per_agent_obs: list):
        actions, log_probs = [], []
        for agent_obs in per_agent_obs:
            a, lp = self.actor.sample(agent_obs)
            actions.append(a)
            log_probs.append(lp)
        return torch.cat(actions, dim=1), sum(log_probs)

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        joint_action_taken = batch["joint_actions_taken"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        joint_obs = torch.cat(obs, dim=1)
        joint_next_obs = torch.cat(next_obs, dim = 1)

        # CRITIC UPDATE
        with torch.no_grad():
            next_joint_actions, next_log_prob = self._jointsample(next_obs)
            target_q1, target_q2 = self.critic_target(joint_next_obs, next_joint_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            backup = rewards + self.gamma *  ( 1 - dones) * target_q
        current_q1, current_q2 = self.critic(joint_obs, joint_action_taken)
        critic_loss = F.mse_loss(current_q1, backup) + F.mse_loss(current_q2, backup)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ACTOR UPDATE
        new_joint_actions, new_log_prob = self._jointsample(obs)
        q1_new, q2_new = self.critic(joint_obs, new_joint_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * new_log_prob - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + ( 1 - self.tau) * target_param.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


# ......................................................................................................................
#                                                   SNN SENSITIVITY RUN
# ......................................................................................................................

def _train_snn_from_scratch(make_snn_fn, optimizer_name = "adam", lr = 3e-5, num_samples = NUM_SNN_TRAIN_SAMPLES, seed = 0):
    torch.manual_seed(seed)
    snn = make_snn_fn()
    opt_cls = _OPTIMIZERS.get(optimizer_name, torch.optim.Adam)
    optimizer = opt_cls(snn.parameters(), lr = lr)
    encoder = DVS_Encoder(rng = np.random.default_rng(seed))
    rng = np.random.default_rng(seed)

    t_start = time.perf_counter()
    losses = []


    for i in range(num_samples):
        case_lat, case_lon = rng.uniform(-10, 10), rng.uniform(44, 58)
        u, v = get_real_or_fallback_field(case_lat, case_lon, "2020-01-01", use_real_currents=False)
        env = HycomBloomEnvironment(u, v, params=bloom_params(max_area_spread_km2=30000),
                                    origin_ref_lat=case_lat, origin_ref_lon=case_lon,
                                    origin_date="2020-01-01", refetch_on_recenter=False)

        env._step(dt_step_day = 1.0)
        events = encoder.encode(env, dt_days = 1.0)
        if events.shape[0] == 0:
            continue
        events_tensor = torch.tensor(events, dtype = torch.float32)
        target =  torch.tensor([env.state.bloom_center_km[0] / GRID_LIMIT_KM,
                                env.state.bloom_center_km[1] / GRID_LIMIT_KM,
                                env.state.bloom_area_km2 / 45000.0], dtype = torch.float32)
        functional.reset_net(snn)
        optimizer.zero_grad()
        pred = snn(events_tensor)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(snn.parameters(), 0.5)
        optimizer.step()
        losses.append(loss.item())
        print(f"    sample {i + 1}/{num_samples}: lat={case_lat:.1f}, lon={case_lon:.1f}, loss={loss.item():.4f}")
    train_time_sec = time.perf_counter() - t_start

    final_loss = float(np.mean(losses[-5:])) if losses else None
    instability = float(np.std(losses[-10:])) if len(losses) >= 10 else None
    return snn, final_loss, instability, train_time_sec

def _eval_snn_on_case(snn, case, encoder = None):
    encoder = encoder or DVS_Encoder()
    trainer = SAC_Trainer(num_agents = 2)
    buffer = ReplayBuffer(capacity = 2000)
    kepler_log = {"A": [], "B": []}
    t_start = time.perf_counter()
    metrics = run_case(case_lat = case["lat"], case_lon = case["lon"], case_date = case["start"],
                       area_km2 = case["area_km2"], snn = snn, encoder = encoder, trainer = trainer,
                       buffer = buffer, num_steps=NUM_STEPS_PER_EVAL, is_training=False,
                       kepler_log=kepler_log, sac_loss_log={"critic": [], "actor": []},
                       snapshot_log=[], use_real_currents=USE_REAL_CURRENTS)
    eval_time_sec = time.perf_counter() - t_start
    return metrics, eval_time_sec

def _run_snn_sweep_point(parameter, value, make_snn_fn, optimizer_name = "adam", lr = 3e-5):
    snn, final_loss, instability, train_time = _train_snn_from_scratch(make_snn_fn, optimizer_name = optimizer_name, lr = lr)
    per_case = {}
    reward_sum = 0.0
    for case in VALIDATION_CASES:
        case_key = case["name"].replace(" ", "_" ).lower()
        metrics, eval_time = _eval_snn_on_case(snn, case)
        per_case[case_key] = {
            "reward_per_step": metrics["reward_per_step"],
            "mean_prediction_error_km": metrics["mean_prediction_error_km"],
            "num_predictions_made": metrics["num_predictions_made"],
            "final_loss": final_loss, "instability": instability,
            "computation_time_sec": round(train_time + eval_time, 3)
        }
        reward_sum += metrics["reward_per_step"]
    print(f"[SNN] {parameter}={value}: mean reward/step={reward_sum/len(VALIDATION_CASES):.4f}")
    return per_case

def sweep_snn_v_threshold():
    return {v: _run_snn_sweep_point("v_threshold", v, lambda v = v: _Variant_SNN_3(v_threshold = v))
            for v in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]}

def sweep_snn_weight_scale():
    return {ws: _run_snn_sweep_point("weight_scale", ws, lambda ws = ws: _Variant_SNN_3(weight_scale=ws))
            for ws in [1.0, 2.0, 3.0, 4.0, 5.0]}


def sweep_snn_tau():
    return {t: _run_snn_sweep_point("tau", t, lambda t = t: _Variant_SNN_3(tau=t))
            for t in [1.1, 1.5, 2.0, 2.5, 3.0]}


def sweep_snn_learning_rate():
    return {lr: _run_snn_sweep_point("learning_rate", lr, lambda: _Variant_SNN_3(), lr=lr)
            for lr in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]}


def sweep_snn_surrogate_type():
    return {s: _run_snn_sweep_point("surrogate_function", s, lambda s =s: _Variant_SNN_3(surrogate_type=s))
            for s in ["atan", "sigmoid", "piecewise_linear"]}


def sweep_snn_network_depth():
    variants = {"3-layer (baseline)": lambda: _Variant_SNN_3(),
                "2-layer (shallow)": lambda: _Variant_SNN_2(),}
    return {name: _run_snn_sweep_point("network_depth", name, fn) for name, fn in variants.items()}


def sweep_snn_optimizer():
    return {opt: _run_snn_sweep_point("optimizer", opt, lambda: _Variant_SNN_3(), optimizer_name=opt)
            for opt in ["adam", "sgd", "rmsprop"]}


def sweep_snn_dropout():
    return {d: _run_snn_sweep_point("dropout_rate", d, lambda d = d: _Variant_SNN_3(dropout_rate=d))
            for d in [0.0, 0.1, 0.2, 0.3, 0.5]}

# ......................................................................................................................
#                                                       SAC SENSITIVITY SWEEP
# ......................................................................................................................

def _run_sac_sweep_point(parameter, value, make_trainer_fn,  fuel_weight = 0.5, num_satellites = 2):
    per_case = {}
    reward_sum = 0.0
    for case in VALIDATION_CASES:
        torch.manual_seed(0)
        trainer = make_trainer_fn()
        encoder = DVS_Encoder()
        buffer = ReplayBuffer(capacity = 2000)
        kepler_log = {_sat_name(i): [] for i in range(num_satellites)}
        t_start = time.perf_counter()
        metrics = run_case(case_lat=case["lat"], case_lon=case["lon"], case_date=case["start"],
                            area_km2=case["area_km2"], snn=SNN(), encoder=encoder,
                            trainer=trainer, buffer=buffer, num_steps=NUM_STEPS_PER_EVAL,
                            is_training=True, kepler_log=kepler_log,
                            sac_loss_log={"critic": [], "actor": []}, snapshot_log=[],
                            use_real_currents=USE_REAL_CURRENTS, fuel_weight=fuel_weight,
                            num_satellites=num_satellites)
        comp_time = time.perf_counter() - t_start
        case_key = case["name"].replace(" ", "_" ).lower()
        per_case[case_key] = {
            "reward_per_step": metrics["reward_per_step"],
            "mean_prediction_error_km": metrics["mean_prediction_error_km"],
            "num_predictions_made": metrics["num_predictions_made"],
            "computation_time_sec": round(comp_time, 3)
        }
        reward_sum += metrics["reward_per_step"]
    print(f"[SAC] {parameter}={value}: mean reward/step={reward_sum / len(VALIDATION_CASES):.4f}")
    return per_case


def sweep_sac_learning_rate():
    return {lr: _run_sac_sweep_point("learning_rate", lr, lambda lr = lr: SAC_Trainer(num_agents = 2, lr = lr))
            for lr in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]}


def sweep_sac_target_tau():
    return {t: _run_sac_sweep_point("target_tau", t, lambda t = t: SAC_Trainer(num_agents = 2, tau = t))
            for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]}


def sweep_sac_alpha():
    return {a: _run_sac_sweep_point("alpha", a, lambda a = a: SAC_Trainer(num_agents = 2, alpha = a))
            for a in [0.05, 0.1, 0.2, 0.3, 0.4]}


def sweep_sac_gamma():
    return {g: _run_sac_sweep_point("gamma", g, lambda g=g: SAC_Trainer(num_agents = 2, gamma=g))
            for g in [0.9, 0.95, 0.99, 0.995, 0.999]}


def sweep_sac_fuel_weight():
    return {fw: _run_sac_sweep_point("fuel_weight", fw, lambda: SAC_Trainer(num_agents = 2), fuel_weight = fw)
            for fw in [0.1, 0.3, 0.5, 0.7, 1.0]}


def sweep_sac_activation():
    return {act: _run_sac_sweep_point("activation_function", act,
                                       lambda act=act: _Variant_SAC_Trainer(num_agents = 2, activation=act))
            for act in ["relu", "tanh", "leaky_relu"]}


def sweep_sac_num_agents():
    return {n: _run_sac_sweep_point("num_agents", n, lambda n=n: SAC_Trainer(num_agents=n),
                                     num_satellites=n) for n in [1, 2, 3, 4]}


def sweep_sac_optimizer():
    return {opt: _run_sac_sweep_point("optimizer", opt,
                                       lambda opt=opt: _Variant_SAC_Trainer(num_agents = 2, optimizer_name = opt))
            for opt in ["adam", "sgd", "rmsprop"]}

# ......................................................................................................................
#                                                   PLOTTING
# ......................................................................................................................

CASE_COLORS = {"2019_banda": "#1f77b4", "2021_socotra": "#d62728"}
CASE_LABELS = {"2019_banda": "2019 Banda", "2021_socotra": "2021 Socotra"}

PARAM_LABELS = {
    "v_threshold": r"$v_{\mathrm{th}}$", "weight_scale": "Conv weight scale", "tau": r"$\tau$",
    "learning_rate": "Learning rate", "surrogate_function": "Surrogate function",
    "network_depth": "Network depth", "optimizer": "Optimizer", "dropout_rate": "Dropout rate",
    "target_tau": r"Target smoothing $\tau$", "alpha": r"Entropy coefficient $\alpha$",
    "gamma": r"Discount factor $\gamma$", "fuel_weight": "Fuel cost weight",
    "activation_function": "Activation function", "num_agents": "Number of agents $N$",
}

METRIC_LABELS = {
    "reward_per_step": "Reward / step", "mean_prediction_error_km": "Mean prediction error (km)",
    "final_loss": "Final training loss", "instability": "Training instability",
    "num_predictions_made": "Valid predictions (count)", "computation_time_sec": "Computation time (s)",
}

TRAINING_ONLY_METRICS = {"final_loss", "instability"}



def _is_numeric(values):
    try:
        [float(v) for v in values]
        return True
    except (TypeError, ValueError):
        return False

def plot_parameter(tier, parameter, param_data, output_dir):
    values = list(param_data.keys())
    numeric = _is_numeric(values)
    if numeric:
        values = sorted(values, key = lambda v: float(v))

    metrics_present = set()
    for v in values:
        for case_key, case_metrics in param_data[v].items():
            for m, val in case_metrics.items():
                if val is not None:
                    metrics_present.add(m)
    metrics_present = [m for m in METRIC_LABELS if m in metrics_present]
    if not metrics_present:
        return
    n_metrics = len(metrics_present)
    if tier == "SNN":
        ncols = -(-n_metrics // 2)
        fig, axes = plt.subplots(2, ncols, figsize = (5 * ncols, 8))
        axes = axes.flatten()
        for extra_ax in axes[n_metrics:]:
            extra_ax.set_visible(False)

    else:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

    label = PARAM_LABELS.get(parameter, parameter)
    for ax, metric in zip(axes, metrics_present):
        is_training_only = metric in TRAINING_ONLY_METRICS

        if is_training_only:
            xs, ys = [], []
            for v in values:
                val = None
                for case_key in CASE_COLORS:
                    val = param_data[v].get(case_key, {}).get(metric)
                    if val is not None:
                        break
                if val is None:
                    continue
                xs.append(float(v) if numeric else str(v))
                ys.append(val)
            if xs:
                if numeric:
                    ax.plot(xs, ys, marker = "o", linewidth = 1.8, color = "#2ca02c", label = "training run")
                else:
                    ax.bar(range(len(xs)), ys, label = "training run", width = 0.5, color = "#2ca02c")
                    ax.set_xticks(range(len(xs)))
                    ax.set_xticklabels(xs, rotation = 30, ha = "right")
                    ax.legend(fontsize = 7)
        else:
            for case_key in CASE_COLORS:
                xs, ys = [], []
                for v in values:
                    y = param_data[v].get(case_key, {}).get(metric)
                    if  y is None:
                        continue
                    xs.append(float(v) if numeric else str(v))
                    ys.append(y)
                if not xs:
                    continue
                color = CASE_COLORS[case_key]
                case_label = CASE_LABELS.get(case_key, case_key)
                if numeric:
                    ax.plot(xs, ys, marker = "o", linewidth = 1.8, color = color, label = case_label)

                else:
                    offset = -0.15 if case_key == "2019_banda" else 0.15
                    positions = [i + offset for i in range(len(xs))]
                    ax.bar(positions, ys, width = 0.3, color = color, label = case_label)
                    ax.set_xticks(range(len(xs)))
                    ax.set_xticklabels(xs, rotation = 20, ha = "right")
            ax.legend(fontsize = 8)

        ax.set_xlabel(label)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{tier}: sensitivity to {label} (validation cases)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{tier}_{parameter}.png"), dpi=150)
    plt.close(fig)


def plot_all():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for tier, fname in [("SNN", "snn_sensitivity.yaml"), ("SAC", "sac_sensitivity.yaml")]:
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            print(f'[WARNING] {path} does not exist so SKIPPING {tier} plots')
            continue
        with open(path) as f:
            data = yaml.safe_load(f)
        for parameter, param_data in data.items():
            plot_parameter(tier, parameter, param_data, PLOTS_DIR)
    print(f"plots saved to {PLOTS_DIR}")

# ......................................................................................................................
#                                                   MAIN
# ......................................................................................................................

if __name__ == "__main__":
    np.random.seed(0)

    SNN_SWEEPS = [
        ("v_threshold", sweep_snn_v_threshold), ("weight_scale", sweep_snn_weight_scale),
        ("tau", sweep_snn_tau), ("learning_rate", sweep_snn_learning_rate),
        ("surrogate_function", sweep_snn_surrogate_type), ("network_depth", sweep_snn_network_depth),
        ("optimizer", sweep_snn_optimizer), ("dropout_rate", sweep_snn_dropout),
    ]
    SAC_SWEEPS = [
        ("learning_rate", sweep_sac_learning_rate), ("target_tau", sweep_sac_target_tau),
        ("alpha", sweep_sac_alpha), ("gamma", sweep_sac_gamma), ("fuel_weight", sweep_sac_fuel_weight),
        ("activation_function", sweep_sac_activation), ("num_agents", sweep_sac_num_agents),
        ("optimizer", sweep_sac_optimizer),
    ]

    def _load_or_empty(path):
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            if data:
                print(f"Resuming {path} ({len(data)} parameter(s) already done)")
            return data
        return {}

    def _run_sweeps_with_checkpoint(sweeps, out_path, tier_label):
        results = _load_or_empty(out_path)
        for param_name, sweep_fn in sweeps:
            if param_name in results:
                continue
            try:
                results[param_name] = sweep_fn()
            except Exception as e:
                print(f"[ERROR] {tier_label} sweep '{param_name}' failed: {e}")
                with open(out_path, "w") as f:
                    yaml.safe_dump(results, f, sort_keys = False, default_flow_style = False)
                raise
            with open(out_path, "w") as f:
                yaml.safe_dump(results, f, sort_keys = False, default_flow_style = False)
        return results

    print("\n///SNN SENSITIVITY///")
    snn_results = _run_sweeps_with_checkpoint(SNN_SWEEPS, f"{OUTPUT_DIR}/snn_sensitivity.yaml", "SNN")
    print("\n///SAC SENSITIVITY///")
    sac_results = _run_sweeps_with_checkpoint(SAC_SWEEPS, f"{OUTPUT_DIR}/sac_sensitivity.yaml", "SAC")
    print("\n///PLOTTING///")
    plot_all()
    print("DONEEEE!!!!")
