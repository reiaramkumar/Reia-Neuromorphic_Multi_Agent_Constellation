import numpy as np
import xarray as xr
from numpy import flatiter
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates, shift as ndi_shift
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from collections import deque
import copy
from spikingjelly.activation_based import neuron, functional, surrogate

PRETRAINED_SNN_PATH = "pretrained_snn.pt"
PRETRAINED_TRAINER_PATH = "pretrained_trainer.pt"

# ......................................................................................................................
#                                           BLOOM DYNAMICS MODEL
#.......................................................................................................................

# constant params
GRID_SIZE_KM = 400.0
STATE_SCALE = np.array([GRID_SIZE_KM, GRID_SIZE_KM, 45000]) # 45km2 - area of bloom

@dataclass
class bloom_params:
    growth_rate_percent: float = 0.16
    peak_day: float = 9
    deccay_rate_percent: float = 0.125
    max_area_spread_km2: float = 45_000
    bloom_lifetime_days: float = 45

    @property
    def initial_area_spread_km2(self) -> float:
        return self.max_area_spread_km2  * np.exp(-self.growth_rate_percent * self.peak_day)

@dataclass
class bloom_state:
    bloom_center_km: np.ndarray = field(default_factory = lambda: (np.array([0.0,0.0])))
    bloom_area_km2: float = 0.0
    current_bloom_day: float = 0.0

class HycomBloomEnvironment:
    def __init__(self, u_firld, v_field, params: bloom_params = None,
                 grid_limits_km: float = 400.0, grid_resolution_km: int = 200,
                 sat_window_duration_hours: float = 1.0, sat_window_periodicity_hours: float = 12.0,
                 recenter_threshold_fraction: float = 0.5,
                 origin_ref_lat: float = None, origin_ref_lon: float = None,
                 origin_date: str = None, refetch_on_recenter: bool = True):

        self.params = params or bloom_params()
        coords = np.linspace(-grid_limits_km, grid_limits_km, grid_resolution_km)
        self.X, self.Y = np.meshgrid(coords, coords)
        self.dx = coords[1] - coords[0]
        self.grid_limits_km = grid_limits_km
        self.grid_resolution_km = grid_resolution_km

        self.u, self.v = u_firld, v_field
        self.grid_offset_km = np.array([0.0,0.0])
        self.recenter_threshold_km = recenter_threshold_fraction * grid_limits_km
        self.origin_ref_lat, self.origin_ref_lon = origin_ref_lat, origin_ref_lon
        self.origin_date = np.datetime64(origin_date) if origin_date else None
        self.refetch_on_recenter = refetch_on_recenter and (origin_ref_lat is not None)
        self.elapsed_time_hours = 0.0
        seed_area = self.params.initial_area_spread_km2
        seed_radius = np.sqrt(seed_area/np.pi)
        self.C = np.exp(-(self.X **2 + self.Y ** 2) / (2 * seed_radius**2))
        self.state = bloom_state(bloom_center_km=np.array([0.0, 0.0]),
                                 bloom_area_km2=seed_area, current_bloom_day=0.0)
        self._rescale_to_target_area(seed_area)
        self.sat_window_duration_hours = sat_window_duration_hours
        self.sat_window_periodicity_hours = sat_window_periodicity_hours

    def _area_rn(self, day, fade_duration_days: float = 10.0):
        if day <= self.params.peak_day:
            area  = self.params.initial_area_spread_km2 * np.exp(self.params.growth_rate_percent * day)
        else:
            area = self.params.max_area_spread_km2 * np.exp(-self.params.deccay_rate_percent * (day - self.params.peak_day))
        lifetime = self.params.bloom_lifetime_days
        taper_start = lifetime - fade_duration_days
        if day >= lifetime:
            area = 0.0
        elif day > taper_start:
            taper_function = (day - taper_start) / fade_duration_days
            taper = 0.5 * (1 + np.cos(np.pi * taper_function))
            area = area * taper
        return max(area, 0.0)

    def _is_dead(self, day):
        return day >= self.params.bloom_lifetime_days

    def _cell_area_km2(self):
        return (2 * self.grid_limits_km / self.grid_resolution_km) ** 2

    def _rescale_to_target_area(self, target_area_km2):
        current_mass = self.C.sum() * self._cell_area_km2()
        if current_mass > 1e-6 and target_area_km2 > 0:
            self.C = self.C * (target_area_km2 / current_mass)
        self.C = np.clip(self.C, 0, None)

    def _update_derived_state(self):
        total = self.C.sum()
        if total > 1e-9:
            cx = (self.C * self.X).sum() / total
            cy = (self.C * self.Y).sum() / total
            self.state.bloom_center_km = np.array([cx, cy])
        self.state.bloom_area_km2 = self.C.sum() * self._cell_area_km2()


    def _step(self, dt_step_day: float = 1.0):
        self.state.current_bloom_day += dt_step_day
        self.elapsed_time_hours += dt_step_day * 24.0
        self.C = semi_lagrangian_advect(self.C, self.u, self.v, dt_step_day, self.dx)
        target_area = self._area_rn(self.state.current_bloom_day)
        self._rescale_to_target_area(target_area)
        self._update_derived_state()
        self._recenter_if_needed()
        return self.state

    def _recenter_if_needed(self):
        cx, cy = self.state.bloom_center_km
        if max(abs(cx), abs(cy)) < self.recenter_threshold_km:
            return
        shift_km = np.array([cx, cy])
        shift_cells = shift_km / self.dx
        self.C = ndi_shift(self.C, shift = (-shift_cells[1], - shift_cells[0]), order = 1, mode = "constant")
        self.grid_offset_km = self.grid_offset_km + shift_km
        self.state.bloom_center_km = self.state.bloom_center_km - shift_km
        self._update_derived_state()

        if self.refetch_on_recenter:
            new_lat = self.origin_ref_lat + self.grid_offset_km[1] / 111.32
            new_lon = self.origin_ref_lon + self.grid_offset_km[0] / (111.32 * np.cos(np.deg2rad(self.origin_ref_lat)))
            new_date = self.origin_date + np.timedelta64(int(self.state.current_bloom_day), "D")

            try:
                new_u, new_v = fetch_hycom_current_field(self.X, self.Y, ref_lat=new_lat, ref_lon=new_lon,
                                                         date=str(new_date)[:10], timeout_note=False)
                self.u, self.v = new_u, new_v
            except Exception as e:
                print(f"[recenter WARNING] re-fetch failed ({e}); reusing previous field.")

    def _is_bloom_visible_by_sat(self):
        '''to avoid the floating pt drift of elapsed_time_hours the near start n wraparound were added'''
        t = self.elapsed_time_hours % self.sat_window_periodicity_hours
        near_start = t < (self.sat_window_duration_hours  + 1e-6)
        near_wraparound = t > (self.sat_window_periodicity_hours - 1e-6)
        return near_start or near_wraparound

    def _intensity_field(self):
        if not self._is_bloom_visible_by_sat():
            return None
        return self._rendering()

    def _rendering(self):
        peak = self.C.max()
        if peak <=1e-9:
            return np.zeros_like(self.C)
        return np.clip(self.C / peak, 0, 1)

    def _is_evolving(self):
        return (self.state.current_bloom_day <= self.params.bloom_lifetime_days
                and self.state.bloom_area_km2 <= self.params.max_area_spread_km2)


# OCEAN CURRENTS
HYCOM_OPENDAP_URL = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z"


def fetch_hycom_current_field(X: np.ndarray, Y: np.ndarray, ref_lat: float, ref_lon: float, date: str,
                              timeout_note: bool = True):
    if timeout_note:
        print('fetching hycom current field with timeout')
    ds = xr.open_dataset(HYCOM_OPENDAP_URL, decode_times = False)
    ds = xr.decode_cf(ds, decode_times=True, mask_and_scale=True, drop_variables = ["tau"] if "tau" in ds.variables else None)
    time_values = ds["time"].values
    target_time = np.datetime64(date)
    time_diffs = np.abs(time_values - target_time)
    nearest_time_idx = int(np.argmin(time_diffs))
    nearest_time = time_values[nearest_time_idx]
    gap_days = abs((nearest_time - target_time)/ np.timedelta64(1, "D"))
    if gap_days > 7:
        print(f"[WARNING] - the nearest available hycom data is {gap_days} days away")
    ds_at_time = ds.isel(time=nearest_time_idx)

    lat_min = ref_lat + Y.min() / 111.32
    lat_max = ref_lat + Y.max() / 111.32
    lon_min = ref_lon + X.min() / (111.32 * np.cos(np.deg2rad(ref_lat)))
    lon_max = ref_lon + X.max() / (111.32 * np.cos(np.deg2rad(ref_lat)))
    lon_min_hycom, lon_max_hycom = lon_min % 360, lon_max % 360

    subset = ds_at_time.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min_hycom, lon_max_hycom))
    u_raw = subset["water_u"].isel(depth = 0).values
    v_raw = subset["water_v"].isel(depth = 0).values
    lats, lons = subset["lat"].values, subset["lon"].values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    x_km_grid = (lon_grid - ref_lon) * 111.32 * np.cos(np.deg2rad(ref_lat))
    y_km_grid = (lat_grid - ref_lat) * 111.32
    valid = ~np.isnan(u_raw) & ~np.isnan(v_raw)
    points = np.column_stack([x_km_grid[valid], y_km_grid[valid]])

    u_kmday = u_raw[valid] * 86.4
    v_kmday = v_raw[valid] * 86.4
    u_field = griddata(points, u_kmday, (X,Y), method = 'linear', fill_value = np.nanmean(u_kmday))
    v_field = griddata(points, v_kmday, (X,Y), method = 'linear', fill_value = np.nanmean(v_kmday))
    return u_field, v_field


def semi_lagrangian_advect(C, u, v, dt_days, dx):
    grid_size = C.shape[0]
    row_idx, col_idx = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing="ij")
    src_row = row_idx - (v * dt_days) / dx
    src_col = col_idx - (u * dt_days) / dx
    return map_coordinates(C, [src_row, src_col], order = 1, mode = "constant", cval = 0.0)




# ......................................................................................................................
#                                                    DVS ENCODER
# ......................................................................................................................
PEAK_RADIANCE_NW = 255.0  # nW/cm^2/sr
FLOOR_NW = 0.5  # detection floor, avoids log(0)


@dataclass
class Dvs_Params:
    shot_noise_coeff: float = 0.0064  # ~ 15% of mean contrast threshold
    threshold_mismatch_percent: float = 2.1  # real measured array mismatch reported in Lichtsteiner et al. 2008 (2.1%)
    contrast_threshold_percent: float = 15.0
    num_substeps: int = 10

    @property
    def contrast_threshold_log10(self) -> float:
        return np.log10(1.0 + self.contrast_threshold_percent / 100.0)



class DVS_Encoder:
    # INITIALIZATION
    def __init__(self, params: Dvs_Params = None, rng: np.random.Generator = None):
        self.params = params or Dvs_Params()
        self.rng = rng or np.random.default_rng()
        self._threshold_map = None  # per-pixel threshold map for event generation built once we know the grid shape

    def _get_frame(self, env):
        '''picks bw stage 1 and 2's gaussian o/p for photodiode'''
        if hasattr(env, '_intensity_field'):
            return env._intensity_field()  # stage 2
        return env._rendering()  # stage 1


    # PHOTO-DIODE
    def _DVS_1_photodiode(self, env) -> np.ndarray:
        ''' _photodiode returns a 2D array of the gaussian bloom which is our i/p / whatever is collected by the photodiode.'''
        raw = self._get_frame(env)
        return raw

    # PHOTORECEPTOR - i/p normalized field from photo-diode
    def _DVS_2_photoreceptor(self, normalized_field: np.ndarray) -> np.ndarray:
        radiance_NW = normalized_field * PEAK_RADIANCE_NW  # [nW - nano watts]
        clipped_radiance_NW = np.clip(radiance_NW, FLOOR_NW, None)  # [nW]
        log_voltage = np.log10(clipped_radiance_NW)  # [dimensionless]

        # SHOT NOISE
        # dim signals are noisier as the photons hitting the detector window follow poission stats - so its given by a coeff(encapsulates camera properties) / sqrt(how bright?)
        noise_std = self.params.shot_noise_coeff / np.sqrt(clipped_radiance_NW)  # [dimensionless]
        noise = self.rng.normal(0.0, noise_std)
        return log_voltage + noise

    # VOLTAGE CHANGE OVER FRAMES
    def _DVS_3_pixel_change(self, log_voltage: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ''' it tells us how much the pixel has changed from it's prev log voltage value '''
        amplified = log_voltage - reference
        return amplified

    # THRESHOLD POTENTIAL
    def _DVS_4_threshold_check(self, delta: np.ndarray) -> np.ndarray:
        if self._threshold_map is None or self._threshold_map.shape != delta.shape:
            self._build_threshold_map(delta.shape)

        n_events = np.floor(np.abs(delta) / self._threshold_map).astype(
            int)  # the no of whole events that are worth considering for each pixel
        polarity = np.sign(delta)  # to identify growth or decay events
        return n_events, polarity

    # WHOLE EVENT SIZE
    def _build_threshold_map(self, shape: tuple):
        p = self.params
        mean_threshold = p.contrast_threshold_log10  # ch
        sigma = mean_threshold * (p.threshold_mismatch_percent / 100.0)
        self._threshold_map = np.clip(self.rng.normal(mean_threshold, sigma, size=shape), 1e-6,
                                      None)  # gen the threshold gaussian map array with random vals

    # DID THE PIXEL FIRE?
    def _DVS_5_fire_and_reset(self, n_events, polarity, reference, threshold_map):
        n_on_counts = np.where(polarity > 0, n_events, 0)  # condition, if true, if false
        n_off_counts = np.where(polarity < 0, n_events, 0)
        fired = n_events > 0
        updated_reference = np.where(fired, reference + polarity * threshold_map * n_events, reference)
        return n_on_counts, n_off_counts, updated_reference

    def encode(self, env, dt_days: float = 1.0):
        p = self.params
        sub_dt = dt_days / p.num_substeps
        log_voltage_frames = []
        for _ in range(p.num_substeps):
            # 1 & 2
            try:
                env._step(dt_step_day=sub_dt)
            except TypeError:
                env._step(dt_days=sub_dt)
            raw = self._DVS_1_photodiode(env)
            log_voltage_frames.append(self._DVS_2_photoreceptor(raw) if raw is not None else None)

        event_bins = []
        reference = None
        for log_voltage in log_voltage_frames:
            if log_voltage is None:
                continue
            if reference is None:
                reference = log_voltage.copy()
                continue
            delta = self._DVS_3_pixel_change(log_voltage, reference)
            n_events, polarity = self._DVS_4_threshold_check(delta)
            n_on_counts, n_off_counts, reference = self._DVS_5_fire_and_reset(n_events, polarity, reference,
                                                                              self._threshold_map)
            event_bins.append((n_on_counts, n_off_counts))

        if not event_bins:
            H, W = env.X.shape
            return np.zeros((0, 2, H, W), dtype=int)  # 2  rep 2 channels (ON/OFF)
        return np.stack(event_bins,
                        axis=0)  # [T,2,H,W] where T is the 1 - no of substeps, 2 is ON/OFF, H is height, W is width

# ......................................................................................................................
#                                                    SNN
# ......................................................................................................................

class SNN(nn.Module):
    def __init__(self, v_threshold: float = 0.3, tau: float = 1.5, weight_scale: float = 2.0):
        super().__init__()
        ''' 3 layers of conv + lif nodes each using:
            kernel size - the ref array passed over patches
            padding     - rings of zero over the array to avoid unaccounted columns
            stride      - the positional skipping of the xolumns as u move the kernel over the surface '''

        self.conv1 = nn.Conv2d(2, 16, kernel_size=5, padding=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=2)

        with torch.no_grad():
            self.conv1.weight *= weight_scale
            self.conv2.weight *= weight_scale
            self.conv3.weight *= weight_scale


        self.lif1 = neuron.LIFNode(tau = tau, v_threshold = v_threshold, surrogate_function = surrogate.ATan())
        self.lif2 = neuron.LIFNode(tau = tau, v_threshold = v_threshold,surrogate_function = surrogate.ATan())
        self.lif3 = neuron.LIFNode(tau = tau, v_threshold = v_threshold,surrogate_function = surrogate.ATan())

        # the o/p must be flattened out to a single array
        flattened_size = 64 * 25 * 25
        self.readout = nn.Linear(flattened_size, 3) # -> o/p (center_x_km, center_y_km, area_km2)

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
            readout_accumulated = readout_accumulated + self.readout(x).squeeze(0)
        return readout_accumulated / T # avg readout over all time steps

class _TwoFrameEnv:
    def __init__(self, frame, X, prev_frame = None):
        self._frame = frame
        self._prev_frame = prev_frame if prev_frame is not None else frame
        self.X = X
        self._render_count = 0

    def _rendering(self):
        self._render_count += 1
        if self._render_count == 1:
            return self._prev_frame
        return self._frame

    def _step(self, dt_step_day=None, dt_days=None):
        return None





def pick_reference_frame(frame_history: list, current_day: float, target_gap_days: float = 1.0):
    if not frame_history:
        return None
    best_frame, best_diff = None, np.inf
    for day, frame in frame_history:
        gap = current_day - day
        if gap <=0:
            continue
        diff = abs(gap - target_gap_days)
        if diff < best_diff:
            best_diff, best_frame = diff, frame
    return best_frame if best_frame is not None else frame_history[-1][1]

def snn_predict(snn: SNN, encoder: DVS_Encoder, frame: np.ndarray, env, prev_frame: np.ndarray = None):
    frozen = _TwoFrameEnv(frame, env.X, prev_frame = prev_frame)
    events = encoder.encode(frozen, dt_days = 1.0)
    if events.shape[0] == 0:
        return None
    events_tensor = torch.tensor(events, dtype=torch.float32)
    with torch.no_grad():
        pred = snn(events_tensor)
    return pred.numpy()



# ......................................................................................................................
#                                                SHARED INTEREST MAP
# ......................................................................................................................

@dataclass
class InterestMapParams:
    deccay_rate: float = 0.95
    deposit_sigma_km: float = 50.0


class InterestMap:
    def __init__(self, X: np.ndarray, Y: np.ndarray, params: InterestMapParams = None):
        self.X = X
        self.Y = Y
        self.params = params or InterestMapParams()
        self.grid = np.zeros_like(X)

    def deposit(self, x_km: float, y_km: float, amount: float = 1.0):
        ''' this fn creates a gaussian view of area of interest for satellites to analyse the region rather than a single point '''
        sigma = self.params.deposit_sigma_km
        dist_sq = (self.X - x_km) ** 2 + (self.Y - y_km) ** 2
        blob = amount * np.exp(- dist_sq / (2 * sigma ** 2))
        self.grid = self.grid + blob

    def evaporate(self):
        self.grid = self.grid * self.params.deccay_rate

    def step(self, deposits: list = None):
        self.evaporate()
        if deposits:
            for x_km, y_km, amount in deposits:
                self.deposit(x_km, y_km, amount)

    def value_at(self, x_km: float, y_km: float) -> float:
        dist_sq = (self.X - x_km) ** 2 + (self.Y - y_km) ** 2
        idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        return self.grid[idx]

    def highest_interest_location(self) -> tuple:
        if self.grid.max() <=0:
            return 0.0, 0.0
        idx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        return self.X[idx], self.Y[idx]


# ......................................................................................................................
#                                                       SAC
# ......................................................................................................................
# constants
OBS_DIM = 8 # [keplerian elements [6], distance to target, local interest value]
ACTION_DIM = 3 # delta v dim * 3
MAX_ACTION_KMS = 0.1 # max delta v mag per axis
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class SAC_Actor(nn.Module):
    def __init__ (self, obs_dim: int= OBS_DIM, action_dim: int = ACTION_DIM,  hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
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

class SAC_Critic(nn.Module):
    def __init__(self, num_agents: int, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128):
        super().__init__()
        joint_dim = num_agents * (obs_dim + action_dim)

        self.q1 = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
                    nn.Linear(joint_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )

    def forward(self, joint_obs: torch.Tensor, joint_actions: torch.Tensor):
        x = torch.cat([joint_obs, joint_actions], dim=1)
        return self.q1(x), self.q2(x)

class ReplayBuffer:
    def __init__(self, capacity: int =  100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size, num_agents):
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        obs = [torch.tensor(np.stack([b[0][a] for b in batch]), dtype=torch.float32) for a in range(num_agents)]
        actions = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        rewards = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32).unsqueeze(-1)
        next_obs = [torch.tensor(np.stack([b[3][a] for b in batch]), dtype=torch.float32) for a in range(num_agents)]
        dones = torch.tensor(np.stack([b[4] for b in batch]), dtype=torch.float32).unsqueeze(-1)
        return {"obs": obs, "joint_actions_taken": actions, "rewards": rewards, "next_obs": next_obs, "dones": dones}

    def __len__(self):
        return len(self.buffer)

def compute_reward(predicted_state, true_state, action, is_observable,
                   fuel_weight = 0.5, visibility_bonus = 0.1, state_scale = None):
    pred, true = np.asarray(predicted_state, dtype = float), np.asarray(true_state, dtype = float)
    if state_scale is not None:
        pred, true = pred / state_scale, true / state_scale
    tracking_error = np.linalg.norm(pred - true)
    fuel_cost = fuel_weight * np.linalg.norm(action)
    vis_bonus = visibility_bonus if is_observable else 0.0
    return float( - tracking_error - fuel_cost + vis_bonus)



class SAC_Trainer:
    def __init__(self, num_agents: int, lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2):
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = SAC_Actor()
        self.critic = SAC_Critic(num_agents=num_agents)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = lr)

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

    def save_checkpoint(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "num_agents": self.num_agents,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path)
        if ckpt.get("num_agents") is not None and ckpt["num_agents"] != self.num_agents:
            raise ValueError(f"Number of agents mismatch:"
                             f"\n ckpt in {path} has {ckpt['num_agents']} agents"
                             f"\n trainer was constructed with {self.num_agents} agents ")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])


# ......................................................................................................................
#                                                           ORBITAL MECHANICS
# ......................................................................................................................
MU_EARTH_KM3_S2 = 398_600.4418
RE_KM = 6378.137
J2 = 1.08262668e-3
EARTH_ROTATION_DEG_PER_DAY = 360.9856235


@dataclass
class Keplerian_Elements:
    a_km: float
    e: float
    i_deg: float
    raan_deg: float
    argp_deg: float
    mean_anomaly_deg: float
    epoch_day: float = 0.0


def mean_motion_rad_s(elements: Keplerian_Elements):
    return np.sqrt(MU_EARTH_KM3_S2 / elements.a_km ** 3)


def j2_secular_rates_deg_per_day(elements: Keplerian_Elements):
    n = mean_motion_rad_s(elements)  # [rad/s]
    n_deg_day = np.rad2deg(n) * 86400.0  # [deg/day]
    p = elements.a_km * (1 - elements.e ** 2)
    factor = 1.5 * J2 * (RE_KM / p) ** 2
    i_rad = np.deg2rad(elements.i_deg)
    raan_dot = -factor * n_deg_day * np.cos(i_rad)
    argp_dot = 0.5 * factor * n_deg_day * (5 * np.cos(i_rad) ** 2 - 1)
    M_dot = n_deg_day + 0.5 * factor * n_deg_day * np.sqrt(1 - elements.e ** 2) * (3 * np.cos(i_rad) ** 2 - 1)
    return raan_dot, argp_dot, M_dot


def propagate(elements: Keplerian_Elements, t_days: float) -> Keplerian_Elements:
    dt = t_days - elements.epoch_day
    raan_dot, argp_dot, M_dot = j2_secular_rates_deg_per_day(elements)
    return Keplerian_Elements(
        a_km=elements.a_km,
        e=elements.e,
        i_deg=elements.i_deg,
        raan_deg=(elements.raan_deg + raan_dot * dt) % 360.0,
        argp_deg=(elements.argp_deg + argp_dot * dt) % 360.0,
        mean_anomaly_deg=(elements.mean_anomaly_deg + M_dot * dt) % 360.0,
        epoch_day=t_days
    )


def _solve_kepler(M_rad: float, e: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    E = M_rad if e < 0.8 else np.pi
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M_rad) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def elements_to_eci(elements: Keplerian_Elements) -> np.ndarray:
    M_rad = np.deg2rad(elements.mean_anomaly_deg)
    E = _solve_kepler(M_rad, elements.e)
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + elements.e) * np.sin(E / 2),
        np.sqrt(1 - elements.e) * np.cos(E / 2))

    r = elements.a_km * (1 - elements.e * np.cos(E))
    x_pf = r * np.cos(true_anomaly)
    y_pf = r * np.sin(true_anomaly)
    i_rad = np.deg2rad(elements.i_deg)
    raan_rad = np.deg2rad(elements.raan_deg)
    argp_rad = np.deg2rad(elements.argp_deg)

    cos_r, sin_r = np.cos(raan_rad), np.sin(raan_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_w, sin_w = np.cos(argp_rad), np.sin(argp_rad)

    x = (cos_r * cos_w - sin_r * sin_w * cos_i) * x_pf + (-cos_r * sin_w - sin_r * cos_w * cos_i) * y_pf
    y = (sin_r * cos_w + cos_r * sin_w * cos_i) * x_pf + (-sin_r * sin_w + cos_r * cos_w * cos_i) * y_pf
    z = (sin_w * sin_i) * x_pf + (cos_w * sin_i) * y_pf

    return np.array([x, y, z])


def elements_to_eci_state(elements: Keplerian_Elements) -> tuple:
    M_rad = np.deg2rad(elements.mean_anomaly_deg)
    E = _solve_kepler(M_rad, elements.e)
    true_anomaly = 2 * np.arctan2(
        np.sqrt(1 + elements.e) * np.sin(E / 2),
        np.sqrt(1 - elements.e) * np.cos(E / 2))

    r = elements.a_km * (1 - elements.e * np.cos(E))
    p = elements.a_km * (1 - elements.e ** 2)
    x_pf = r * np.cos(true_anomaly)
    y_pf = r * np.sin(true_anomaly)

    i_rad = np.deg2rad(elements.i_deg)
    raan_rad = np.deg2rad(elements.raan_deg)
    argp_rad = np.deg2rad(elements.argp_deg)

    term_1 = np.sqrt(MU_EARTH_KM3_S2 / p)
    vx_pf = -term_1 * np.sin(true_anomaly)
    vy_pf = term_1 * (elements.e + np.cos(true_anomaly))

    cos_r, sin_r = np.cos(raan_rad), np.sin(raan_rad)
    cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
    cos_w, sin_w = np.cos(argp_rad), np.sin(argp_rad)

    def rotate(x_pf, y_pf):
        x = (cos_r * cos_w - sin_r * sin_w * cos_i) * x_pf + (-cos_r * sin_w - sin_r * cos_w * cos_i) * y_pf
        y = (sin_r * cos_w + cos_r * sin_w * cos_i) * x_pf + (-sin_r * sin_w + cos_r * cos_w * cos_i) * y_pf
        z = (sin_w * sin_i) * x_pf + (cos_w * sin_i) * y_pf
        return np.array([x, y, z])

    pos_eci = rotate(x_pf, y_pf)
    vel_eci = rotate(vx_pf, vy_pf)

    return pos_eci, vel_eci


def cart_to_kep(pos_eci: np.ndarray, vel_eci: np.ndarray, epoch_day: float) -> Keplerian_Elements:
    mu = MU_EARTH_KM3_S2
    r_vec, v_vec = pos_eci, vel_eci
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    n_vec = np.cross([0, 0, 1], h_vec)
    n = np.linalg.norm(n_vec)

    e_vec = ((v ** 2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    energy = v ** 2 / 2 - mu / r

    if not np.isfinite(energy) or energy >= 0:
        a = RE_KM + 200.0 # just a failsafe alt for any issues in traj during long episodes
        e = 0.001
        print("[WARNING] - energy >= 0 so val was clamped to failsafe vals of alt = 200 km, e = 0.001 ")
    else:
        a = -mu / (2 * energy)
        if not np.isfinite(e) or e >= 1.0:
            print("[WARNING] - ecentricity >= 1 so val was clamped to e = 0.999")
            e = 0.999

    i_rad = np.arccos(np.clip(h_vec[2] / h, -1, 1))
    raan_rad = np.arccos(np.clip(n_vec[0] / n, -1, 1)) if n > 1e-10 else 0.0

    if n_vec[1] < 0:
        raan_rad = 2 * np.pi - raan_rad

    if n > 1e-10 and e > 1e-10:
        argp_rad = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1, 1))
        if e_vec[2] < 0:
            argp_rad = 2 * np.pi - argp_rad
    else:
        argp_rad = 0.0

    if e > 1e-10:
        true_anomaly = np.arccos(np.clip(np.dot(e_vec, r_vec) / (e * r), -1, 1))
        if np.dot(r_vec, v_vec) < 0:
            true_anomaly = 2 * np.pi - true_anomaly
    else:
        true_anomaly = 0.0

    E = 2 * np.arctan2(np.sqrt(max(1 - e, 0)) * np.sin(true_anomaly / 2), np.sqrt(max(1 + e, 0)) * np.cos(true_anomaly / 2))
    M_rad = E - e * np.sin(E)

    return Keplerian_Elements(
        a_km=a, e=e, i_deg=np.rad2deg(i_rad), raan_deg=np.rad2deg(raan_rad) % 360,
        argp_deg=np.rad2deg(argp_rad) % 360.0,
        mean_anomaly_deg=np.rad2deg(M_rad) % 360.0,
        epoch_day=epoch_day
    )


def apply_delta_v(elements: Keplerian_Elements, delta_v_kms: np.ndarray, t_days: float) -> Keplerian_Elements:
    propagated = propagate(elements, t_days)
    pos_eci, vel_eci = elements_to_eci_state(propagated)
    new_vel_eci = vel_eci + delta_v_kms
    return cart_to_kep(pos_eci, new_vel_eci, epoch_day=t_days)


def eci_to_subpoint(pos_eci: np.ndarray, t_days: float) -> tuple:
    x, y, z = pos_eci
    r_xy = np.sqrt(x ** 2 + y ** 2)
    lat_deg = np.rad2deg(np.arctan2(z, r_xy))
    earth_rotation_deg = (EARTH_ROTATION_DEG_PER_DAY * t_days) % 360.0
    lon_deg = np.rad2deg(np.arctan2(y, x)) - earth_rotation_deg
    lon_deg = ((lon_deg + 180) % 360) - 180
    return lat_deg, lon_deg


def _great_circle_distance_km(lat1, lon1, lat2, lon2, earth_radius_km = 6378.137) -> float:
    ''' compute circular dist bw 2 coords'''
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(np.clip(a,0,1)))


def default_sso_pair(raan_offset_deg: float = 180.0) -> tuple:
    sat_a = Keplerian_Elements(a_km=RE_KM + 824.0,
                               e=0.0001,
                               i_deg=98.7,
                               raan_deg=0.0, argp_deg=0.0, mean_anomaly_deg=0.0, epoch_day=0.0)
    sat_b = Keplerian_Elements(a_km=RE_KM + 824.0,
                               e=0.0001,
                               i_deg=98.7,
                               raan_deg=raan_offset_deg, argp_deg=0.0, mean_anomaly_deg=0.0, epoch_day=0.0)
    return sat_a, sat_b

def build_observation(sat_elements: Keplerian_Elements, imap: InterestMap, ref_lat: float, ref_lon: float):
    a_norm = sat_elements.a_km / 10_000.0
    e_norm = sat_elements.e
    i_norm = sat_elements.i_deg /180.0
    raan_norm = sat_elements.raan_deg / 360.0
    argp_norm = sat_elements.argp_deg / 360.0
    M_norm = sat_elements.mean_anomaly_deg / 360.0
    peak_x, peak_y = imap.highest_interest_location()
    pos_eci = elements_to_eci(sat_elements)
    lat, lon = eci_to_subpoint(pos_eci, sat_elements.epoch_day)
    sat_x_km = (lon - ref_lon) * 111.32 * np.cos(np.deg2rad(ref_lat))
    sat_y_km = (lat - ref_lat) * 111.32
    dist_to_peak = np.sqrt((sat_x_km - peak_x) ** 2 + (sat_y_km - peak_y) ** 2) / (2 * GRID_LIMIT_KM)
    local_interest = imap.value_at(sat_x_km, sat_y_km)
    return np.array([a_norm, e_norm, i_norm, raan_norm, argp_norm, M_norm, dist_to_peak, local_interest], dtype=np.float32)

# ......................................................................................................................
#                                                        N Satellite Environment
# ......................................................................................................................
def make_constellation(num_satellites: int, altitude_km: float = 824.0,
                        inclination_deg: float = 98.7, raan_spacing_deg: float = None,
                        clustered_spread_deg: float = None):
    '''sats are either evenly spaced out in raan or clustered_spread_deg spreads em out within that deg of raan band '''
    if clustered_spread_deg is not None and num_satellites >1:
        raans = np.linspace(0, clustered_spread_deg, num_satellites)
    else:
        spacing = raan_spacing_deg or (360/num_satellites)
        raans = [i * spacing for i in range(num_satellites)]
    return [
        Keplerian_Elements(a_km=RE_KM + altitude_km, e=0.0001, i_deg=inclination_deg,
                           raan_deg=raan, argp_deg=0.0, mean_anomaly_deg=0.0, epoch_day=0.0)
        for raan in raans
    ]

def compute_collision_penalty(satellites: list, t_days: float, safe_distance_km: float, weight: float = 1.0):
    if len(satellites) <2:
        return 0.0
    positions = [elements_to_eci(propagate(elem, t_days)) for elem in satellites]
    penalty = 0.0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist_km = np.linalg.norm(positions[i] - positions[j])
            if dist_km < safe_distance_km:
                penalty -= weight * (safe_distance_km - dist_km) / safe_distance_km
    return penalty


class NSatelliteBloomEnvironment(HycomBloomEnvironment):
    def __init__(self, u_field, v_field, sat_elements: list = None, num_satellites: int = 2,
                 swath_half_width_km: float = 400.0, altitude_km: float = 824.0,
                 raan_clustered_spread_deg: float = None, **kwargs):

        super().__init__(u_field, v_field, **kwargs)
        if sat_elements is not None:
            self.satellites = list(sat_elements)
        else:
            self.satellites = make_constellation(num_satellites, altitude_km=altitude_km,
                                                 clustered_spread_deg=raan_clustered_spread_deg)
        self.swath_half_width_km = swath_half_width_km
        self._elapsed_days = 0.0

    def _resolve_index(self, name):
        if isinstance(name, str):
            if name == "A":
                return 0
            if name == "B":
                return 1
            return int(name)
        return int(name)

    @property
    def sat_a_elements(self):
        return self.satellites[0]

    @sat_a_elements.setter
    def sat_a_elements(self, value):
        self.satellites[0] = value

    @property
    def sat_b_elements(self):
        return self.satellites[1]

    @sat_b_elements.setter
    def sat_b_elements(self, value):
        self.satellites[1] = value

    def _bloom_real_location(self):
        x_km, y_km = self.state.bloom_center_km
        abs_x_km, abs_y_km = x_km + self.grid_offset_km[0], y_km + self.grid_offset_km[1]
        lat = self.origin_ref_lat + abs_y_km / 111.32
        lon = self.origin_ref_lon + abs_x_km / (111.32 * np.cos(np.deg2rad(self.origin_ref_lat)))
        return lat, lon

    def visible_satellites(self, check_substeps: int = 20):
        bloom_lat, bloom_lon = self._bloom_real_location()
        result = {}
        for idx, elements in enumerate(self.satellites):
            name = "A" if idx == 0 else ("B" if idx == 1 else idx)
            visible = False
            for step_i in range(check_substeps):
                t_check = self._elapsed_days - (1.0 / 24.0) + step_i * (1.0 / 24.0 / check_substeps)
                propagated = propagate(elements, t_check)
                pos_eci = elements_to_eci(propagated)
                sat_lat, sat_lon = eci_to_subpoint(pos_eci, t_check)
                if _great_circle_distance_km(sat_lat, sat_lon, bloom_lat, bloom_lon) <= self.swath_half_width_km:
                    visible = True
                    break
            result[name] = visible
        return result

    def get_satellite_view(self, sat_name, visibility: dict = None):
        visibility = visibility or self.visible_satellites()
        if not visibility.get(sat_name, False):
            return None
        return self._rendering()

    def _step(self, dt_step_day: float = 1.0):
        state = super()._step(dt_step_day=dt_step_day)
        self._elapsed_days += dt_step_day
        return state

    def apply_satellite_action(self, sat_name, delta_v_kms: np.ndarray):
        idx = self._resolve_index(sat_name)
        new_elements = apply_delta_v(self.satellites[idx], delta_v_kms, self._elapsed_days)
        self.satellites[idx] = new_elements
        return new_elements

    def collision_penalty(self, weight: float = 1.0, safe_distance_km: float = 10.0):
        return compute_collision_penalty(self.satellites, self._elapsed_days, safe_distance_km=safe_distance_km,
                                         weight=weight)

TwoSatelliteBloomEnvironment = NSatelliteBloomEnvironment  # num_satellites defaults to 2

# ......................................................................................................................
#                                                      LITERATURE SIGHTINGS (Miller et al. 2021)
# ......................................................................................................................

TABLE_1_ALL_CASES = [
    {"name": "2013 Socotra", "lat": 15.0, "lon": 58.0, "start": "2013-07-31", "end": "2013-08-13", "area_km2": 9000},
    {"name": "2014 Banda", "lat": -5.0, "lon": 126.0, "start": "2014-08-20", "end": "2014-08-24", "area_km2": 18000},
    {"name": "2015 Somalia Phase 1", "lat": 0.0, "lon": 44.0, "start": "2015-01-15", "end": "2015-01-28",
     "area_km2": 23000},
    {"name": "2015 Somalia Phase 2", "lat": 0.0, "lon": 50.0, "start": "2015-01-21", "end": "2015-01-26",
     "area_km2": 60000},
    {"name": "2015 Banda", "lat": -5.0, "lon": 129.0, "start": "2015-08-12", "end": "2015-08-18", "area_km2": 30000},
    {"name": "2015 Socotra Phase 1", "lat": 10.0, "lon": 53.0, "start": "2015-09-07", "end": "2015-09-11",
     "area_km2": 750},
    {"name": "2015 Socotra Phase 2", "lat": 11.0, "lon": 52.0, "start": "2015-09-12", "end": "2015-09-20",
     "area_km2": 12000},
    {"name": "2017 Somalia", "lat": 2.0, "lon": 47.0, "start": "2017-01-21", "end": "2017-01-31", "area_km2": 17000},
    {"name": "2018 Somalia Phase 1", "lat": 2.0, "lon": 47.0, "start": "2018-01-12", "end": "2018-01-19",
     "area_km2": 30000},
    {"name": "2018 Somalia Phase 2", "lat": 5.0, "lon": 55.0, "start": "2018-01-19", "end": "2018-01-24",
     "area_km2": 15000},
    {"name": "2019 Somalia", "lat": 2.0, "lon": 50.0, "start": "2019-01-28", "end": "2019-02-07", "area_km2": 100000},
    {"name": "2019 Java Phase 1", "lat": -9.0, "lon": 110.0, "start": "2019-07-25", "end": "2019-08-09",
     "area_km2": 100000},
    {"name": "2019 Java Phase 2", "lat": -9.0, "lon": 110.0, "start": "2019-08-25", "end": "2019-09-07",
     "area_km2": 50000},
    {"name": "2019 Banda", "lat": -5.0, "lon": 127.0, "start": "2019-07-26", "end": "2019-08-04", "area_km2": 60000},
    {"name": "2021 Socotra/Somalia Ph1", "lat": -11.0, "lon": 58.0, "start": "2021-01-07", "end": "2021-01-22",
     "area_km2": 10000},
    {"name": "2021 Socotra/Somalia Ph2", "lat": 7.0, "lon": 52.0, "start": "2021-01-15", "end": "2021-01-18",
     "area_km2": 20000},
    {"name": "2021 Socotra", "lat": 8.0, "lon": 56.0, "start": "2021-02-07", "end": "2021-02-20", "area_km2": 6000},
]

# ......................................................................................................................
#                                                           DATASETS
# ......................................................................................................................
TEST_CASES = [c for c in TABLE_1_ALL_CASES if c["name"] in {"2019 Java Phase 1", "2019 Somalia"}]
VALIDATION_CASES = [c for c in TABLE_1_ALL_CASES if c["name"] in {"2019 Banda", "2021 Socotra"}]

areas = [c["area_km2"] for c in TABLE_1_ALL_CASES]
NW_INDIAN_OCEAN_CASES = [c for c in TABLE_1_ALL_CASES if c["lon"] < 70]
MARITIME_CONTINENT_CASES = [c for c in TABLE_1_ALL_CASES if c["lon"] >=70]
REGION_BOUNDS = {
    "nw_indian_ocean": {"lat": (min(c["lat"] for c in NW_INDIAN_OCEAN_CASES), max(c["lat"] for c in NW_INDIAN_OCEAN_CASES)),
                        "lon":  (min(c["lon"] for c in NW_INDIAN_OCEAN_CASES), max(c["lon"] for c in NW_INDIAN_OCEAN_CASES))},
    "maritime_continent": { "lat": (min(c["lat"] for c in MARITIME_CONTINENT_CASES), max(c["lat"] for c in MARITIME_CONTINENT_CASES)),
                            "lon":  (min(c["lon"] for c in MARITIME_CONTINENT_CASES), max(c["lon"] for c in MARITIME_CONTINENT_CASES))},
}

TRAINING_RANGES = {"area_km2": (min(areas), max(areas)), "regions": REGION_BOUNDS}

def sample_training_case(rng: np.random.Generator):
    region = "nw_indian_ocean" if rng.random() < (len(NW_INDIAN_OCEAN_CASES) / len(TABLE_1_ALL_CASES)) else "maritime_continent"
    bounds = REGION_BOUNDS[region]
    area = rng.uniform(*TRAINING_RANGES["area_km2"])
    lat = rng.uniform(*bounds["lat"])
    lon = rng.uniform(*bounds["lon"])
    start = np.datetime64("2018-12-04") # api stored records begin from
    end = np.datetime64("2026-01-01")
    span_days = int((end - start)/np.timedelta64(1, "D"))
    random_date = start + np.timedelta64(int(rng.integers(0, span_days)), "D")
    return {"lat": lat, "lon": lon, "area_km2": area, "region": region, "date": str(random_date)}



# just some naming inconsistensies
GRID_LIMIT_KM = GRID_SIZE_KM


