# run_hpo/temps.py
import numpy as np
from copy import deepcopy

def build_temps(cfg, args, yaml_space=None, default=(300.0, 3100.0, 100.0), include_max=False):
    c = deepcopy(cfg)
    y = yaml_space or {}
    tmin  = float(getattr(args, "temp_min",  y.get("temp_min",  default[0])))
    tmax  = float(getattr(args, "temp_max",  y.get("temp_max",  default[1])))
    tstep = float(getattr(args, "temp_step", y.get("temp_step", default[2])))

    if tstep <= 0: raise ValueError(f"temp_step must be > 0 (got {tstep})")
    if tmax <= tmin: raise ValueError(f"temp_max ({tmax}) must be > temp_min ({tmin})")

    temps = np.arange(tmin, tmax, tstep, dtype=float)  # endpoint excluded (matches your current call)
    if include_max and (len(temps) == 0 or abs(temps[-1] - tmax) > 1e-9):
        temps = np.append(temps, tmax)
    return temps.tolist()
