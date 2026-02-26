import numpy as np


def rbf_expand(values, num_centers=20, r_min=None, r_max=None, gamma=None):
    values = np.asarray(values)
    if r_min is None:
        r_min = float(np.min(values))
    if r_max is None:
        r_max = float(np.max(values))
    # Generate evenly spaced centers
    centers = np.linspace(r_min, r_max, num_centers)
    if gamma is None:
        # Set gamma so adjacent bases overlap well
        gamma = 1.0 / (centers[1] - centers[0]) ** 2
    # Compute RBF
    expanded = np.exp(-gamma * (values[..., None] - centers) ** 2)
    return expanded  # shape: (len(values), num_centers)


def dihedral_to_sin_cos(dihedrals_deg, radians=True):
    if not radians:
        dihedrals_rad = np.deg2rad(dihedrals_deg)
    else:
        dihedrals_rad = np.asarray(dihedrals_deg)

    sin_vals = np.sin(dihedrals_rad)
    cos_vals = np.cos(dihedrals_rad)
    return np.stack([sin_vals, cos_vals], axis=-1)


def normalize_angle(angle_deg, a_min=0.0, a_max=180.0):
    # If your angles can go up to 180, otherwise adjust a_max as needed
    angle_deg = np.asarray(angle_deg)
    return (angle_deg - a_min) / (a_max - a_min)
