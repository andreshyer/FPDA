from math import pi, sin, cos
from numpy import arange, argmin, argmax
from scipy.integrate import odeint


def new_drop_profile(bond_number, max_worthington_number, delta_s):
    # Calculate baseline drop profile
    def pendant_drop(y, _):
        xi, zi, phi_i = y
        d_x = cos(phi_i)
        d_z = sin(phi_i)
        d_phi = 2 - bond_number * zi - sin(phi_i) / xi
        return d_x, d_z, d_phi

    x0, z0, phi0 = delta_s, 0, delta_s
    s_range = arange(delta_s, 4, delta_s)
    sol = odeint(pendant_drop, (x0, z0, phi0), s_range)

    # Ensure there are only two inflection points
    apex_index = argmax(sol[:, 0], axis=0)
    max_index = argmin(sol[apex_index:, 0], axis=0)
    sol = sol[:max_index + apex_index, :]

    # Calculate drop parameter
    volume = 0
    area = 0
    max_drop_radius = 0
    cap_diameter = 0
    worthington_number = 0
    upper_index = 0

    for i in range(len(sol)):
        upper_index = i
        row = sol[i]
        x, z, phi = row

        volume += pi * (x ** 2) * sin(phi) * delta_s
        area += 2 * pi * x * delta_s
        cap_diameter = 2 * x

        if x > max_drop_radius:
            max_drop_radius = x

        worthington_number = bond_number * (volume / (pi * cap_diameter * (max_drop_radius ** 2)))

        if worthington_number > max_worthington_number:
            break

    # Cutoff drop where max_worthington_number = worthington_number, or at the max height
    sol = sol[:upper_index, :]

    # Only keep x, z data
    sol = sol[:, :2]

    # Normalize parameters
    drop_parameters = dict(
        volume=volume / (pi * cap_diameter * (max_drop_radius ** 2)),
        area=area / (pi * cap_diameter * max_drop_radius),
        cap_diameter=cap_diameter / max_drop_radius,
        worthington_number=worthington_number,
        bond_number=bond_number,
    )

    return drop_parameters, sol
