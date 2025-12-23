import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Projectile Motion Simulator
# With and Without Air Resistance
# ----------------------------

# ---- Parameters you can change ----
g = 9.81                    # m/s^2
theta_deg = 45              # launch angle in degrees
v0 = 20.0                   # initial speed (m/s)
y0 = 0.0                    # initial height (m)

# Drag options:
use_quadratic_drag = True   # True: quadratic drag, False: linear drag

# Physical-ish defaults (good enough for a project)
m = 0.145                   # mass (kg) ~ baseball
rho = 1.225                 # air density (kg/m^3)
Cd = 0.47                   # drag coefficient (~sphere)
r = 0.0366                  # radius (m) ~ baseball
A = np.pi * r**2            # cross-sectional area (m^2)

# Quadratic drag coefficient: a_drag = -(k/m)*|v|*v
k_quad = 0.5 * rho * Cd * A

# Linear drag coefficient: a_drag = -(b/m)*v
b_lin = 0.02                # kg/s (tweak this if using linear drag)

dt = 0.001                  # time step (s)
t_max = 10.0                # max sim time (s)

# ---- Helper functions ----
def acceleration_no_drag(vx, vy):
    return 0.0, -g

def acceleration_with_drag(vx, vy):
    if use_quadratic_drag:
        v = np.hypot(vx, vy)
        ax = -(k_quad / m) * v * vx
        ay = -g - (k_quad / m) * v * vy
    else:
        ax = -(b_lin / m) * vx
        ay = -g - (b_lin / m) * vy
    return ax, ay

def rk4_step(state, dt, accel_func):
    # state = [x, y, vx, vy]
    def f(s):
        x, y, vx, vy = s
        ax, ay = accel_func(vx, vy)
        return np.array([vx, vy, ax, ay], dtype=float)

    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate(accel_func):
    theta = np.deg2rad(theta_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    state = np.array([0.0, y0, vx0, vy0], dtype=float)

    t_vals = [0.0]
    x_vals = [state[0]]
    y_vals = [state[1]]
    vx_vals = [state[2]]
    vy_vals = [state[3]]

    t = 0.0
    for _ in range(int(t_max / dt)):
        state = rk4_step(state, dt, accel_func)
        t += dt

        # Stop when projectile hits the ground (y <= 0) after launch
        if state[1] < 0 and t > dt:
            # Linear interpolation to estimate landing point nicely
            x_prev, y_prev = x_vals[-1], y_vals[-1]
            x_new, y_new = state[0], state[1]
            frac = y_prev / (y_prev - y_new) if (y_prev - y_new) != 0 else 1.0
            x_land = x_prev + frac * (x_new - x_prev)
            t_land = t_vals[-1] + frac * (t - t_vals[-1])

            # store landing point
            t_vals.append(t_land)
            x_vals.append(x_land)
            y_vals.append(0.0)
            vx_vals.append(state[2])
            vy_vals.append(state[3])
            break

        t_vals.append(t)
        x_vals.append(state[0])
        y_vals.append(state[1])
        vx_vals.append(state[2])
        vy_vals.append(state[3])

    return (np.array(t_vals), np.array(x_vals), np.array(y_vals),
            np.array(vx_vals), np.array(vy_vals))

# ---- Run simulations ----
t_n, x_n, y_n, vx_n, vy_n = simulate(acceleration_no_drag)
t_d, x_d, y_d, vx_d, vy_d = simulate(acceleration_with_drag)

# ---- Quick stats ----
range_n = x_n[-1]
range_d = x_d[-1]
maxh_n = y_n.max()
maxh_d = y_d.max()
time_n = t_n[-1]
time_d = t_d[-1]

print("No drag:")
print(f"  Range: {range_n:.2f} m | Max height: {maxh_n:.2f} m | Time: {time_n:.2f} s")
print("With drag:")
print(f"  Range: {range_d:.2f} m | Max height: {maxh_d:.2f} m | Time: {time_d:.2f} s")

# ---- Plots ----
# 1) Trajectory
plt.figure()
plt.plot(x_n, y_n, label="No air resistance")
plt.plot(x_d, y_d, label="With air resistance")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Projectile Trajectory")
plt.legend()
plt.grid(True)
plt.show()

# 2) Speed vs time
speed_n = np.hypot(vx_n, vy_n)
speed_d = np.hypot(vx_d, vy_d)

plt.figure()
plt.plot(t_n, speed_n, label="No air resistance")
plt.plot(t_d, speed_d, label="With air resistance")
plt.xlabel("time (s)")
plt.ylabel("speed (m/s)")
plt.title("Speed vs Time")
plt.legend()
plt.grid(True)
plt.show()

# 3) y(t) comparison (optional but nice)
plt.figure()
plt.plot(t_n, y_n, label="No air resistance")
plt.plot(t_d, y_d, label="With air resistance")
plt.xlabel("time (s)")
plt.ylabel("height y (m)")
plt.title("Height vs Time")
plt.legend()
plt.grid(True)
plt.show()
