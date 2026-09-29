import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# PARAMETERS
# =========================================================

m = 0.2
k_base = 5.0
#ka0 = 10.0      # chirality magnitude
ka = [0, 10, 9, -8, -10]
kc = 100.0      # cubic nonlinearity
gamma = 0.3     # damping
kn = 5.0        # coupling strength

a = 1.0         # spacing (for plotting)
d = 2.0         # layer separation

dt = 0.01
T = 50.0
steps = int(T / dt)

# =========================================================
# GEOMETRY: CROSS WITH ONE PASSIVE CENTER SITE
# =========================================================
# Indexing:
# 0: center (passive, ka=0)
# 1: top    (ka=+ka0)
# 2: bottom (ka=+ka0)
# 3: left   (ka=-ka0)
# 4: right  (ka=-ka0)

Ns = 5

positions = np.array([
    [0.0,   0.0],   # center
    [0.0,   a],     # top
    [0.0,  -a],     # bottom
    [-a,   0.0],    # left
    [ a,   0.0],    # right
])

site_names = [
    "Center (ka=0)",
    "Top (+ka)",
    "Bottom (+ka)",
    "Left (−ka)",
    "Right (−ka)"
]

# Neighbors: only central site connected to the four outer sites
neighbors = [[] for _ in range(Ns)]
for s in [1, 2, 3, 4]:
    neighbors[0].append(s)
    neighbors[s].append(0)

# Indices per type (for plotting / animation)
idx_passive = np.array([0])
idx_pos     = np.array([1, 2])
idx_neg     = np.array([3, 4])

# =========================================================
# SITE PARAMETERS
# =========================================================

k = k_base * np.ones(Ns)


# =========================================================
# INITIAL CONDITIONS
# =========================================================

y1 = np.zeros((steps, Ns))
y2 = np.zeros((steps, Ns))
v1 = np.zeros((steps, Ns))
v2 = np.zeros((steps, Ns))

# Excite central site in layer 1
y1[0, 0] = 1.0

# =========================================================
# GRAPH LAPLACIAN OPERATOR
# =========================================================

def lap(y):
    out = np.zeros_like(y)
    for i, nb in enumerate(neighbors):
        for j in nb:
            out[i] += y[j] - y[i]
    return out

# =========================================================
# TIME EVOLUTION (Euler)
# =========================================================

for t in range(steps - 1):
    L1 = lap(y1[t])
    L2 = lap(y2[t])

    a1 = (-k * y1[t] + ka * y2[t] + kn * L1 - gamma * v1[t] - kc * y1[t]**3) / m
    a2 = (-k * y2[t] - ka * y1[t] + kn * L2 - gamma * v2[t] - kc * y2[t]**3) / m

    v1[t + 1] = v1[t] + dt * a1
    v2[t + 1] = v2[t] + dt * a2

    y1[t + 1] = y1[t] + dt * v1[t + 1]
    y2[t + 1] = y2[t] + dt * v2[t + 1]

# =========================================================
# ANGULAR MOMENTUM
# =========================================================

J = y1 * v2 - y2 * v1   # shape (steps, Ns)

# =========================================================
# 3D ANIMATION OF BILAYER CROSS
# =========================================================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("3D bilayer cross dynamics (passive / +ka / −ka)")

ax.set_xlim(np.min(positions[:, 0]) - 1, np.max(positions[:, 0]) + 1)
ax.set_ylim(np.min(positions[:, 1]) - 1, np.max(positions[:, 1]) + 1)
ax.set_zlim(-d - 3, d + 3)

# initial z positions
z_top0 = d + y1[0]
z_bot0 = -d + y2[0]

# Top layer scatters
top_scatter_passive = ax.scatter(
    positions[idx_passive, 0],
    positions[idx_passive, 1],
    z_top0[idx_passive],
    c='gray',
    s=80,
    marker='o',
    label='passive (top)'
)
top_scatter_pos = ax.scatter(
    positions[idx_pos, 0],
    positions[idx_pos, 1],
    z_top0[idx_pos],
    c='red',
    s=80,
    marker='^',
    label='+ka (top)'
)
top_scatter_neg = ax.scatter(
    positions[idx_neg, 0],
    positions[idx_neg, 1],
    z_top0[idx_neg],
    c='blue',
    s=80,
    marker='s',
    label='−ka (top)'
)

# Bottom layer scatters
bot_scatter_passive = ax.scatter(
    positions[idx_passive, 0],
    positions[idx_passive, 1],
    z_bot0[idx_passive],
    c='lightgray',
    s=80,
    marker='o',
    label='passive (bottom)'
)
bot_scatter_pos = ax.scatter(
    positions[idx_pos, 0],
    positions[idx_pos, 1],
    z_bot0[idx_pos],
    c='pink',
    s=80,
    marker='^',
    label='+ka (bottom)'
)
bot_scatter_neg = ax.scatter(
    positions[idx_neg, 0],
    positions[idx_neg, 1],
    z_bot0[idx_neg],
    c='lightblue',
    s=80,
    marker='s',
    label='−ka (bottom)'
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("layer / displacement")
ax.legend(loc='upper right')

def update(frame):
    z_top = d + y1[frame]
    z_bot = -d + y2[frame]

    # Update offsets for each group
    top_scatter_passive._offsets3d = (
        positions[idx_passive, 0],
        positions[idx_passive, 1],
        z_top[idx_passive]
    )
    top_scatter_pos._offsets3d = (
        positions[idx_pos, 0],
        positions[idx_pos, 1],
        z_top[idx_pos]
    )
    top_scatter_neg._offsets3d = (
        positions[idx_neg, 0],
        positions[idx_neg, 1],
        z_top[idx_neg]
    )

    bot_scatter_passive._offsets3d = (
        positions[idx_passive, 0],
        positions[idx_passive, 1],
        z_bot[idx_passive]
    )
    bot_scatter_pos._offsets3d = (
        positions[idx_pos, 0],
        positions[idx_pos, 1],
        z_bot[idx_pos]
    )
    bot_scatter_neg._offsets3d = (
        positions[idx_neg, 0],
        positions[idx_neg, 1],
        z_bot[idx_neg]
    )

    return (top_scatter_passive, top_scatter_pos, top_scatter_neg,
            bot_scatter_passive, bot_scatter_pos, bot_scatter_neg)

ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=20,
    blit=False
)

plt.show()

# =========================================================
# FREQUENCY ESTIMATION HELPER
# =========================================================

def estimate_frequency_from_zerocross(signal, dt):
    """
    Estimate frequency from zero-crossings (mean-subtracted signal).
    We look for crossings from negative to positive.
    Returns (mean_frequency, std_frequency). If not enough crossings,
    returns (nan, nan).
    """
    sig = signal - np.mean(signal)
    sign = np.sign(sig)

    # Avoid exact zeros causing ambiguous sign changes
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]

    crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]

    if len(crossings) < 2:
        return np.nan, np.nan

    periods = np.diff(crossings) * dt
    freqs = 1.0 / periods
    return np.mean(freqs), np.std(freqs)

# =========================================================
# LIMIT CYCLES + AMPLITUDE & FREQUENCY PER SITE
# =========================================================

# Use last part of the time series to approximate steady limit cycles
start_idx = steps * 2 // 3

J_avg = np.mean(J[start_idx:], axis=0)

fig2, axes = plt.subplots(1, Ns, figsize=(4 * Ns, 4), sharex=True, sharey=True)
if Ns == 1:
    axes = [axes]

for s, ax in enumerate(axes):
    # Post-transient data
    y1s = y1[start_idx:, s]
    y2s = y2[start_idx:, s]

    # Phase-space trajectory
    ax.plot(y1s, y2s, lw=1.5, color='k')
    ax.scatter(y1s[0],  y2s[0],  color='green', s=30, label='start (post-transient)')
    ax.scatter(y1s[-1], y2s[-1], color='red',   s=30, label='end')

    # Radial amplitude quantities
    r = np.sqrt(y1s**2 + y2s**2)
    r_mean = np.mean(r)
    r_std  = np.std(r)
    r_amp  = 0.5 * (np.max(r) - np.min(r))  # half peak-to-peak

    # Frequency from zero-crossings of y1 (post-transient)
    freq, freq_std = estimate_frequency_from_zerocross(y1s, dt)

    # Title with all requested quantities
    ax.set_title(
        f"{site_names[s]}\n"
        f"⟨J⟩ = {J_avg[s]:.3f}\n"
        f"A ≈ {r_amp:.3f}\n"
        f"⟨r⟩ = {r_mean:.3f}, σ_r = {r_std:.3f}\n"
        f"f ≈ {freq:.3f} ± {freq_std:.3f}",
        fontsize=9
    )

    ax.set_xlabel("y1")
    if s == 0:
        ax.set_ylabel("y2")
        ax.legend(loc='best', fontsize=7)

    ax.grid(True)
    ax.set_aspect('equal', 'box')

plt.suptitle("Limit cycles (y1, y2) with amplitude, frequency, and ⟨J⟩ per site", y=1.03)
plt.tight_layout()
plt.show()