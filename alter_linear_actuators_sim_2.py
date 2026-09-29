import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# PARAMETERS
# =========================================================

Lx, Ly = 1, 1

m = 0.2
k_base = 5.0
k_noise = 0.0
ka0 = 10.0        # chirality
np.random.seed(42)
ka_noise = 2.0
kc = 100.0
gamma = 0.3
kn = 7.0

a = 1.0          # lattice spacing
d = 2.0          # layer separation (IMPORTANT for clarity)

dt = 0.01
T = 30.0
steps = int(T / dt)

# =========================================================
# BUILD LIEB LATTICE
# =========================================================

def build_lieb(Lx, Ly):

    pos = []
    typ = []

    A, B, C = {}, {}, {}

    idx = 0

    for i in range(Lx):
        for j in range(Ly):

            A[(i,j)] = idx
            pos.append([i*a, j*a])
            typ.append("A")
            idx += 1

            B[(i,j)] = idx
            pos.append([i*a + 0.5*a, j*a])
            typ.append("B")
            idx += 1

            C[(i,j)] = idx
            pos.append([i*a, j*a + 0.5*a])
            typ.append("C")
            idx += 1

    Ns = idx

    neigh = [[] for _ in range(Ns)]

    for i in range(Lx):
        for j in range(Ly):

            a0 = A[(i,j)]
            b0 = B[(i,j)]
            c0 = C[(i,j)]

            neigh[a0].append(b0)
            neigh[b0].append(a0)

            neigh[a0].append(c0)
            neigh[c0].append(a0)

            if i > 0:
                bL = B[(i-1,j)]
                neigh[a0].append(bL)
                neigh[bL].append(a0)

            if j > 0:
                cD = C[(i,j-1)]
                neigh[a0].append(cD)
                neigh[cD].append(a0)

    return np.array(pos), typ, neigh


positions, site_type, neighbors = build_lieb(Lx, Ly)
Ns = len(positions)

# =========================================================
# PARAMETERS PER SITE
# =========================================================

k = k_base * np.ones(Ns)

ka = np.zeros(Ns)
for i,t in enumerate(site_type):
    if t == "A":
        ka[i] = 0
    elif t == "B":
        ka[i] = -ka0 + ka_noise * np.random.randn()
    elif t == "C":
        ka[i] = +ka0 + ka_noise * np.random.randn()

# =========================================================
# INITIAL CONDITIONS
# =========================================================

y1 = np.zeros((steps, Ns))
y2 = np.zeros((steps, Ns))
v1 = np.zeros((steps, Ns))
v2 = np.zeros((steps, Ns))

center = Ns // 2
y1[0, center] = 1.0

# =========================================================
# GRAPH LAPLACIAN
# =========================================================

def lap(y):
    out = np.zeros_like(y)
    for i, nb in enumerate(neighbors):
        for j in nb:
            out[i] += y[j] - y[i]
    return out

# =========================================================
# TIME EVOLUTION (simple Euler for clarity)
# =========================================================

for t in range(steps - 1):

    L1 = lap(y1[t])
    L2 = lap(y2[t])

    a1 = (-k*y1[t] + ka*y2[t] + kn*L1 - gamma*v1[t] - kc*y1[t]**3) / m
    a2 = (-k*y2[t] - ka*y1[t] + kn*L2 - gamma*v2[t] - kc*y2[t]**3) / m

    v1[t+1] = v1[t] + dt * a1
    v2[t+1] = v2[t] + dt * a2

    y1[t+1] = y1[t] + dt * v1[t+1]
    y2[t+1] = y2[t] + dt * v2[t+1]

# =========================================================
# 3D PLOT SETUP
# =========================================================

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("3D bilayer Lieb lattice dynamics")

ax.set_xlim(np.min(positions[:,0]), np.max(positions[:,0]))
ax.set_ylim(np.min(positions[:,1]), np.max(positions[:,1]))
ax.set_zlim(-d-3, d+3)

# initial positions
z_top0 = d + y1[0]
z_bot0 = -d + y2[0]

top_scatter = ax.scatter(
    positions[:,0],
    positions[:,1],
    z_top0,
    c='red',
    s=40
)

bot_scatter = ax.scatter(
    positions[:,0],
    positions[:,1],
    z_bot0,
    c='blue',
    s=40
)

# =========================================================
# UPDATE FUNCTION (CORRECTED)
# =========================================================

def update(frame):

    z_top = d + y1[frame]
    z_bot = -d + y2[frame]

    # CORRECT way for 3D scatter update:
    top_scatter._offsets3d = (
        positions[:,0],
        positions[:,1],
        z_top
    )

    bot_scatter._offsets3d = (
        positions[:,0],
        positions[:,1],
        z_bot
    )

    return top_scatter, bot_scatter

# =========================================================
# ANIMATION
# =========================================================

ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    interval=20,
    blit=False   # IMPORTANT for 3D
)

plt.show()

# =========================================================
# ANGULAR MOMENTUM
# =========================================================

J = y1*v2 - y2*v1
J_avg = np.mean(J, axis=0)

#print("Total J (avg):", np.mean(np.sum(J, axis=1)))

# =========================================================
# LIMIT CYCLES + AMPLITUDE & FREQUENCY PER SITE
# =========================================================

def estimate_frequency_from_zerocross(signal, dt):
    """
    Estimate frequency from zero-crossings of a (roughly) periodic signal.
    Uses crossings of the mean level with positive slope.
    Returns (mean_frequency, std_frequency).
    """
    sig = signal - np.mean(signal)
    sign = np.sign(sig)

    # Avoid exact zeros breaking the sign change logic
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i-1]

    # Indices where we cross from negative to positive
    crossings = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0]

    if len(crossings) < 2:
        return np.nan, np.nan

    periods = np.diff(crossings) * dt
    freqs = 1.0 / periods
    return np.mean(freqs), np.std(freqs)

# Use only the last part of the trajectory to approximate the limit cycle
start_idx = steps * 2 // 3   # last third of the time series

fig, axes = plt.subplots(1, Ns, figsize=(5 * Ns, 4), sharex=True, sharey=True)
if Ns == 1:
    axes = [axes]

for s, ax in enumerate(axes):
    # Take post-transient data for site s
    y1s = y1[start_idx:, s]
    y2s = y2[start_idx:, s]

    # Phase-space trajectory (limit cycle)
    ax.plot(y1s, y2s, lw=1.5, color='k')
    ax.scatter(y1s[0],  y2s[0],  color='green', s=30, label='start (post-transient)')
    ax.scatter(y1s[-1], y2s[-1], color='red',   s=30, label='end')

    # Radial amplitude: r(t) = sqrt(y1^2 + y2^2)
    r = np.sqrt(y1s**2 + y2s**2)
    r_mean = np.mean(r)
    r_std  = np.std(r)
    r_amp  = 0.5 * (np.max(r) - np.min(r))   # half peak-to-peak amplitude

    # Frequency from zero-crossings of y1 (post-transient)
    freq, freq_std = estimate_frequency_from_zerocross(y1s, dt)

    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    # Multi-line title with all requested quantities
    ax.set_title(
        "Site {s}\n"
        "ka = {ka:.2f}\n"
        "⟨J⟩ = {J:.3f}\n"
        "Amplitude A ≈ {A:.3f}\n"
        "⟨r⟩ = {rm:.3f}, σ_r = {rs:.3f}\n"
        "f ≈ {f:.3f} ± {fs:.3f}".format(
            s=s,
            J=J_avg[s],
            A=r_amp,
            ka=ka[s],
            rm=r_mean,
            rs=r_std,
            f=freq,
            fs=freq_std
        ),
        fontsize=9
    )

    if s == 0:
        ax.legend(loc='best', fontsize=7)

plt.suptitle("Limit cycles in (y1, y2) with amplitude & frequency per site", y=1.02)
plt.tight_layout()
plt.show()

# # =========================================================
# # PHASE-SPACE LIMIT CYCLES AND ⟨J⟩ PER SITE
# # =========================================================

# # y1, y2 have shape (steps, Ns)
# # J_avg has shape (Ns,)

# fig, axes = plt.subplots(1, Ns, figsize=(4*Ns, 4), sharex=True, sharey=True)

# # Make sure axes is iterable even if Ns = 1
# if Ns == 1:
#     axes = [axes]

# for s, ax in enumerate(axes):
#     # Plot the (y1, y2) phase trajectory for site s
#     ax.plot(y1[:, s], y2[:, s], lw=1.5, color='k')
    
#     # Mark start and end of trajectory
#     ax.scatter(y1[0, s],  y2[0, s],  color='green', s=30, label='start')
#     ax.scatter(y1[-1, s], y2[-1, s], color='red',   s=30, label='end')
    
#     ax.set_xlabel("y1")
#     ax.set_ylabel("y2")
#     ax.set_title(f"Site {s}  ⟨J⟩ = {J_avg[s]:.3f}")
#     ax.grid(True)
#     ax.set_aspect('equal', 'box')
    
#     if s == 0:
#         ax.legend(loc='best', fontsize=8)

# plt.suptitle("Limit cycles in (y1, y2) for each site", y=1.02)
# plt.tight_layout()
# plt.show()