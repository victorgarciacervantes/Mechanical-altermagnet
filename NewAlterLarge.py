import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- numba (optional fast) ---
try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def wrap(fn): return fn
        return wrap

# ============================================================
# GENERAL N x N TESSELLATION (ANY N)
# Perturbation ONLY at square (1,1)
# ============================================================

# -------------------------
# USER PARAMETERS
# -------------------------
Nside = 9         # <-- odd N
a = 1.0
b = 0.18 * a

Gamma = 0.7
k0    = 3.0
kc    = 5.0

k_link = 10.0
ka     = 5.0

dt = 0.01
T  = 20.0
steps = int(T / dt)

# single-site perturbation
amp   = 2
omega = 10.0

# -------------------------
# PATTERN RULES
# -------------------------
def is_missing(i, j):
    # generalized from your 4x4 case
    return (i % 2 == 0) and (j % 2 == 0)

def square_sign(i, j):
    if is_missing(i, j):
        return 0
    if j % 2 == 1:       # odd row
        return +1 if (i % 2 == 0) else 0
    else:                # even row
        return -1 if (i % 2 == 1) else 0

# -------------------------
# BUILD LATTICE
# -------------------------
square_list = []
sq_to_base = {}
centers = []
pos_list = []

for j in range(1, Nside + 1):
    for i in range(1, Nside + 1):
        if is_missing(i, j):
            continue
        base = len(pos_list)
        sq_to_base[(i, j)] = base
        square_list.append((i, j))

        x0, y0 = (i - 1) * a, (j - 1) * a
        centers.append([x0 + 0.5 * a, y0 + 0.5 * a])

        pos_list.append([x0 + b,     y0 + b])       # A
        pos_list.append([x0 + a - b, y0 + a - b])   # B

pos = np.array(pos_list)
centers = np.array(centers)

Nsq = len(square_list)
Nrot = pos.shape[0]

# rest angles
rest_angle = np.zeros(Nrot)
for s, (i, j) in enumerate(square_list):
    c = centers[s]
    base = sq_to_base[(i, j)]
    for k in (0, 1):
        d = c - pos[base + k]
        rest_angle[base + k] = np.arctan2(d[1], d[0])

# -------------------------
# PASSIVE BONDS
# -------------------------
passive_bonds = []
for (i, j) in square_list:
    base = sq_to_base[(i, j)]
    if (i + 1, j) in sq_to_base:
        br = sq_to_base[(i + 1, j)]
        passive_bonds += [(base, br), (base + 1, br + 1)]
    if (i, j + 1) in sq_to_base:
        bu = sq_to_base[(i, j + 1)]
        passive_bonds += [(base, bu), (base + 1, bu + 1)]

pairs = np.array(passive_bonds, dtype=np.int64)

# -------------------------
# ACTIVE BONDS
# -------------------------
active_up, active_down = [], []
active_A, active_B, active_sgn = [], [], []

for (i, j) in square_list:
    sgn = square_sign(i, j)
    if sgn == 0:
        continue
    base = sq_to_base[(i, j)]
    if sgn > 0:
        active_up.append((base, base + 1))
    else:
        active_down.append((base, base + 1))
    active_A.append(base)
    active_B.append(base + 1)
    active_sgn.append(float(sgn))

active_A = np.array(active_A, dtype=np.int64)
active_B = np.array(active_B, dtype=np.int64)
active_sgn = np.array(active_sgn)

# -------------------------
# DYNAMICS
# -------------------------
@njit(cache=True)
def accel(phi, v):
    a = np.zeros_like(phi)

    for i in range(phi.shape[0]):
        a[i] = -Gamma*v[i] - k0*phi[i] - kc*phi[i]**3

    for i, j in pairs:
        d = phi[i] - phi[j]
        a[i] += -k_link*d
        a[j] += +k_link*d

    for k in range(active_A.shape[0]):
        ia, ib = active_A[k], active_B[k]
        sgn = active_sgn[k]
        a[ia] += -(ka*sgn)*phi[ib]
        a[ib] += +(ka*sgn)*phi[ia]

    return a

def integrate(phi0, v0):
    phi = np.zeros((steps, Nrot))
    v   = np.zeros((steps, Nrot))
    phi[0], v[0] = phi0, v0

    a0 = accel(phi[0], v[0])
    for t in range(steps - 1):
        v_half = v[t] + 0.5*dt*a0
        phi[t+1] = phi[t] + dt*v_half
        a1 = accel(phi[t+1], v_half)
        v[t+1] = v_half + 0.5*dt*a1
        a0 = a1
    return phi, v

# -------------------------
# INITIAL CONDITIONS
# 
# -------------------------
phi0 = np.zeros(Nrot)
v0   = np.zeros(Nrot)

# perturb passive sites at (1, 2n+1): j = 1,3,5,...
# for j in range(1, Nside + 1, 2):
#     coord = (1, j)
#     if coord in sq_to_base and square_sign(1, j) == 0:
#         base = sq_to_base[coord]
#         phi0[base]     = amp
#         phi0[base + 1] = 0.0    
#         v0[base]       = 0.0
#         v0[base + 1]   = omega * amp

# === MODIFIED: Perturb only the single passive site at the bottom-left corner (1, 1) ===
coord = (1, 1)

if coord in sq_to_base and square_sign(1, 1) == 0:
    base = sq_to_base[coord]
    phi0[base]     = amp
    phi0[base + 1] = amp # 0.0 this is original
    v0[base]       = omega * amp # 0.0 this is original
    v0[base + 1]   = omega * amp
else:
    print(f"Warning: Coordinate {coord} is either not in the registry or is not a passive site.")

phi, v = integrate(phi0, v0)

# -------------------------
# ANIMATION
# -------------------------
sample = 3
phi_s = phi[::sample]
phi_disp = phi_s + rest_angle
t_s = np.arange(phi_s.shape[0])*dt*sample

fig, ax = plt.subplots(figsize=(8.5,8.5))
ax.set_aspect("equal")
ax.set_title(
    f"N={Nside} lattice | \n"
    "passive=green | active +ka=red | active -ka=blue"
)

# square outlines
def draw_square(x0,y0):
    sq = np.array([[x0,y0],[x0+a,y0],[x0+a,y0+a],[x0,y0+a],[x0,y0]])
    ax.plot(sq[:,0],sq[:,1],color="black",lw=2)

for j in range(Nside):
    for i in range(Nside):
        draw_square(i*a,j*a)

ax.scatter(pos[:,0],pos[:,1],s=12,color="black")

(passive_lines,) = ax.plot([],[],color="green",lw=1.4)
(active_up_lines,) = ax.plot([],[],color="red",lw=2.2)
(active_down_lines,) = ax.plot([],[],color="blue",lw=2.2)

bars = [ax.plot([],[],lw=2)[0] for _ in range(Nrot)]

Q = ax.quiver(
    centers[:,0], centers[:,1],
    np.zeros(Nsq), np.zeros(Nsq),
    angles="xy", scale_units="xy", scale=1,
    width=0.009, pivot="mid"
)

L = 0.22*a
spin_scale = 0.18*a

def bar_pts(c,ang):
    d = np.array([np.cos(ang),np.sin(ang)])*L
    return c-d, c+d

def set_lines(line,bonds):
    xs,ys=[],[]
    for i,j in bonds:
        xs += [pos[i,0],pos[j,0],np.nan]
        ys += [pos[i,1],pos[j,1],np.nan]
    line.set_data(xs,ys)

def update(f):
    for i in range(Nrot):
        p1,p2 = bar_pts(pos[i],phi_disp[f,i])
        bars[i].set_data([p1[0],p2[0]],[p1[1],p2[1]])

    set_lines(passive_lines,passive_bonds)
    set_lines(active_up_lines,active_up)
    set_lines(active_down_lines,active_down)

    U,V = np.zeros(Nsq),np.zeros(Nsq)
    for s,(i,j) in enumerate(square_list):
        base = sq_to_base[(i,j)]
        U[s] = spin_scale*phi_s[f,base]
        V[s] = spin_scale*phi_s[f,base+1]
    Q.set_UVC(U,V)

    return (*bars, passive_lines, active_up_lines, active_down_lines, Q)

ani = FuncAnimation(fig, update, frames=len(phi_s), interval=30)
plt.show()

# ani.save(f"N{Nside}_perturb_only_11.mp4",
#          writer="ffmpeg",
#          fps=int(1/(dt*sample)),
#          dpi=150)

# print("Saved:","N{Nside}_perturb_only_11.mp4")

# Victor code to store and plot 

# Initialize empty lists
U_history = []
V_history = []

# Process each time step directly
for frame in range(len(phi_s)):
    dev = phi_s[frame]

    U = np.zeros(Nsq, dtype=np.float64)
    V = np.zeros(Nsq, dtype=np.float64)
    for s, (i, j) in enumerate(square_list):
        base = sq_to_base[(i, j)]
        U[s] = spin_scale * dev[base]
        V[s] = spin_scale * dev[base + 1]

    U_history.append(U.copy())
    V_history.append(V.copy())

# Optionally convert to arrays for easier downstream processing
U_history = np.array(U_history)
V_history = np.array(V_history)
dU_dt = np.gradient(U_history, axis=0)
dV_dt = np.gradient(V_history, axis=0)

Z_history = np.sqrt(U_history**2 + V_history**2)

# classify squares by sign
sgn_list = np.array([square_sign(i, j) for (i, j) in square_list])
idx_plus  = np.where(sgn_list > 0)[0]
idx_minus = np.where(sgn_list < 0)[0]
idx_pass  = np.where(sgn_list == 0)[0]

# time array (already computed above as t_s)
t = t_s

# # prepare figure with 3 subplots
# fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
# cats = [
#     ("Active +", idx_plus, "red"),
#     ("Active -", idx_minus, "blue"),
#     ("Passive",  idx_pass,  "green"),
# ]

# for ax, (title, idxs, color) in zip(axes, cats):
#     ax.set_title(f"{title} (n={len(idxs)})")
#     if len(idxs) == 0:
#         ax.text(0.5, 0.5, "no sites", transform=ax.transAxes, ha="center")
#         continue

#     # plot individual site traces (faint)
#     ax.plot(t[:, None], Z_history[:, idxs], color=color, alpha=0.25, lw=0.8)

#     # plot mean trace
#     mean_trace = Z_history[:, idxs].mean(axis=1)
#     ax.plot(t, mean_trace, color=color, lw=2.2, label="mean")

#     ax.set_ylabel("Z (U^2+V^2)")
#     ax.grid(alpha=0.3)
#     ax.legend()

# axes[-1].set_xlabel("time")
# plt.tight_layout()

# plot U and V for passive sites on the right side (i == Nside), one subplot per site stacked vertically
right_pass_idxs = [s for s, (i, j) in enumerate(square_list) if (i == Nside and square_sign(i, j) == 0)]
n_sites = len(right_pass_idxs)

if n_sites == 0:
    print("No passive sites on the right side of the lattice.")
else:
    fig, axes = plt.subplots(n_sites, 1, sharex=True, figsize=(8, 2.5 * n_sites))
    if n_sites == 1:
        axes = [axes]

    for ax, s in zip(axes, right_pass_idxs):
        i, j = square_list[s]
        ax.plot(t, U_history[:, s], label="U", color="C0", lw=1.5)
        ax.plot(t, V_history[:, s], label="V", color="C1", lw=1.5)
        ax.set_ylabel("amplitude")
        ax.set_title(f"Passive site at (i={i}, j={j})  index={s}")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("time")
    plt.tight_layout()

fig, axes = plt.subplots(n_sites, 1, sharex=True, figsize=(8, 2.5 * n_sites))
for ax, s in zip(axes, right_pass_idxs):
    i, j = square_list[s]
    ax.plot(t, dU_dt[:, s], label="dU/dt", color="C2", lw=1.5)
    ax.plot(t, dV_dt[:, s], label="dV/dt", color="C3", lw=1.5)
    ax.set_ylabel("angular velocity")
    ax.set_title(f"Passive site at (i={i}, j={j})  index={s}")
    ax.grid(alpha=0.3)
    ax.legend()   

axes[-1].set_xlabel("time")
plt.tight_layout() 

if n_sites == 0:
    print("No passive sites on the right side of the lattice.")
else:
    fig, axes = plt.subplots(n_sites, 1, sharex=True, figsize=(8, 2.5 * n_sites))
    if n_sites == 1:
        axes = [axes]
    for ax, s in zip(axes, right_pass_idxs):
        i, j = square_list[s]
        J = U_history[:, s] * dV_dt[:, s] - V_history[:, s] * dU_dt[:, s]
        J_avg = np.mean(J)
        ax.axhline(y=J_avg, color="darkred", linestyle="--", linewidth=1.5, 
                   label=f"Average $\\langle J \\rangle = {J_avg:.3e}$")
        ax.plot(t, J, label="J", color="C2", lw=1.5)
        ax.set_ylabel("J")
        ax.set_title(f"Passive site at (i={i}, j={j})  index={s}")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("time")
    plt.tight_layout()

    plt.show()

    # 1. Extract passive site indices specifically located along the top boundary
top_pass_idxs = [s for s, (i, j) in enumerate(square_list) if (j == Nside and square_sign(i, j) == 0)]
n_sites = len(top_pass_idxs)

if n_sites == 0:
    print(f"No passive sites found along the top boundary (j = {Nside}) of the lattice.")
else:
    # 2. Initialize stacked subplot layout matching the top-site footprint counts
    fig, axes = plt.subplots(n_sites, 1, sharex=True, figsize=(8, 2.8 * n_sites))
    
    # Standardize to iterable list if only a single subplot layout exists
    if n_sites == 1:
        axes = [axes]
        
    for ax, s in zip(axes, top_pass_idxs):
        i, j = square_list[s]
        
        # 3. Calculate instantaneous tracking and average angular momentum values
        J = U_history[:, s] * dV_dt[:, s] - V_history[:, s] * dU_dt[:, s]
        J_avg = np.mean(J)
        
        # Plot continuous instantaneous angular momentum tracking line
        ax.plot(t, J, label="Instantaneous $J$", color="C2", lw=1.5)
        
        # Overlay horizontal indicator representing the computed mean reference frame
        ax.axhline(y=J_avg, color="darkred", linestyle="--", linewidth=1.5, 
                   label=f"Average $\\langle J \\rangle = {J_avg:.3e}$")
        
        # 4. Polish labels, canvas grid fields, and legends per panel
        ax.set_ylabel("J", fontsize=10)
        ax.set_title(f"Top Boundary Passive Site at ($i$={i}, $j$={j}) — Index={s}", 
                     loc="left", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9)
        
    # Enforce global standalone horizontal labeling on bottom-most plot panel
    axes[-1].set_xlabel("Time", fontsize=11)
    
    plt.tight_layout()
    plt.show()


# ============================================================
# STATIC CHIRALITY DIAGRAM WITH LINKS & EDGE CURRENT SIGNS
# ============================================================

snapshot_frame = -1  # Snapshot at the final time frame

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
# ax.set_title(
#     f"N={Nside} Lieb Lattice Structure | State & Boundary Current Signs $\\langle J \\rangle$\n"
#     "Links: Passive=Green | Active +ka=Red | Active -ka=Blue",
#     fontsize=12, fontweight="bold", pad=15
# )

# 1. Helper function to plot interconnecting structural links (bonds)
def plot_links(ax, bonds, color, lw):
    xs, ys = [], []
    for i, j in bonds:
        xs += [pos[i, 0], pos[j, 0], np.nan]
        ys += [pos[i, 1], pos[j, 1], np.nan]
    ax.plot(xs, ys, color=color, lw=lw, zorder=2)

# 2. Draw background grid outlines using your 1-indexed spatial system
for j in range(1, Nside + 1):
    for i in range(1, Nside + 1):
        if is_missing(i, j):
            continue
        x0, y0 = (i - 1) * a, (j - 1) * a
        sq = np.array([[x0, y0], [x0+a, y0], [x0+a, y0+a], [x0, y0+a], [x0, y0]])
        ax.plot(sq[:, 0], sq[:, 1], color="black", lw=1.0, alpha=0.25, zorder=1)

# 3. Plot all structural interconnect links (Bonds) natively
#plot_links(ax, passive_bonds, color="green", lw=1.6)
plot_links(ax, active_up, color="red", lw=2.4)
plot_links(ax, active_down, color="blue", lw=2.4)

# 4. Draw individual rotor bars (sub-elements A and B) oriented by phi_disp state
for s, (i, j) in enumerate(square_list):
    base = sq_to_base[(i, j)]
    for k in (0, 1):
        rotor_idx = base + k
        center_pos = pos[rotor_idx]
        current_angle = phi_disp[snapshot_frame, rotor_idx]
        
        # Calculate rod line coordinates using your custom layout function
        p1, p2 = bar_pts(center_pos, current_angle)
        #ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", lw=1.2, zorder=3)

# Place pivot node points
ax.scatter(pos[:, 0], pos[:, 1], s=12, color="black", zorder=4)

# 5. Evaluate and annotate current direction signs on all 4 outer borders
for s, (i, j) in enumerate(square_list):
    sgn = square_sign(i, j)
    
    # Identify if a passive unit falls along any outer boundary coordinate
    is_border = (i == 1 or i == Nside or j == 1 or j == Nside)
    
    if sgn == 0 and is_border:
        # Pull instantaneous values to calculate net transport currents
        J = U_history[:, s] * dV_dt[:, s] - V_history[:, s] * dU_dt[:, s]
        J_avg = np.mean(J)
        
        # Sort sign profiles cleanly
        if J_avg > 1e-7:
            sign_text = "+"
            sign_color = "yellow"
        elif J_avg < -1e-7:
            sign_text = "−"
            sign_color = "green"
        else:
            sign_text = "0"
            sign_color = "gray"
            
        # Center coordinates inside the respective unit cell box
        sq_center_x = (i - 1 + 0.5) * a
        sq_center_y = (j - 1 + 0.5) * a
        
        # Render tag overlay with solid circular bounding background mask
        ax.text(
            sq_center_x, sq_center_y, 
            sign_text, 
            color=sign_color, 
            fontsize=30,           # <-- Scale up the font size
            fontweight="bold", # <-- Use a very thick font face
            ha="center", va="center", zorder=5,
            bbox=dict(
                boxstyle="circle,pad=0.1", 
                facecolor="white",  # Solid backdrop forces it to stand out
                edgecolor=sign_color, # Match border line to the sign color
                linewidth=1.5,
                alpha=0.95
            )
        )
        # ============================================================
        # HIGHLIGHT INITIAL PERTURBATION SITE (1,1) BASED ON OMEGA SIGN
        # ============================================================
        # Determine initial kick chirality from the sign of omega
        if omega > 0:
            perturb_text = "+ init"  # Counter-clockwise kick
            perturb_color = "darkred"
        elif omega < 0:
            perturb_text = "− init"  # Clockwise kick
            perturb_color = "darkblue"
        else:
            perturb_text = "Init"
            perturb_color = "purple"

        # Center coordinates for the (1,1) square cell envelope
        init_x = (1 - 1 + 0.5) * a
        init_y = (1 - 1 + 0.5) * a

        # Overlay a highlighted indicator badge
        ax.text(
            init_x, init_y, 
            perturb_text, 
            color="white", 
            fontsize=14, 
            fontweight="bold",
            ha="center", 
            va="center",
            zorder=6,  # Sits on top of everything else
            bbox=dict(
                boxstyle="round,pad=0.3", 
                facecolor=perturb_color, 
                edgecolor="gold",    # Golden border to explicitly emphasize the source node
                linewidth=2.0,
                alpha=1.0
            )
        )

# Polish canvas margins
ax.set_xlim(-0.5 * a, Nside * a + 0.5 * a)
ax.set_ylim(-0.5 * a, Nside * a + 0.5 * a)
ax.grid(True, linestyle=":", alpha=0.15)
plt.tight_layout()
#plt.savefig("lieb_sml_neg.pdf", format="pdf", bbox_inches="tight")

# ============================================================
# STATIC CHIRALITY DIAGRAM — COMPACTED PASSIVE BORDER RING
# Assumes Nside is odd. Passive sites sit at odd (i,j) indices.
# Compacted ring size: M = (Nside+1)//2 cells per side.
# ============================================================

snapshot_frame = -1

def is_passive_square(i, j):
    return square_sign(i, j) == 0

def side_of(i, j):
    """Assign a border cell to exactly one side (corners go to bottom/top)."""
    if j == 1:     return "bottom"
    if j == Nside: return "top"
    if i == Nside: return "right"
    return "left"

# ----------------------------------------------------------------
# M = number of passive cells per side of the compacted ring
# For odd Nside, every border side has exactly M passive cells.
# ----------------------------------------------------------------
M = (Nside + 1) // 2
cell = a  # keep the same cell size

# ----------------------------------------------------------------
# Build new_centers: map (i,j) -> (cx, cy) in the compacted ring.
# Walk each side in its natural drawing direction and assign slots 0..M-1.
# ----------------------------------------------------------------
new_centers = {}

# Bottom: i = 1,3,5,...,Nside  (left to right),  j=1
for slot, i in enumerate(range(1, Nside + 1, 2)):
    cx = (slot + 0.5) * cell
    cy = 0.5 * cell
    new_centers[(i, 1)] = (cx, cy)

# Right: j = 1,3,5,...,Nside  (bottom to top),  i=Nside
for slot, j in enumerate(range(1, Nside + 1, 2)):
    cx = (M - 0.5) * cell
    cy = (slot + 0.5) * cell
    new_centers[(Nside, j)] = (cx, cy)

# Top: i = Nside,Nside-2,...,1  (right to left),  j=Nside
for slot, i in enumerate(range(Nside, 0, -2)):
    cx = (M - slot - 0.5) * cell
    cy = (M - 0.5) * cell
    new_centers[(i, Nside)] = (cx, cy)

# Left: j = Nside,Nside-2,...,1  (top to bottom),  i=1
for slot, j in enumerate(range(Nside, 0, -2)):
    cx = 0.5 * cell
    cy = (M - slot - 0.5) * cell
    new_centers[(1, j)] = (cx, cy)

# Corners appear in two sides above — the second write wins,
# but both sides produce the same (cx,cy) for corners so it's fine.

# ----------------------------------------------------------------
# Figure sized exactly to the compacted ring
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

# 1. Draw compacted passive cell outlines
for (i, j), (cx, cy) in new_centers.items():
    x0, y0 = cx - 0.5 * cell, cy - 0.5 * cell
    sq = np.array([[x0, y0], [x0+cell, y0], [x0+cell, y0+cell],
                   [x0, y0+cell], [x0, y0]])
    ax.plot(sq[:, 0], sq[:, 1], color="black", lw=1.4, alpha=0.7, zorder=2)

# 2. Chirality signs
for s_idx, (i, j) in enumerate(square_list):
    if (i, j) not in new_centers:
        continue

    J = U_history[:, s_idx] * dV_dt[:, s_idx] - V_history[:, s_idx] * dU_dt[:, s_idx]
    J_avg = np.mean(J)

    if J_avg > 1e-7:
        sign_text, sign_color = "+", "red"
    elif J_avg < -1e-7:
        sign_text, sign_color = "−", "blue"
    else:
        sign_text, sign_color = "0", "gray"

    cx, cy = new_centers[(i, j)]
    ax.text(
        cx, cy, sign_text,
        color=sign_color, fontsize=40, fontweight="bold",
        ha="center", va="center", zorder=4,
        bbox=dict(boxstyle="circle,pad=0.1", facecolor="white",
                  edgecolor=sign_color, linewidth=1.5, alpha=0.95)
    )

# # 3. Init perturbation badge at the new position of cell (1,1)
# if omega > 0:
#     perturb_text, perturb_color = "+ init", "darkred"
# elif omega < 0:
#     perturb_text, perturb_color = "− init", "darkblue"
# else:
perturb_text, perturb_color = "Init", "purple"

bx, by = new_centers[(1, 1)]
ax.text(
    bx+0.4, by+0.4, perturb_text, color="white",
    fontsize=15, fontweight="bold", ha="center", va="center", zorder=5,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=perturb_color,
              edgecolor="gold", linewidth=2.0, alpha=1.0)
)

# 4. Canvas — exactly fits the M×M compacted ring
ax.set_xlim(0, M * cell)
ax.set_ylim(0, M * cell)
plt.tight_layout()
#plt.savefig("lieb_sml_gen_compacted.pdf", format="pdf", bbox_inches="tight")
plt.show()
