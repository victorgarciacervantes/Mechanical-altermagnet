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
# GENERAL NxM TESSELLATION (ANY Nx,Ny) + BC OPTIONS
#
# BC options for passive neighbor coupling graph:
#   BC = "open"     : only internal neighbor bonds (no wrap)
#   BC = "periodic" : wrap i=Nx -> 1 and j=Ny -> 1 (bonds only if both squares exist)
#   BC = "fixed"    : like "open" + pin boundary squares to ground (phi=0) via k_fix
#
# Missing squares rule (generalized from your 4x4 example):
#   remove all (even, even) squares: i even AND j even -> no rotors
#
# Activity pattern:
#   Row j odd  : i odd -> passive, i even -> NR + (red)
#   Row j even : i odd -> NR - (blue), i even -> passive
#
# Pseudo-spin per occupied square:
#   R = phi_A + i phi_B -> quiver (U,V)=(phi_A,phi_B) at square center
#
# IC: perturb ONLY square (1,1) (if it exists)
# ============================================================

# -------------------------
# USER PARAMETERS
# -------------------------
Nx = 11             # number of squares in x
Ny = 1             # number of squares in y
BC = "open"            # "open", "periodic", or "fixed"

a = 1.0
b = 0.18 * a            # 0 < b < a/2

Gamma = 1
k0    = 15.0
kc    = 80.0

k_link = 20           # passive neighbor coupling
ks= 80            # passive cross coupling
ka = 15.0           # active magnitude

k_fix  = 100.0          # only used when BC="fixed": boundary pin stiffness

dt = 0.01
T  = 10.0
steps = int(T / dt)

# single-site perturbation 
amp   = 3.0
omega = -5

# -------------------------
# PATTERN RULES
# -------------------------
def is_missing(i, j):
    # remove all even-even squares
    return (i % 2 == 0) and (j % 2 == 0)

def square_sign(i, j):
    """
    Returns:
      0   -> passive
      +1  -> nonreciprocal + (red)
      -1  -> nonreciprocal - (blue)
    """
    if is_missing(i, j):
        return 0  # unused; no rotors anyway

    if j % 2 == 1:       # odd row
        return +1 if (i % 2 == 0) else 0
    else:                # even row
        return -1 if (i % 2 == 1) else 0

# -------------------------
# BUILD OCCUPIED SQUARES, ROTORS, CENTERS
# -------------------------
square_list = []             # occupied (i,j)
sq_to_base = {}              # (i,j) -> base rotor index (A=base, B=base+1)
centers = []                 # occupied square centers
pos_list = []                # rotor positions

for j in range(1, Ny + 1):
    for i in range(1, Nx + 1):
        if is_missing(i, j):
            continue

        base = len(pos_list)
        sq_to_base[(i, j)] = base
        square_list.append((i, j))

        x0 = (i - 1) * a
        y0 = (j - 1) * a
        centers.append([x0 + 0.5 * a, y0 + 0.5 * a])

        pos_list.append([x0 + b,     y0 + b])       # A
        pos_list.append([x0 + a - b, y0 + a - b])   # B

pos = np.array(pos_list, dtype=np.float64)
centers = np.array(centers, dtype=np.float64)
Nsq_occ = len(square_list)
Nrot = pos.shape[0]

# rest angles: phi=0 points toward that square center
rest_angle = np.zeros(Nrot, dtype=np.float64)
for s, (i, j) in enumerate(square_list):
    c = centers[s]
    base = sq_to_base[(i, j)]
    for k in (0, 1):
        d = c - pos[base + k]
        rest_angle[base + k] = np.arctan2(d[1], d[0])

# -------------------------
# BUILD PASSIVE NEIGHBOR BONDS WITH BC
# -------------------------
def neighbor_right(i, j):
    if BC == "periodic":
        return (1, j) if i == Nx else (i + 1, j)
    else:
        return (i + 1, j) if i < Nx else None

def neighbor_up(i, j):
    if BC == "periodic":
        return (i, 1) if j == Ny else (i, j + 1)
    else:
        return (i, j + 1) if j < Ny else None

passive_bonds = []
for (i, j) in square_list:
    base = sq_to_base[(i, j)]
    ia, ib = base, base + 1

    nr = neighbor_right(i, j)
    if nr is not None and nr in sq_to_base:
        br = sq_to_base[nr]
        passive_bonds.append((ia, br))       # A-A
        passive_bonds.append((ib, br + 1))   # B-B

    nu = neighbor_up(i, j)
    if nu is not None and nu in sq_to_base:
        bu = sq_to_base[nu]
        passive_bonds.append((ia, bu))       # A-A
        passive_bonds.append((ib, bu + 1))   # B-B

pairs = np.array(passive_bonds, dtype=np.int64) if passive_bonds else np.zeros((0, 2), dtype=np.int64)

# For fixed BC: pin boundary squares (existing ones only)
fixed_pins = []
if BC == "fixed":
    for (i, j) in square_list:
        if (i == 1) or (i == Nx) or (j == 1) or (j == Ny):
            base = sq_to_base[(i, j)]
            fixed_pins.append(base)      # A
            fixed_pins.append(base + 1)  # B
fixed_pins = np.array(fixed_pins, dtype=np.int64) if len(fixed_pins) else np.zeros((0,), dtype=np.int64)

# -------------------------
# BUILD ACTIVE INTERNAL BONDS (COLOR + DYNAMICS ARRAYS)
# -------------------------
active_up = []
active_down = []
active_A = []
active_B = []
active_sgn = []

for (i, j) in square_list:
    sgn = square_sign(i, j)
    if sgn == 0:
        continue
    base = sq_to_base[(i, j)]
    ia, ib = base, base + 1

    if sgn > 0:
        active_up.append((ia, ib))
    else:
        active_down.append((ia, ib))

    active_A.append(ia)
    active_B.append(ib)
    active_sgn.append(float(sgn))

active_A = np.array(active_A, dtype=np.int64)
active_B = np.array(active_B, dtype=np.int64)
active_sgn = np.array(active_sgn, dtype=np.float64)

# -------------------------
# DYNAMICS (phi = deviation)
# -------------------------
@njit(cache=True)
def accel(phi, v,
          Gamma, k0, kc,
          k_link, pairs,
          ka, active_A, active_B, active_sgn,
          k_fix, fixed_pins):
    a_out = np.zeros_like(phi)

    # onsite
    for i in range(phi.shape[0]):
        x = phi[i]
        a_out[i] = -Gamma * v[i] - k0 * x - kc * (x * x * x)

    # passive reciprocal springs
    for p in range(pairs.shape[0]):
        i = pairs[p, 0]
        j = pairs[p, 1]
        d = phi[i] - phi[j]
        ss= phi[i] + phi[j]
        
        a_out[i] += -k_link * d + ks * ss
        a_out[j] += +k_link * d + ks * ss

    # active antisymmetric coupling inside active squares
    for k in range(active_A.shape[0]):
        ia = active_A[k]
        ib = active_B[k]
        sgn = active_sgn[k]
        a_out[ia] += -(ka * sgn) * v[ib]     # here we changed to velocity dependent non-reciprocal force
        a_out[ib] += +(ka * sgn) * v[ia]

    # fixed BC pins (to ground phi=0)
    for t in range(fixed_pins.shape[0]):
        idx = fixed_pins[t]
        a_out[idx] += -k_fix * phi[idx]

    return a_out

@njit(cache=True)
def integrate(phi0, v0, steps, dt,
              Gamma, k0, kc,
              k_link, pairs,
              ka, active_A, active_B, active_sgn,
              k_fix, fixed_pins):
    N = phi0.shape[0]
    phi = np.zeros((steps, N), dtype=np.float64)
    v   = np.zeros((steps, N), dtype=np.float64)

    phi[0] = phi0
    v[0]   = v0

    a0 = accel(phi[0], v[0],
               Gamma, k0, kc,
               k_link, pairs,
               ka, active_A, active_B, active_sgn,
               k_fix, fixed_pins)

    # kick-drift-kick
    for t in range(steps - 1):
        v_half = v[t] + 0.5 * dt * a0
        phi[t + 1] = phi[t] + dt * v_half
        a1 = accel(phi[t + 1], v_half,
                   Gamma, k0, kc,
                   k_link, pairs,
                   ka, active_A, active_B, active_sgn,
                   k_fix, fixed_pins)
        v[t + 1] = v_half + 0.5 * dt * a1
        a0 = a1

    return phi, v

# -------------------------
# INITIAL CONDITIONS: perturb ONLY square (1,1)
# -------------------------
phi0 = np.zeros(Nrot, dtype=np.float64)   
v0   = np.zeros(Nrot, dtype=np.float64)

if (Nx, 1) in sq_to_base:
    base = sq_to_base[(Nx, 1)]
    # positive chirality at (Nx,1): chi = phi_A*v_B - phi_B*v_A = omega*amp^2 > 0
    phi0[base]     = amp
    phi0[base + 1] = 0.0
    v0[base]       = 0.0
    v0[base + 1]   = omega * amp

phi, v = integrate(phi0, v0, steps, dt,
                   Gamma, k0, kc,
                   k_link, pairs,
                   ka, active_A, active_B, active_sgn,
                   k_fix, fixed_pins)

# -------------------------
# ANIMATION PREP
# -------------------------
sample_every = 3
phi_s = phi[::sample_every]                 # deviations
t_s   = np.arange(phi_s.shape[0]) * dt * sample_every
phi_disp_s = phi_s + rest_angle[None, :]    # displayed bar angles

# -------------------------
# PLOT SETUP
# -------------------------
fig, ax = plt.subplots(figsize=(8.4, 8.4))
ax.set_aspect("equal")
ax.set_title(
    f"Nx={Nx}, Ny={Ny}, BC={BC} | outlines=black | passive=green | active +ka=red | active -ka=blue\n"
    "pseudo-spin quiver at occupied centers: R = phi_A + i phi_B"
)

# draw all square outlines in BLACK (including missing)
def draw_square(x0, y0, a):
    sq = np.array([[x0, y0], [x0 + a, y0], [x0 + a, y0 + a], [x0, y0 + a], [x0, y0]], dtype=np.float64)
    ax.plot(sq[:, 0], sq[:, 1], color="black", linewidth=2)

for j in range(1, Ny + 1):
    for i in range(1, Nx + 1):
        draw_square((i - 1) * a, (j - 1) * a, a)

# rotor points
ax.scatter(pos[:, 0], pos[:, 1], s=10, color="black")

# colored bond lines
(passive_lines,) = ax.plot([], [], color="green", linewidth=1.4)
(active_up_lines,) = ax.plot([], [], color="red", linewidth=2.2)
(active_down_lines,) = ax.plot([], [], color="blue", linewidth=2.2)

# bars
bars = [ax.plot([], [], linewidth=2.0)[0] for _ in range(Nrot)]

# legend proxies
ax.plot([], [], color="green", linewidth=1.4, label="passive (reciprocal)")
ax.plot([], [], color="red", linewidth=2.2, label="active +ka (up)")
ax.plot([], [], color="blue", linewidth=2.2, label="active -ka (down)")
#ax.legend(loc="upper right", frameon=True)

# pseudo-spin quiver at occupied centers
spin_scale = 0.18 * a
Qspin = ax.quiver(
    centers[:, 0], centers[:, 1],
    np.zeros(Nsq_occ), np.zeros(Nsq_occ),
    angles="xy", scale_units="xy", scale=1.0,
    width=0.009, pivot="mid"
)

pad = 0.25 * a
ax.set_xlim(-pad, Nx * a + pad)
ax.set_ylim(-pad, Ny * a + pad)

L = 0.22 * a
def bar_endpoints(center, ang):
    dx = L * np.cos(ang)
    dy = L * np.sin(ang)
    p1 = center + np.array([-dx, -dy])
    p2 = center + np.array([ dx,  dy])
    return p1, p2

def set_lines(line, bonds):
    xs, ys = [], []
    for i, j in bonds:
        xs += [pos[i, 0], pos[j, 0], np.nan]
        ys += [pos[i, 1], pos[j, 1], np.nan]
    line.set_data(xs, ys)

def init():
    for ln in bars:
        ln.set_data([], [])
    passive_lines.set_data([], [])
    active_up_lines.set_data([], [])
    active_down_lines.set_data([], [])
    Qspin.set_UVC(np.zeros(Nsq_occ), np.zeros(Nsq_occ))
    return (*bars, passive_lines, active_up_lines, active_down_lines, Qspin)

U_history = []  # Will store all U values over time
V_history = []  # Will store all V values over time

def update(frame):
    ang = phi_disp_s[frame]
    dev = phi_s[frame]

    # bars
    for i in range(Nrot):
        p1, p2 = bar_endpoints(pos[i], ang[i])
        bars[i].set_data([p1[0], p2[0]], [p1[1], p2[1]])

    # bonds (colored)
    set_lines(passive_lines, passive_bonds)
    set_lines(active_up_lines, active_up)
    set_lines(active_down_lines, active_down)

    # pseudo-spin per occupied square: (U,V)=(phi_A, phi_B)
    U = np.zeros(Nsq_occ, dtype=np.float64)
    V = np.zeros(Nsq_occ, dtype=np.float64)
    for s, (i, j) in enumerate(square_list):
        base = sq_to_base[(i, j)]
        U[s] = spin_scale * dev[base]
        V[s] = spin_scale * dev[base + 1]
    
    U_history.append(U.copy())  # Store for plotting
    V_history.append(V.copy())
    
    Qspin.set_UVC(U, V)

    return (*bars, passive_lines, active_up_lines, active_down_lines, Qspin)



# ani = FuncAnimation(fig, update, frames=len(phi_s), init_func=init, interval=30, blit=False, repeat=True)
# plt.show()

# -------------------------
# Save animation (MP4)
# -------------------------
# fps = int(1.0 / (dt * sample_every))
# outname = f"Nx{Nx}_Ny{Ny}_BC-{BC}_perturb-11.mp4"
# ani.save(outname, writer="ffmpeg", fps=fps, dpi=150)
# print("Saved:", outname)
# plt.close(fig)



# Victor's code to plot pseudospin: 

print("Processing simulation data...")

# Initialize empty lists
U_history = []
V_history = []

# Process each time step directly
for frame in range(len(phi_s)):
    ang = phi_disp_s[frame]
    dev = phi_s[frame]
    
    # Calculate U and V (same as in update())
    U = np.zeros(Nsq_occ, dtype=np.float64)
    V = np.zeros(Nsq_occ, dtype=np.float64)
    for s, (i, j) in enumerate(square_list):
        base = sq_to_base[(i, j)]
        U[s] = spin_scale * dev[base]
        V[s] = spin_scale * dev[base + 1]
    
    # Store for plotting
    U_history.append(U.copy())
    V_history.append(V.copy())

# ------------------------------------------------------------
# PLOTTING CODE 
# ------------------------------------------------------------

# 1. First, make sure we have data
if len(U_history) > 0:
    print(f"Collected {len(U_history)} frames of data")
    
    # Create time array (assuming equal time steps)
    time_steps = np.arange(len(U_history))
    
    # Convert to numpy arrays for easier manipulation
    U_array = np.array(U_history)  # Shape: (num_frames, Nsq_occ)
    V_array = np.array(V_history)  # Shape: (num_frames, Nsq_occ)
    
    # --------------------------------------------------------
    # PLOT 1: Time evolution of specific squares
    # --------------------------------------------------------
    plt.figure(figsize=(14, 10))
    
    # Choose which squares to plot (e.g., first 2 squares or specific positions)
    squares_to_plot = [0, 5, 9, 10]  # Plot squares
    n = len(squares_to_plot)
    
    for idx in range(n):
        plt.subplot(len(squares_to_plot), 1, idx+1)
        
        # Get U and V for this square over all frames
        U_square = U_array[:, squares_to_plot[idx]]
        V_square = V_array[:, squares_to_plot[idx]]

        plt.plot(time_steps, U_square, 'b-', label=f'U (ϕ_A)', linewidth=2)
        plt.plot(time_steps, V_square, 'r-', label=f'V (ϕ_B)', linewidth=2)
        plt.xlabel('Time/Frame Number')
        plt.ylabel('Angle')
        plt.title(f'Square {squares_to_plot[idx]} at position {square_list[squares_to_plot[idx]]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pseudospin_time_evolution.png', dpi=150, bbox_inches='tight')
    
    # --------------------------------------------------------
    # PLOT 2: Spatial distribution at first, middle, last frame
    # --------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    frames_to_plot = [0, len(U_history)//2, len(U_history)-1]
    frame_titles = ['Start', 'Middle', 'End']
    
    for row, (frame_idx, title) in enumerate(zip(frames_to_plot, frame_titles)):
        U_frame = U_array[frame_idx]
        V_frame = V_array[frame_idx]
        
        # Get positions
        positions = np.array(square_list)
        
        # Plot U
        sc1 = axes[row, 0].scatter(positions[:, 0], positions[:, 1], 
                                   c=U_frame, cmap='RdBu', s=100, alpha=0.8)
        axes[row, 0].set_title(f'U (ϕ_A) at {title} (frame {frame_idx})')
        axes[row, 0].set_xlabel('i position')
        axes[row, 0].set_ylabel('j position')
        axes[row, 0].grid(True, alpha=0.3)
        plt.colorbar(sc1, ax=axes[row, 0])
        
        # Plot V
        sc2 = axes[row, 1].scatter(positions[:, 0], positions[:, 1], 
                                   c=V_frame, cmap='RdBu', s=100, alpha=0.8)
        axes[row, 1].set_title(f'V (ϕ_B) at {title} (frame {frame_idx})')
        axes[row, 1].set_xlabel('i position')
        axes[row, 1].set_ylabel('j position')
        axes[row, 1].grid(True, alpha=0.3)
        plt.colorbar(sc2, ax=axes[row, 1])
    
    plt.tight_layout()
    plt.savefig('pseudospin_spatial_distribution.png', dpi=150, bbox_inches='tight')
    
    # --------------------------------------------------------
    # PLOT 3: Mean U and V over time (system average)
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    
    mean_U_over_time = np.mean(U_array, axis=1)
    mean_V_over_time = np.mean(V_array, axis=1)
    
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, mean_U_over_time, 'b-', linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Mean U value')
    plt.title('System Average: U (ϕ_A)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, mean_V_over_time, 'r-', linewidth=2)
    plt.xlabel('Frame Number')
    plt.ylabel('Mean V value')
    plt.title('System Average: V (ϕ_B)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pseudospin_system_average.png', dpi=150, bbox_inches='tight')
    
    # --------------------------------------------------------
    # PLOT 4: Phase space (U vs V) for a specific square
    # --------------------------------------------------------
    plt.figure(figsize=(12, 10))
    
    for idx in range(len(squares_to_plot)):  # Plot first 4 squares
        plt.subplot(2, 2, idx+1)
        
        U_square = U_array[:, squares_to_plot[idx]]
        V_square = V_array[:, squares_to_plot[idx]]
        # Color by time (frame number)
        scatter = plt.scatter(U_square, V_square, c=time_steps, 
                             cmap='viridis', s=20, alpha=0.7)
        plt.xlabel('U (ϕ_A)')
        plt.ylabel('V (ϕ_B)')
        plt.title(f'Phase Space: Square {squares_to_plot[idx]} at {square_list[squares_to_plot[idx]]}')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter)
        cbar.set_label('Frame Number')
        
        # Mark start and end points
        plt.scatter(U_square[0], V_square[0], s=100, c='green', 
                   marker='o', label='Start', edgecolors='black')
        plt.scatter(U_square[-1], V_square[-1], s=100, c='red', 
                   marker='s', label='End', edgecolors='black')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('pseudospin_phase_space.png', dpi=150, bbox_inches='tight')

    # Plot 5: spatio temporal heatmap of U and V
    plt.figure(figsize=(14, 6))
    # Spatio-temporal heatmap for U and V
    plt.subplot(1, 2, 1)
    plt.imshow(U_array.T, aspect='auto', cmap='RdBu', origin='lower', 
               extent=[0, len(U_history), 0, Nsq_occ])
    plt.xlabel('Frame Number')
    plt.ylabel('Square Index')
    plt.title('Spatio-Temporal Heatmap: U (ϕ_A)')
    plt.colorbar(label='U value')

    plt.subplot(1, 2, 2)
    plt.imshow(V_array.T, aspect='auto', cmap='RdBu', origin='lower',
               extent=[0, len(U_history), 0, Nsq_occ])
    plt.xlabel('Frame Number')
    plt.ylabel('Square Index')
    plt.title('Spatio-Temporal Heatmap: V (ϕ_B)')
    plt.colorbar(label='V value')

    plt.tight_layout()
    plt.show()
    
    # --------------------------------------------------------
    # OPTIONAL: Save data to file for later analysis
    # --------------------------------------------------------
    save_data = True  # Set to False if you don't want to save files
    if save_data:
        np.save('U_history.npy', U_array)
        np.save('V_history.npy', V_array)
        np.save('square_positions.npy', positions)
        print("Data saved to: U_history.npy, V_history.npy, square_positions.npy")
        
        # Also save as text file (easier to open in Excel)
        np.savetxt('U_history.csv', U_array, delimiter=',')
        np.savetxt('V_history.csv', V_array, delimiter=',')
        print("Data saved as CSV files for Excel/other programs")

else:
    print("No data collected! Make sure U_history and V_history are being filled.")
    print(f"U_history length: {len(U_history)}, V_history length: {len(V_history)}")
