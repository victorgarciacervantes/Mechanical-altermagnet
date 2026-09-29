import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# =========================================================
# PARAMETERS
# =========================================================

# Lieb lattice size
Lx = 1
Ly = 1

m = 0.1

# onsite restoring force
k_base = 1.0
k_noise = 0.0

# chiral coupling
ka0 = 3.0

# cubic nonlinearity
kc = 200.0

# damping
gamma = 0.1

# nearest-neighbor spring coupling
kn = 1.0

# random seed
np.random.seed(42)

# =========================================================
# TIME
# =========================================================

dt = 0.01
T = 10.0

steps = int(T/dt)
t = np.linspace(0, T, steps)

# =========================================================
# BUILD LIEB LATTICE
# =========================================================

def build_lieb_lattice(Lx, Ly):

    positions = []
    site_type = []

    A = {}
    B = {}
    C = {}

    idx = 0

    for i in range(Lx):
        for j in range(Ly):

            A[(i,j)] = idx
            positions.append([i, j])
            site_type.append("A")
            idx += 1

            B[(i,j)] = idx
            positions.append([i+0.5, j])
            site_type.append("B")
            idx += 1

            C[(i,j)] = idx
            positions.append([i, j+0.5])
            site_type.append("C")
            idx += 1

    Ns = idx

    neighbors = [[] for _ in range(Ns)]

    for i in range(Lx):
        for j in range(Ly):

            a = A[(i,j)]
            b = B[(i,j)]
            c = C[(i,j)]

            neighbors[a].append(b)
            neighbors[b].append(a)

            neighbors[a].append(c)
            neighbors[c].append(a)

            if i > 0:
                b_left = B[(i-1,j)]
                neighbors[a].append(b_left)
                neighbors[b_left].append(a)

            if j > 0:
                c_down = C[(i,j-1)]
                neighbors[a].append(c_down)
                neighbors[c_down].append(a)

    return (
        np.array(positions),
        np.array(site_type),
        neighbors
    )

positions, site_type, neighbors = build_lieb_lattice(Lx, Ly)

Ns = len(positions)

# =========================================================
# PARAMETERS ON EACH SITE
# =========================================================

k = k_base + k_noise*np.random.randn(Ns)

ka = np.zeros(Ns)

for n, tp in enumerate(site_type):

    if tp == "A":
        ka[n] = 0

    elif tp == "B":
        ka[n] = -ka0

    elif tp == "C":
        ka[n] = +ka0

# =========================================================
# INITIAL CONDITIONS
# =========================================================

y1_0 = np.zeros(Ns)
y2_0 = np.zeros(Ns)

v1_0 = np.zeros(Ns)
v2_0 = np.zeros(Ns)

# localized excitation near center

center = np.argmin(
    np.sum(
        (positions - positions.mean(axis=0))**2,
        axis=1
    )
)

y1_0[center] = 1.0
v2_0[center] = 0.5

state0 = np.concatenate([
    y1_0,
    v1_0,
    y2_0,
    v2_0
])

# =========================================================
# GRAPH LAPLACIAN
# =========================================================

def graph_laplacian(y):

    lap = np.zeros_like(y)

    for i, nbrs in enumerate(neighbors):

        if len(nbrs) > 0:
            lap[i] = np.sum(y[nbrs] - y[i])

    return lap

# =========================================================
# RHS
# =========================================================

def rhs(state):

    y1 = state[0:Ns]
    v1 = state[Ns:2*Ns]

    y2 = state[2*Ns:3*Ns]
    v2 = state[3*Ns:4*Ns]

    lap1 = graph_laplacian(y1)
    lap2 = graph_laplacian(y2)

    a1 = (
        -k*y1
        + ka*y2
        - kc*y1**3
        - gamma*v1
        + kn*lap1
    )/m

    a2 = (
        -k*y2
        - ka*y1
        - kc*y2**3
        - gamma*v2
        + kn*lap2
    )/m

    return np.concatenate([
        v1,
        a1,
        v2,
        a2
    ])

# =========================================================
# RK4
# =========================================================

sol = np.zeros((steps, 4*Ns))
sol[0] = state0

for i in range(steps-1):

    s = sol[i]

    k1 = rhs(s)
    k2 = rhs(s + 0.5*dt*k1)
    k3 = rhs(s + 0.5*dt*k2)
    k4 = rhs(s + dt*k3)

    sol[i+1] = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# =========================================================
# EXTRACT FIELDS
# =========================================================

y1 = sol[:,0:Ns]
v1 = sol[:,Ns:2*Ns]

y2 = sol[:,2*Ns:3*Ns]
v2 = sol[:,3*Ns:4*Ns]

R = y1 + 1j*y2

AmpR = np.abs(R)
ArgR = np.angle(R)

# =========================================================
# ANGULAR MOMENTUM
# =========================================================

J = y1*v2 - y2*v1

J_avg = np.mean(J, axis=0)

J_total = np.sum(J, axis=1)

J_total_avg = np.mean(J_total)

print("Time-averaged total angular momentum = ",
      J_total_avg)

# =========================================================
# PLOTS
# =========================================================

fig = plt.figure(figsize=(18,5))

# ---------------------------------------------------------
# average angular momentum
# ---------------------------------------------------------

ax1 = plt.subplot(1,5,1)

for i,nbrs in enumerate(neighbors):
    for j in nbrs:
        if j > i:
            ax1.plot(
                [positions[i,0], positions[j,0]],
                [positions[i,1], positions[j,1]],
                'k-',
                alpha=0.1
            )

sc = ax1.scatter(
    positions[:,0],
    positions[:,1],
    c=J_avg,
    s=100,
    cmap='RdBu_r'
)

ax1.set_title(r'$\langle J \rangle$')
ax1.set_aspect('equal')

plt.colorbar(sc, ax=ax1)

# ---------------------------------------------------------
# y1
# ---------------------------------------------------------

ax2 = plt.subplot(1,5,2)

im = ax2.imshow(
    y1,
    aspect='auto',
    origin='lower'
)

ax2.set_title(r'$y_1$')

plt.colorbar(im, ax=ax2)

# ---------------------------------------------------------
# y2
# ---------------------------------------------------------

ax3 = plt.subplot(1,5,3)

im = ax3.imshow(
    y2,
    aspect='auto',
    origin='lower'
)

ax3.set_title(r'$y_2$')

plt.colorbar(im, ax=ax3)

# ---------------------------------------------------------
# phase
# ---------------------------------------------------------

ax4 = plt.subplot(1,5,4)

im = ax4.imshow(
    ArgR,
    aspect='auto',
    origin='lower',
    vmin=-np.pi,
    vmax=np.pi
)

ax4.set_title(r'arg(R)')

plt.colorbar(im, ax=ax4)

# ---------------------------------------------------------
# amplitude
# ---------------------------------------------------------

ax5 = plt.subplot(1,5,5)

im = ax5.imshow(
    AmpR,
    aspect='auto',
    origin='lower'
)

ax5.set_title(r'|R|')

plt.colorbar(im, ax=ax5)

plt.tight_layout()

# =========================================================
# ANIMATION
# =========================================================

fig2, ax = plt.subplots(figsize=(7,7))

for i,nbrs in enumerate(neighbors):
    for j in nbrs:
        if j > i:
            ax.plot(
                [positions[i,0], positions[j,0]],
                [positions[i,1], positions[j,1]],
                'k-',
                alpha=0.15
            )

ax.set_aspect('equal')

amp0 = np.sqrt(
    y1[0]**2 + y2[0]**2
)

scat = ax.scatter(
    positions[:,0],
    positions[:,1],
    c=amp0,
    s=120,
    cmap='viridis'
)

ax.set_title("Lieb lattice dynamics")

def update(frame):

    amp = np.sqrt(
        y1[frame]**2 +
        y2[frame]**2
    )

    scat.set_array(amp)

    disp = positions + 0.1*np.column_stack(
        [y1[frame], y2[frame]]
    )

    scat.set_offsets(disp)

    return scat,

ani = FuncAnimation(
    fig2,
    update,
    frames=steps,
    interval=10,
    blit=True
)

plt.show()