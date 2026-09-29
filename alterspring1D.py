from turtle import pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib.colors import PowerNorm

# =========================================================
# Parameters
# =========================================================
N = 20         # number of unit cells
m = 1.0
k_base = 1.0          # onsite linear restoring
k_noise = 0
np.random.seed(42)  # for reproducibility
k = k_base + k_noise * np.random.randn(N)  # add some disorder
ka_base = 4.0      # nonreciprocal intracell coupling
ka_noise = 1
#ka = ka_base*(2*np.random.randint(0,2, size=N)-1) + ka_noise * np.random.randn(N)  # add some disorder
ka = np.abs(ka_base + ka_noise * np.random.randn(N))
#ka = [15, -15]
kc = 10.0        # cubic nonlinearity
gamma = 1
#gamma = 1 + abs(0.2* np.random.randn(N) )  # damping
kn = 10.0     # nearest-neighbor coupling along x
d = 10.0          # vertical offset within each cell
a = 1.0          # lattice spacing

# Time
dt = 0.01
T = 100.0
steps = int(T / dt)
t = np.linspace(0, T, steps)

# =========================================================
# Initial conditions
# y1[n], v1[n], y2[n], v2[n]
# =========================================================
y1_0 = np.linspace(-2, 2, N)  # initial profile with kink 
v1_0 = np.zeros(N)
y2_0 = np.ones(N)
v2_0 = np.zeros(N)

# y1_0 = np.random.randn(N)
# v1_0 = np.random.randn(N)
# y2_0 = np.random.randn(N)
# v2_0 = np.random.randn(N)

# Example: localized excitation near the center
# c = 0#N // 2
# y1_0[c] = 0.8
# v2_0[c] = 0.6

# added by Victor
# y1_0[1] = 0.5

# Pack state:
# state = [y1(0..N-1), v1(0..N-1), y2(0..N-1), v2(0..N-1)]
state0 = np.concatenate([y1_0, v1_0, y2_0, v2_0])

# =========================================================
# Open-boundary discrete Laplacian
# =========================================================
def laplacian_open(arr):
    lap = np.zeros_like(arr)
    if len(arr) == 1:
        return lap

    lap[0] = arr[1] - arr[0]
    lap[-1] = arr[-2] - arr[-1]
    lap[1:-1] = arr[2:] + arr[:-2] - 2.0 * arr[1:-1]
    return lap

# =========================================================
# RHS of the ODE system
# =========================================================
def rhs(state):
    y1 = state[0:N]
    v1 = state[N:2*N]
    y2 = state[2*N:3*N]
    v2 = state[3*N:4*N]

    lap1 = laplacian_open(y1)
    lap2 = laplacian_open(y2)

    a1 = (-k * y1 + ka * y2 - kc * y1**3 - gamma * v1 + kn * lap1) / m
    a2 = (-k * y2 - ka * y1 - kc * y2**3 - gamma * v2 + kn * lap2) / m

    return np.concatenate([v1, a1, v2, a2])

# =========================================================
# RK4 integration
# =========================================================
sol = np.zeros((steps, 4 * N))
sol[0] = state0

for i in range(steps - 1):
    s = sol[i]

    k1_rk = rhs(s)
    k2_rk = rhs(s + 0.5 * dt * k1_rk)
    k3_rk = rhs(s + 0.5 * dt * k2_rk)
    k4_rk = rhs(s + dt * k3_rk)

    sol[i + 1] = s + (dt / 6.0) * (k1_rk + 2*k2_rk + 2*k3_rk + k4_rk)

# =========================================================
# Extract solution
# =========================================================
y1 = sol[:, 0:N]
v1 = sol[:, N:2*N]
y2 = sol[:, 2*N:3*N]
v2 = sol[:, 3*N:4*N]

# Absolute coordinates for the live geometry
x_sites = np.arange(N) * a
Y1 = d + y1
Y2 = -d + y2

# =========================================================
# Complex field R = y1 + i y2
# =========================================================
R = y1 + 1j * y2
AmpR = np.abs(R)
ArgR = np.angle(R)   # in [-pi, pi]

##### plot of initial configuration
plt.figure(figsize=(25, 3))
plt.title("initial config")
plt.subplot(2,2,1)
plt.plot(x_sites, y1[-0])
plt.subplot(2,2,2)
plt.plot(x_sites, ArgR[0])

plt.subplot(2,2,3)
plt.plot(x_sites, y1[-1])
plt.subplot(2,2,4)
plt.plot(x_sites, ArgR[-1])
plt.show()

# ---------------------------------------------------------
# 0) Limit cycles
# ---------------------------------------------------------
# plt.figure(figsize=(25,3))
# plt.title("Phase space: y1 vs y2")
# for n in range(N):
#     ax6 = plt.subplot(1, N, n+1)
#     ax6.set_xlabel(r"$y_1$")
#     ax6.set_ylabel(r"$y_2$")
#     ax6.plot(y1[:,n], y2[:,n], 'k.', markersize=1, alpha=0.5)
#     ax6.set_aspect('equal')
#     ax6.set_title(f"k={k[n]:.2f}, ka={ka[n]:.2f}", fontsize=7)
# plt.show()

# =========================================================
# Plot setup: 5 panels
#   1) live geometry
#   2) y1(x,t)
#   3) y2(x,t)
#   4) arg(R)(x,t)
#   5) |R|(x,t)# =========================================================
fig, ax = plt.subplots(figsize=(22, 5))
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, fontweight='bold')

# ---------------------------------------------------------
# 1) Geometry panel
# ---------------------------------------------------------
ax1 = plt.subplot(1, 5, 1)
ax1.set_title("Lattice geometry")
ax1.set_ylim(-a, (N - 1) * a + a)
ax1.set_xlim(-d - 2.0, d + 2.0)
ax1.set_aspect('equal')
ax1.axvline(0, color='k', linestyle='--', alpha=0.3)

# rest positions
ax1.plot(d * np.ones(N), x_sites, 'kx', ms=5)
ax1.plot( -d * np.ones(N), x_sites,  'kx', ms=5)

# faint neighbor guides
for n in range(N - 1):
    ax1.plot([d, d], [x_sites[n], x_sites[n+1]], color='gray', alpha=0.12)
    ax1.plot([-d, -d], [x_sites[n], x_sites[n+1]], color='gray', alpha=0.12)

top_scatter, = ax1.plot([], [], 'o', ms=7, label='top masses')
bot_scatter, = ax1.plot([], [], 'o', ms=7, label='bottom masses')
ax1.legend(loc='center right')

# ---------------------------------------------------------
# 2) y1(x,t)
# ---------------------------------------------------------
ax2 = plt.subplot(1, 5, 2)
ax2.set_title(r"$y_1(x,t)$")
ax2.set_xlabel("x")
ax2.set_ylabel("time")

vmax1 = np.max(np.abs(y1))
if vmax1 == 0:
    vmax1 = 1.0

im1 = ax2.imshow(
    y1,
    aspect='auto',
    origin='lower',
    extent=[x_sites[0], x_sites[-1], t[0], t[-1]],
    vmin=-vmax1,
    vmax=vmax1,
    interpolation='nearest'
)
plt.colorbar(im1, ax=ax2, label=r"$y_1$")

# ---------------------------------------------------------
# 3) y2(x,t)
# ---------------------------------------------------------
ax3 = plt.subplot(1, 5, 3)
ax3.set_title(r"$y_2(x,t)$")
ax3.set_xlabel("x")
ax3.set_ylabel("time")

vmax2 = np.max(np.abs(y2))
if vmax2 == 0:
    vmax2 = 1.0

im2 = ax3.imshow(
    y2,
    aspect='auto',
    origin='lower',
    extent=[x_sites[0], x_sites[-1], t[0], t[-1]],
    vmin=-vmax2,
    vmax=vmax2,
    interpolation='nearest'
)
plt.colorbar(im2, ax=ax3, label=r"$y_2$")

# ---------------------------------------------------------
# 4) arg(R)(x,t)
# ---------------------------------------------------------
ax4 = plt.subplot(1, 5, 4)
ax4.set_title(r"$\arg(R)(x,t)$")
ax4.set_xlabel("x")
ax4.set_ylabel("time")

im3 = ax4.imshow(
    ArgR,
    aspect='auto',
    origin='lower',
    extent=[x_sites[0], x_sites[-1], t[0], t[-1]],
    vmin=-np.pi,
    vmax=np.pi,
    interpolation='nearest'
)
cbar3 = plt.colorbar(im3, ax=ax4, label=r"$\arg(R)$")
cbar3.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar3.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

# ---------------------------------------------------------
# 5) |R|(x,t)
# ---------------------------------------------------------
ax5 = plt.subplot(1, 5, 5)
ax5.set_title(r"$|R|(x,t)$")
ax5.set_xlabel("x")
ax5.set_ylabel("time")

vmax_amp = np.max(AmpR[-1])
if vmax_amp == 0:
    vmax_amp = 1.0

im4 = ax5.imshow(
    AmpR,
    aspect='auto',
    origin='lower',
    extent=[x_sites[0], x_sites[-1], t[0], t[-1]],
    vmin=0,
    vmax=vmax_amp,
    interpolation='nearest'
)
plt.colorbar(im4, ax=ax5, label=r"$|R|$")

# ---------------------------------------------------------
# 
# ---------------------------------------------------------

# =========================================================
# Animation: only the geometry panel is animated
# =========================================================
def init():
    top_scatter.set_data([], [])
    bot_scatter.set_data([], [])
    time_text.set_text('')
    return top_scatter, bot_scatter, time_text

def update(frame):
    top_scatter.set_data(Y1[frame], x_sites)
    bot_scatter.set_data(Y2[frame], x_sites)
    time_text.set_text(f"t = {frame * dt:.2f} s")
    return top_scatter, bot_scatter, time_text

ani = FuncAnimation(
    fig,
    update,
    frames=steps,
    init_func=init,
    interval=1,
    blit=True
)

plt.tight_layout()
plt.show()

# fig, ax5 = plt.subplots(figsize=(5,5))
# #ax5.set_title(r"$|R|(x,t)$")
# ax5.set_xlabel("Chain position", fontsize=20)
# ax5.set_ylabel("time", fontsize=20)
# ax5.tick_params(labelsize=16)

# vmax_amp = np.max(AmpR)
# if vmax_amp == 0:
#     vmax_amp = 1.0

# jtime = 4000

# im4 = ax5.imshow(
#     AmpR[:jtime, :],
#     aspect='auto',
#     origin='lower',
#     extent=[x_sites[0], x_sites[-1], t[0], t[jtime]],
#     norm=PowerNorm(gamma=3.0, vmin=0, vmax=vmax_amp),
#     interpolation='nearest'
# )
# cbar = plt.colorbar(im4, ax=ax5, ticks=[0, 1.0, 1.2])
# cbar.set_label(r"$|R|$", fontsize=20)
# cbar.ax.tick_params(labelsize=16)
# plt.tight_layout()
# # plt.savefig("kink_chain.pdf", format="pdf", bbox_inches="tight")
# plt.show()

# =========================================================
# Measure Amplitude and Frequency for Individual Masses
# =========================================================
transient_fraction = 0.5
start_idx = int(steps * transient_fraction)
t_steady = t[start_idx:]
N_steady = len(t_steady)

# FFT Frequency axis (in Hz)
freqs = np.fft.fftfreq(N_steady, d=dt)
pos_freq_mask = freqs >= 0
freqs_pos = freqs[pos_freq_mask]

# results = []

# print("=========================================================")
# print("     STEADY-STATE AMPLITUDE & FREQUENCY PER MASS         ")
# print("=========================================================")

# # Function to analyze a 1D time series signal
# def analyze_signal(signal, label):
#     # 1. Peak Amplitude (Envelope method for limit cycles)
#     amp_peak = (np.max(signal) - np.min(signal)) / 2.0
#     amp_rms = np.std(signal) * np.sqrt(2) # RMS equivalent for pure sine
    
#     # 2. Dominant Frequency via FFT
#     # Subtract mean to remove DC offset before FFT
#     signal_centered = signal - np.mean(signal)
#     fft_mag = np.abs(np.fft.fft(signal_centered))[pos_freq_mask]
    
#     peak_idx = np.argmax(fft_mag)
#     f_peak_hz = freqs_pos[peak_idx]
#     omega_peak = 2.0 * np.pi * f_peak_hz
    
#     print(f"\n--- {label} ---")
#     print(f"  • Peak Amplitude:    {amp_peak:.4f}")
#     print(f"  • RMS Amplitude:    {amp_rms:.4f}")
#     print(f"  • Frequency (FFT):   {omega_peak:.4f} rad/s ({f_peak_hz:.4f} Hz)")
    
#     return {
#         'label': label,
#         'signal': signal,
#         'amplitude': amp_peak,
#         'omega': omega_peak,
#         'freq_hz': f_peak_hz,
#         'fft_mag': fft_mag
#     }

# # Loop through each unit cell and measure individual masses y1 and y2
# mass_data = []

# for n in range(N):
#     # Top Mass y1[n]
#     y1_signal = y1[start_idx:, n]
#     res_y1 = analyze_signal(y1_signal, f"Cell {n} - Top Mass (y1)")
#     mass_data.append(res_y1)
    
#     # Bottom Mass y2[n]
#     y2_signal = y2[start_idx:, n]
#     res_y2 = analyze_signal(y2_signal, f"Cell {n} - Bottom Mass (y2)")
#     mass_data.append(res_y2)

# # =========================================================
# # Plotting Individual Mass Spectra & Signals
# # =========================================================
# fig, axes = plt.subplots(len(mass_data), 2, figsize=(12, 2.5 * len(mass_data)))

# for idx, res in enumerate(mass_data):
#     # Time Series Plot
#     axes[idx, 0].plot(t_steady, res['signal'], 'b-', alpha=0.8)
#     axes[idx, 0].set_ylabel(r"$y$ displacement")
#     axes[idx, 0].set_xlabel("Time (s)")
#     axes[idx, 0].set_title(f"{res['label']} - Time Domain (Amp = {res['amplitude']:.3f})")
#     axes[idx, 0].grid(True, alpha=0.3)
    
#     # Frequency Spectrum Plot
#     axes[idx, 1].plot(freqs_pos, res['fft_mag'], 'r-')
#     axes[idx, 1].set_ylabel("FFT Magnitude")
#     axes[idx, 1].set_xlabel("Frequency (Hz)")
#     axes[idx, 1].set_xlim(0, 5) # Focus on low-frequency window
#     axes[idx, 1].set_title(f"{res['label']} - Spectrum (Peak = {res['omega']:.3f} rad/s)")
#     axes[idx, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# =========================================================
# Create a list to hold dataframes for each site
# Create DataFrame for all top sites (y1)
all_dfs = []
for n in range(N):
    df = pd.DataFrame({
        'time': t,
        'disp_y': y1[:, n],
        'node_id': f'top_{n}'
    })
    all_dfs.append(df)

# Create DataFrame for all bottom sites (y2)
for n in range(N):
    df = pd.DataFrame({
        'time': t,
        'disp_y': y2[:, n],
        'node_id': f'bottom_{n}'
    })
    all_dfs.append(df)

# Export to CSV
#all_dfs = pd.concat(all_dfs, ignore_index=True)
#all_dfs.to_csv('simulation_data_wave_coarse2.csv', index=False)