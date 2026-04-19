"""
Triple Pendulum – undamped, unforced chaotic dynamics
Physics ported from triple-pendulum.html (RK4 + Lagrangian / Cramer's rule)

Run:
    pip install matplotlib numpy
    python triple_pendulum.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Parameters ────────────────────────────────────────────────────────────────
G  = 9.81
m1, m2, m3 = 1.0, 1.0, 1.0
L1, L2, L3 = 1.0, 1.0, 1.0

DT             = 0.004   # time step (s)
STEPS_PER_FRAME = 6      # physics steps per rendered frame
TRACE_LEN      = 600     # how many tail points to keep

# ── Initial state: [θ1, ω1, θ2, ω2, θ3, ω3] ──────────────────────────────────
def default_state():
    return np.array([np.pi / 2, 0.0,
                     np.pi / 2 * 0.9, 0.0,
                     np.pi / 2 * 1.1, 0.0])

# ── Equations of motion (Lagrangian, 3×3 mass matrix solved by Cramer's rule) ─
def derivatives(s):
    t1, w1, t2, w2, t3, w3 = s

    c12, s12 = np.cos(t1 - t2), np.sin(t1 - t2)
    c13, s13 = np.cos(t1 - t3), np.sin(t1 - t3)
    c23, s23 = np.cos(t2 - t3), np.sin(t2 - t3)

    # Mass matrix
    M = np.array([
        [(m1 + m2 + m3) * L1,  (m2 + m3) * L2 * c12,  m3 * L3 * c13],
        [(m2 + m3) * L1 * c12, (m2 + m3) * L2,         m3 * L3 * c23],
        [m3 * L1 * c13,         m3 * L2 * c23,          m3 * L3      ],
    ])

    # Right-hand side (generalised forces)
    R = np.array([
        -(m2 + m3) * L2 * w2**2 * s12
         - m3 * L3 * w3**2 * s13
         - (m1 + m2 + m3) * G * np.sin(t1),

         (m2 + m3) * L1 * w1**2 * s12
         - m3 * L3 * w3**2 * s23
         - (m2 + m3) * G * np.sin(t2),

         m3 * L1 * w1**2 * s13
         + m3 * L2 * w2**2 * s23
         - m3 * G * np.sin(t3),
    ])

    # Angular accelerations via Cramer's rule (numpy linalg for clarity)
    alpha = np.linalg.solve(M, R)   # [α1, α2, α3]

    return np.array([w1, alpha[0], w2, alpha[1], w3, alpha[2]])

# ── RK4 integrator ────────────────────────────────────────────────────────────
def rk4(s, dt):
    k1 = derivatives(s)
    k2 = derivatives(s + 0.5 * dt * k1)
    k3 = derivatives(s + 0.5 * dt * k2)
    k4 = derivatives(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ── Cartesian positions from state ────────────────────────────────────────────
def positions(s):
    t1, _, t2, _, t3, _ = s
    x1 = L1 * np.sin(t1);  y1 = -L1 * np.cos(t1)
    x2 = x1 + L2 * np.sin(t2);  y2 = y1 - L2 * np.cos(t2)
    x3 = x2 + L3 * np.sin(t3);  y3 = y2 - L3 * np.cos(t3)
    return (0.0, 0.0), (x1, y1), (x2, y2), (x3, y3)

# ── Total energy (conservation check) ────────────────────────────────────────
def energy(s):
    t1, w1, t2, w2, t3, w3 = s
    y1 = -L1 * np.cos(t1)
    y2 = y1 - L2 * np.cos(t2)
    y3 = y2 - L3 * np.cos(t3)
    v1sq = L1**2 * w1**2
    v2sq = (L1**2 * w1**2 + L2**2 * w2**2
            + 2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2))
    v3sq = (L1**2 * w1**2 + L2**2 * w2**2 + L3**2 * w3**2
            + 2 * L1 * L2 * w1 * w2 * np.cos(t1 - t2)
            + 2 * L1 * L3 * w1 * w3 * np.cos(t1 - t3)
            + 2 * L2 * L3 * w2 * w3 * np.cos(t2 - t3))
    KE = 0.5 * (m1 * v1sq + m2 * v2sq + m3 * v3sq)
    PE = G * (m1 * y1 + m2 * y2 + m3 * y3)
    return KE + PE

# ── Matplotlib animation ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), facecolor='#070709')
ax.set_facecolor('#070709')
ax.set_xlim(-(L1 + L2 + L3) * 1.15, (L1 + L2 + L3) * 1.15)
ax.set_ylim(-(L1 + L2 + L3) * 1.15, (L1 + L2 + L3) * 1.15)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Triple Pendulum', color='#e8c84a', fontsize=13,
             fontfamily='monospace', pad=12)

# Rods
rod,  = ax.plot([], [], '-', color='#dcd2be', lw=2.0, zorder=2)
# Bobs
bob1, = ax.plot([], [], 'o', color='#e8c84a', ms=12, zorder=3)
bob2, = ax.plot([], [], 'o', color='#c0392b', ms=9,  zorder=3)
bob3, = ax.plot([], [], 'o', color='#2ecc71', ms=7,  zorder=3)
# Trace of the third bob
trace_x, trace_y = [], []
trace_line, = ax.plot([], [], '-', color='#2ecc71', lw=0.8,
                      alpha=0.55, zorder=1)
# Energy text
etxt = ax.text(0.02, 0.97, '', transform=ax.transAxes,
               color='#aaa890', fontsize=8, va='top', fontfamily='monospace')

state = default_state()

def init():
    rod.set_data([], [])
    bob1.set_data([], [])
    bob2.set_data([], [])
    bob3.set_data([], [])
    trace_line.set_data([], [])
    etxt.set_text('')
    return rod, bob1, bob2, bob3, trace_line, etxt

def animate(_frame):
    global state, trace_x, trace_y
    for _ in range(STEPS_PER_FRAME):
        state = rk4(state, DT)

    p0, p1, p2, p3 = positions(state)

    # Rods
    xs = [p0[0], p1[0], p2[0], p3[0]]
    ys = [p0[1], p1[1], p2[1], p3[1]]
    rod.set_data(xs, ys)

    bob1.set_data([p1[0]], [p1[1]])
    bob2.set_data([p2[0]], [p2[1]])
    bob3.set_data([p3[0]], [p3[1]])

    # Rolling trace
    trace_x.append(p3[0])
    trace_y.append(p3[1])
    if len(trace_x) > TRACE_LEN:
        trace_x.pop(0)
        trace_y.pop(0)
    trace_line.set_data(trace_x, trace_y)

    etxt.set_text(f'E = {energy(state):.4f} J   '
                  f'θ₁={np.degrees(state[0]):+.1f}°  '
                  f'θ₂={np.degrees(state[2]):+.1f}°  '
                  f'θ₃={np.degrees(state[4]):+.1f}°')

    return rod, bob1, bob2, bob3, trace_line, etxt

ani = animation.FuncAnimation(
    fig, animate, init_func=init,
    interval=16,        # ~60 fps
    blit=True,
    cache_frame_data=False,
)

plt.tight_layout()
plt.show()
