import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 参数设置 ---
mu = 3.986e14  # 地球引力常数
Re = 6371000  # 地球半径
h = 500000  # 轨道高度 500km
a = Re + h  # 半长轴
n = np.sqrt(mu / a ** 3)  # 轨道平均角速度 (rad/s)
m = 100.0  # 追踪器质量 (kg)
kp = 0.01  # 势场增益 (模拟 kp)

# 初始状态 [x, y, z, vx, vy, vz]
# 初始位置：目标后方 200米 (V-bar 负方向)
r0 = np.array([0, -200, 0])
v0 = np.array([0, 0, 0])
Y0 = np.concatenate((r0, v0))


# --- 动力学方程 (CW 方程 + 仅势场控制力) ---
def dynamics_only_potential(t, Y):
    r = Y[0:3]
    v = Y[3:6]

    # 仅施加第一项：势场梯度力 (模拟 - grad(Vp))
    # 假设 Vp = 0.5 * kp * |r|^2，则 grad(Vp) = kp * r
    # 控制力 F = - kp * r
    # 加速度 u = F / m
    u = - (kp / m) * r

    # CW 方程 (Clohessy-Wiltshire)
    # x: 径向 (R-bar), y: 切向 (V-bar), z: 轨道法向
    ax = 2 * n * v[1] + 3 * n ** 2 * r[0] + u[0]
    ay = -2 * n * v[0] + u[1]
    az = -n ** 2 * r[2] + u[2]

    return np.concatenate((v, [ax, ay, az]))


# --- 积分求解 ---
t_span = (0, 10000)  # 仿真 10000 秒 (约 1.7 个轨道周期)
t_eval = np.linspace(0, 10000, 1000)

sol = solve_ivp(dynamics_only_potential, t_span, Y0, t_eval=t_eval, method='RK45')
# --- 图1: 3D 相对轨迹 ---
fig1 = plt.figure(figsize=(6, 5))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(sol.y[0], sol.y[1], sol.y[2], 'b-', linewidth=2, label='Chaser Trajectory')
ax1.plot(0, 0, 0, 'ro', markersize=10, label='Target (Origin)')
ax1.set_title('Relative Trajectory (Potential Term Only)\n(No Damping -> Oscillation)')
ax1.set_xlabel('X (Radial) [m]')
ax1.set_ylabel('Y (Tangential) [m]')
ax1.set_zlabel('Z (Normal) [m]')
ax1.legend()
ax1.grid(True)


# --- 图2: 相对距离 ---
fig2 = plt.figure(figsize=(6, 5))
dist = np.sqrt(sol.y[0] ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2)
plt.plot(sol.t, dist, 'g-', linewidth=2)
plt.title('Relative Distance vs Time')
plt.xlabel('Time [s]')
plt.ylabel('Distance [m]')
plt.grid(True)


# --- 图3: 势函数 ---
fig3 = plt.figure(figsize=(6, 5))
Vp = 0.5 * kp * (sol.y[0] ** 2 + sol.y[1] ** 2 + sol.y[2] ** 2)
plt.plot(sol.t, Vp, 'r-', linewidth=2)
plt.title('Potential Energy Vp vs Time')
plt.xlabel('Time [s]')
plt.ylabel('Vp')
plt.grid(True)

plt.show()