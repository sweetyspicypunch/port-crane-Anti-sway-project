import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81          # gravity (m/s^2)
L = 1.0           # pendulum length (m)
theta = np.radians(20)  # initial angle (radians)
omega = 0.0      # initial angular velocity (rad/s)
dt = 0.01         # time step (s)

# --- Anti-Sway 제어기 파라미터 ---
px_target = 0.0  # 피봇의 목표 위치 (m)

# PD 제어 이득 (이 값들을 조절하면서 성능을 튜닝할 수 있습니다)
Kp_x = 2.0      # 피봇 위치 비례 이득
Kd_x = 3.0      # 피봇 속도 미분 이득 (댐핑)
Kp_theta = 40.0 # 진자 각도 비례 이득
Kd_omega = 15.0 # 진자 각속도 미분 이득 (댐핑)
pa_limit = 15.0 # 최대 가속도 제한 (m/s^2)
# --- ------------------------ ---

# pivot
px, pv, pa = 0.0, 0.0, 0.0

def f(theta, omega, pa):
    dtheta = omega
    domega = -(g/L) * np.sin(theta) - (pa/L) * np.cos(theta)
    return dtheta, domega

def rk4_step(theta, omega, dt):
    # rk4_step은 전역변수 pa를 f 함수 내부에서 참조하므로 수정 불필요
    k1_theta, k1_omega = f(theta, omega, pa)
    k2_theta, k2_omega = f(theta + 0.5*dt*k1_theta, omega + 0.5*dt*k1_omega, pa)
    k3_theta, k3_omega = f(theta + 0.5*dt*k2_theta, omega + 0.5*dt*k2_omega, pa)
    k4_theta, k4_omega = f(theta + dt*k3_theta, omega + dt*k3_omega, pa)

    theta += (dt/6)*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    omega += (dt/6)*(k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    return theta, omega

def on_key(event):
    global px_target
    if event.key == 'right':
        px_target += 1.0  # 목표 위치 1.0m 오른쪽으로
    elif event.key == 'left':
        px_target -= 1.0  # 목표 위치 1.0m 왼쪽으로
    print(f"New Target Position: {px_target:.1f} m")


def update(frame):
    global theta, omega
    global px, pv, pa
    global px_target # 목표 위치 참조

    # --- Anti-Sway PD 컨트롤러 ---
    # 1. 피봇 위치/속도 제어 (목표: px -> px_target, pv -> 0)
    #    오차 = (목표값 - 현재값)
    a_pos = Kp_x * (px_target - px) + Kd_x * (0.0 - pv)

    # 2. 진자 각도/각속도 제어 (목표: theta -> 0, omega -> 0)
    #    EOM: d(omega) = ... - (pa/L)*cos(theta)
    #    - theta > 0 (오른쪽) 일 때 pa > 0 (오른쪽 가속) -> d(omega)는 음수 (복원력)
    #    - omega > 0 (오른쪽 스윙) 일 때 pa > 0 (오른쪽 가속) -> d(omega)는 음수 (댐핑)
    a_angle = Kp_theta * theta + Kd_omega * omega

    # 3. 총 가속도 = 위치제어 + 각도제어
    pa = a_pos + a_angle

    # 4. 가속도 제한 (너무 큰 값이 들어가지 않도록)
    pa = np.clip(pa, -pa_limit, pa_limit)
    # --- --------------------- ---

    # pa *= np.exp(-dt / tau) # <--- 기존 펄스 로직 삭제

    # 피봇의 다음 상태 계산
    px += pv * dt
    pv += pa * dt

    # 진자의 다음 상태 계산 (새로 계산된 'pa'가 f 함수에 적용됨)
    theta, omega = rk4_step(theta, omega, dt)
    
    x_bob = L * np.sin(theta)
    y_bob = -L * np.cos(theta)

    rod.set_data([px, px + x_bob], [0.0, y_bob])
    bob.set_data([px + x_bob], [y_bob])
    
    # 정보 텍스트 업데이트
    info_text.set_text(
        f"θ={np.degrees(theta): .1f} deg   ω={np.degrees(omega): .1f} deg/s\n"
        f"px={px: .2f} m     pv={pv: .2f} m/s\n"
        f"Target={px_target: .1f} m   pa={pa: .2f} m/s²"
    )
    return rod, bob, info_text

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)
rod,       = ax.plot([ ], [ ], lw=2)
bob,       = ax.plot([ ], [ ], 'o', ms=12)
# 텍스트가 여러 줄이 되므로 y 위치를 살짝 조정
info_text  = ax.text(-2.1, 1.35, "", fontsize=9, family="monospace") 
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.6, 1.8)
ax.grid(True)
ani = FuncAnimation(fig, update, frames=1000, interval=10, blit=True)
plt.show()
