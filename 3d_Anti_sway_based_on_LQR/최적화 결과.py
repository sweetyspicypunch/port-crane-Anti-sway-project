import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.linalg import solve_continuous_are
import matplotlib.animation as animation
import random
import sys

# --- 1. 물리 엔진 (Force 저장 기능 포함) ---
class CranePhysicsRigidBody:
    def __init__(self):
        self.M = 3800.0; self.m_hook = 500.0; self.m_cont = 30000.0 
        self.current_m = self.m_hook
        self.g = 9.81; self.dt = 0.02
        self.state = np.zeros(10) 
        self.L = 10.0; self.wind_velocity = np.zeros(2)
        
        self.I_zz_cont = self.m_cont * (2.4**2 + 6.0**2) / 12.0 
        self.I_zz_hook = self.m_hook * (0.5**2 + 0.5**2) / 12.0
        self.current_I = self.I_zz_hook

        self.Q_sub = np.diag([500.0, 10.0, 30000000.0, 5000000.0]) 
        self.R_sub = np.array([[5.3*1e-6]]) 
        self.K_sub = None
        
        self.Kp_skew = 2000.0; self.Kd_skew = 1000.0
        self.rho = 1.225
        self.Area_Side_Cont = 20.0 * 2.6; self.Area_Front_Cont = 2.4 * 2.6; self.Area_Block = 1.5
        
        self.last_force = 0.0 

    def set_payload_attached(self, is_attached):
        if is_attached:
            self.current_m = self.m_hook + self.m_cont
            self.current_I = self.I_zz_cont
            self.Kp_skew = 150000.0; self.Kd_skew = 80000.0 
        else:
            self.current_m = self.m_hook
            self.current_I = self.I_zz_hook
            self.Kp_skew = 3000.0; self.Kd_skew = 1500.0

    def get_aerodynamic_wrench(self, v_wind, psi):
        if np.isnan(psi) or np.abs(psi) > 100: psi = 0.0
        if self.current_m > self.m_hook:
            area_side = self.Area_Side_Cont; area_front = self.Area_Front_Cont; cd = 1.5
            cp_offset = np.random.uniform(-0.5, 0.5) 
        else:
            area_wire = 0.05 * self.L; area_side = area_wire + self.Area_Block
            area_front = area_wire + self.Area_Block; cd = 1.2
            cp_offset = np.random.uniform(-0.1, 0.1)

        Fx = 0.5 * self.rho * area_side * cd * v_wind[0] * abs(v_wind[0])
        Fy = 0.5 * self.rho * area_front * cd * v_wind[1] * abs(v_wind[1])
        Torque_Z = Fx * cp_offset - Fy * cp_offset 
        return np.array([Fx, Fy]), Torque_Z

    def compute_gain_1d(self):
        L_safe = max(self.L, 1.0); m = self.current_m 
        A = np.array([[0, 1, 0, 0], [0, 0, m*self.g/self.M, 0], [0, 0, 0, 1], [0, 0, -(self.M+m)*self.g/(self.M*L_safe), 0]])
        B = np.array([[0], [1/self.M], [0], [-1/(self.M*L_safe)]])
        P = solve_continuous_are(A, B, self.Q_sub, self.R_sub)
        return np.linalg.inv(self.R_sub) @ B.T @ P

    def step(self, target_x, target_y, target_L):
        self.K_sub = self.compute_gain_1d()
        L_dot = (target_L - self.L) / self.dt; self.L = target_L
        psi, w_psi = self.state[8], self.state[9]
        if np.isnan(psi): psi = 0.0; self.state[8] = 0.0
        if np.isnan(w_psi): w_psi = 0.0; self.state[9] = 0.0

        force_wind, torque_wind = self.get_aerodynamic_wrench(self.wind_velocity, psi)
        MAX_FORCE = 400000.0
        
        state_x = self.state[0:4]; error_x = state_x - np.array([target_x, 0, 0, 0])
        u_x = -self.K_sub @ error_x; force_x = np.clip(u_x[0], -MAX_FORCE, MAX_FORCE)
        
        state_y = self.state[4:8]; error_y = state_y - np.array([target_y, 0, 0, 0])
        u_y = -self.K_sub @ error_y; force_y = np.clip(u_y[0], -MAX_FORCE, MAX_FORCE)

        # 모터가 X방향, Y방향으로 쓴 힘의 총합(크기)
        self.last_force = np.sqrt(force_x**2 + force_y**2)

        def phys_x(s, f, w_f): return self.physics_step_1d(s, f, self.L, L_dot, w_f)
        self.state[0:4] += self.rk4(phys_x, self.state[0:4], force_x, force_wind[0])
        def phys_y(s, f, w_f): return self.physics_step_1d(s, f, self.L, L_dot, w_f)
        self.state[4:8] += self.rk4(phys_y, self.state[4:8], force_y, force_wind[1])

        control_torque = -self.Kp_skew * psi - self.Kd_skew * w_psi
        max_torque = 200000.0 if self.current_m > self.m_hook else 5000.0
        control_torque = np.clip(control_torque, -max_torque, max_torque)
        
        total_torque = torque_wind + control_torque
        alpha_psi = total_torque / self.current_I
        self.state[9] += alpha_psi * self.dt; self.state[9] *= 0.99 
        self.state[8] += self.state[9] * self.dt

    def physics_step_1d(self, state, force_ctrl, L, L_dot, force_wind):
        _, v, th, w = state; m, M, g = self.current_m, self.M, self.g
        sin_th, cos_th = np.sin(th), np.cos(th); denom = L * (M + m - m * cos_th**2)
        th_acc = (-force_ctrl*cos_th + force_wind*cos_th - m*L*(w**2)*cos_th*sin_th - (M+m)*g*sin_th) / denom
        th_acc -= (2 * L_dot * w) / L 
        x_acc = (force_ctrl + m*sin_th*(L*w**2 - g*cos_th)) / (M + m - m * cos_th**2)
        return np.array([v, x_acc, w, th_acc])

    def rk4(self, func, state, f_ctrl, f_wind):
        k1 = func(state, f_ctrl, f_wind)
        k2 = func(state + 0.5*self.dt*k1, f_ctrl, f_wind)
        k3 = func(state + 0.5*self.dt*k2, f_ctrl, f_wind)
        k4 = func(state + self.dt*k3, f_ctrl, f_wind)
        return (self.dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# --- 2. 작업 관리자 ---
class TaskManager:
    def __init__(self):
        self.phase_idx = 0; self.timer = 0.0; self.stable_counter = 0.0
        self.L_HIGH = 10.0
        self.L_LOW = 22.4 
        
        self.POS_START = (0.0, 0.0); self.POS_END = (15.0, 5.0)
        self.WIND_LIMIT = 22.0; self.is_danger_mode = False; self.frozen_state = None
        self.is_payload_attached = False; self.container_pos = list(self.POS_START)
        
        self.HARD_LIMIT_SKEW = 6.0; self.HARD_LIMIT_SWAY = 10.0  
        self.LANDING_TOLERANCE_SKEW = 1.5; self.LANDING_TOLERANCE_SWAY = 1.0 

        self.phases = [
            ("INIT_READY", *self.POS_START, self.L_HIGH, 2.0),
            ("LOWER_TO_PICK", *self.POS_START, self.L_LOW, 5.0),
            ("LATCHING", *self.POS_START, self.L_LOW, 1.5),
            ("HOIST_UP", *self.POS_START, self.L_HIGH, 5.0),
            ("MOVE_TO_TGT", *self.POS_END, self.L_HIGH, 12.0),
            ("STABILIZE", *self.POS_END, self.L_HIGH, 3.0),
            ("LOWER_TO_PLACE", *self.POS_END, self.L_LOW, 5.0),
            ("UNLATCHING", *self.POS_END, self.L_LOW, 1.5),
            ("HOIST_EMPTY", *self.POS_END, self.L_HIGH, 4.0),
            ("RETURN_HOME", *self.POS_START, self.L_HIGH, 10.0),
            ("DONE", *self.POS_START, self.L_HIGH, 0.0)
        ]

    def update(self, dt, cur_x, cur_y, cur_L, cur_deg_x, cur_deg_y, cur_skew, cur_v, wind_spd):
        if self.phase_idx >= len(self.phases): 
            return 0,0,10,"Mission Complete", "done"
        
        rad_x = np.radians(cur_deg_x); rad_y = np.radians(cur_deg_y)
        sway_angle = np.sqrt(cur_deg_x**2 + cur_deg_y**2)

        if wind_spd > self.WIND_LIMIT or abs(cur_skew) > self.HARD_LIMIT_SKEW or sway_angle > self.HARD_LIMIT_SWAY:
            if not self.is_danger_mode:
                payload_x = cur_x + cur_L * np.sin(rad_x)
                payload_y = cur_y + cur_L * np.sin(rad_y)
                self.frozen_state = (payload_x, payload_y, cur_L) 
                self.is_danger_mode = True
            reason = "WIND" if wind_spd > self.WIND_LIMIT else "ANGLE LIMIT"
            return self.frozen_state[0], self.frozen_state[1], self.frozen_state[2], f"EMERGENCY: {reason}", "danger"
        else: 
            self.is_danger_mode = False

        p_name, p_tx, p_ty, p_L, p_dur = self.phases[self.phase_idx]
        
        if p_name == "STABILIZE":
            self.timer += dt 
            is_stable = (sway_angle < self.LANDING_TOLERANCE_SWAY and abs(cur_skew) < self.LANDING_TOLERANCE_SKEW and cur_v < 0.1)
            
            if is_stable:
                self.stable_counter += dt
                msg = f"Stabilizing... {self.stable_counter:.1f}s"
                status = "holding"
                if self.stable_counter > 2.0: self.next_phase()
            else:
                self.stable_counter = 0
                msg = f"Wait.. Sway:{sway_angle:.1f} Skew:{abs(cur_skew):.1f}"
                status = "unstable"
            
            if self.timer > 8.0:
                print(f"!!! STABILIZE TIMEOUT (Sway: {sway_angle:.2f}) - Forcing Next Phase !!!")
                self.next_phase()
        else:
            status = "normal"
            msg = f"{p_name}"
            
            if self.timer == 0.0:
                if p_name == "LATCHING": 
                    self.is_payload_attached = True
                    print(">>> LATCHED: Payload Attached")
                elif p_name == "UNLATCHING": 
                    self.is_payload_attached = False
                    self.container_pos = [cur_x + cur_L*np.sin(rad_x), cur_y + cur_L*np.sin(rad_y)]
                    print(">>> UNLATCHED: Payload Released")
            
            self.timer += dt 
            
            if self.timer >= p_dur: self.next_phase()

        if self.phase_idx > 0: _, prev_tx, prev_ty, prev_L, _ = self.phases[self.phase_idx-1]
        else: prev_tx, prev_ty, prev_L = p_tx, p_ty, p_L

        progress = np.clip(self.timer / p_dur, 0.0, 1.0) if p_dur > 0 else 0.0
        s = progress * progress * (3 - 2 * progress)
        
        tgt_x, tgt_y, tgt_L = p_tx, p_ty, p_L
        if "MOVE" in p_name or "RETURN" in p_name:
            tgt_x = prev_tx + (p_tx - prev_tx)*s; tgt_y = prev_ty + (p_ty - prev_ty)*s
        elif "LOWER" in p_name or "HOIST" in p_name:
            tgt_L = prev_L + (p_L - prev_L)*s
        
        return tgt_x, tgt_y, tgt_L, msg, status

    def next_phase(self): self.phase_idx += 1; self.timer = 0; self.stable_counter = 0


# --- 3. WindManager ---
class WindManager:
    def __init__(self, min_speed=1.0, max_speed=15.0): 
        self.current_vel = np.array([0.0, 0.0]); self.target_vel = np.array([0.0, 0.0])
        self.gust_timer = 0.0; self.min_spd = min_speed; self.max_spd = max_speed
    def update(self, dt):
        if self.gust_timer <= 0:
            if random.random() < 0.02: 
                speed = np.random.uniform(self.min_spd, self.max_spd)
                angle = np.random.uniform(0, 2 * np.pi)
                self.target_vel = np.array([speed * np.cos(angle), speed * np.sin(angle)])
                self.gust_timer = 3.0 
        else:
            self.gust_timer -= dt
            if self.gust_timer <= 0: self.target_vel = np.zeros(2) 
        self.current_vel = self.current_vel * 0.92 + self.target_vel * 0.08
        return self.current_vel + np.random.normal(0, 0.5, 2)

# --- 4. 시각화 및 실행 ---
sim = CranePhysicsRigidBody(); task = TaskManager(); wind = WindManager()

# 애니메이션용 Figure (기존과 동일)
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.95, wspace=0.35, hspace=0.3)
gs = fig.add_gridspec(3, 5) 

ax3d = fig.add_subplot(gs[0:2, :], projection='3d')
ax3d.set_title("Crane Cycle: Pick -> Move -> Place -> Return")
ax3d.set_xlim(-10, 25); ax3d.set_ylim(-10, 15); ax3d.set_zlim(-30, 0)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y')

ax_len = fig.add_subplot(gs[2, 0]); ax_len.set_title("Cable Length")
ax_len.set_ylim(0, 30); ax_len.set_xlim(0, 10); ax_len.invert_yaxis()
l_len, = ax_len.plot([], [], 'g-', lw=2)

ax_sway = fig.add_subplot(gs[2, 1]); ax_wind = ax_sway.twinx()
ax_sway.set_title("Sway & Wind"); ax_sway.set_ylim(0, 8); ax_sway.set_xlim(0, 10)
l_sway, = ax_sway.plot([], [], 'r-', lw=2, label='Sway')
l_wind, = ax_wind.plot([], [], 'b--', alpha=0.5, label='Wind')
ax_wind.set_ylim(0, 25); ax_wind.axhline(22, color='red', ls=':')

ax_skew = fig.add_subplot(gs[2, 2]); ax_skew.set_title("Skew Angle", color='purple') 
ax_skew.set_ylim(-15, 15); ax_skew.set_xlim(0, 10)
l_skew, = ax_skew.plot([], [], 'purple', lw=2)
ax_skew.axhline(0, color='k', lw=1, ls='-')

ax_err = fig.add_subplot(gs[2, 3]); ax_err.set_title("Payload Sway Offset (m)", color='black')
ax_err.set_xlim(0, 10); ax_err.set_ylim(0, 3.0); ax_err.grid(True)
l_err, = ax_err.plot([], [], 'k-', lw=2, label='Offset')

ax_force = fig.add_subplot(gs[2, 4]); ax_force.set_title("Motor Control Force (N)", color='darkorange')
ax_force.set_xlim(0, 10); 
ax_force.set_ylim(0, 10000) 
ax_force.grid(True, which="both", ls="--", alpha=0.4)
l_force, = ax_force.plot([], [], '-', color='darkorange', lw=1.5, label='Force')

cart, = ax3d.plot([], [], [], 's-', markersize=10, color='black')
target_pt, = ax3d.plot([], [], [], 'rx', markersize=12, markeredgewidth=2.5)
error_line, = ax3d.plot([], [], [], 'r:', alpha=0.6)
cables = [ax3d.plot([], [], [], 'k-', lw=0.5)[0] for _ in range(4)] 
payload_lines = [ax3d.plot([], [], [], '-', lw=2)[0] for _ in range(12)]
spreader_lines = [ax3d.plot([], [], [], 'y-', lw=3)[0] for _ in range(12)]

shadow, = ax3d.plot([], [], [], 'gray', alpha=0.3)
xx, yy = np.meshgrid(np.linspace(-10, 25, 10), np.linspace(-10, 15, 10))
zz = np.full_like(xx, -25.0)
ax3d.plot_surface(xx, yy, zz, alpha=0.1, color='green')
ax3d.plot([task.POS_START[0]], [task.POS_START[1]], [-25], 'kx') 
ax3d.plot([task.POS_END[0]], [task.POS_END[1]], [-25], 'bx')        

txt_stat = fig.text(0.05, 0.92, "Init", fontsize=14, fontweight='bold')
txt_info = fig.text(0.70, 0.92, "", fontsize=12)

# [데이터 저장소 분리]
# 1. 애니메이션용 (최근 200개 데이터만 유지)
hist_t, hist_L, hist_sway, hist_wind, hist_skew, hist_err, hist_force = [], [], [], [], [], [], []
# 2. 전체 기록용 (처음부터 끝까지 모두 저장)
all_t, all_L, all_sway, all_wind, all_skew, all_err, all_force = [], [], [], [], [], [], []

global_time = 0.0

def get_box_corners(cx, cy, cz, th_x, th_y, psi, dx, dy, dz):
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy,  dz], [dx, -dy,  dz], [dx, dy,  dz], [-dx, dy,  dz]
    ])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(th_y), 0, np.sin(th_y)], [0, 1, 0], [-np.sin(th_y), 0, np.cos(th_y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(th_x), -np.sin(th_x)], [0, np.sin(th_x), np.cos(th_x)]])
    R_total = Rx @ Ry @ Rz 
    rot_corners = (R_total @ corners.T).T
    return rot_corners + [cx, cy, cz], R_total

# [고속 렌더링 설정]
STEPS_PER_FRAME = 5 
TOTAL_SIM_TIME = 60.0 
FPS = 15
TOTAL_FRAMES = int(TOTAL_SIM_TIME / (sim.dt * STEPS_PER_FRAME))

def animate_fast(frame):
    global global_time
    
    for _ in range(STEPS_PER_FRAME):
        w_vel = wind.update(sim.dt); sim.wind_velocity = w_vel
        x, vx, th_x, wx = sim.state[0:4]
        y, vy, th_y, wy = sim.state[4:8]
        psi, w_psi = sim.state[8], sim.state[9]
        v_tot = np.sqrt(vx**2 + vy**2)
        
        tgt_x, tgt_y, tgt_L, msg, stat = task.update(
            sim.dt, x, y, sim.L, np.degrees(th_x), np.degrees(th_y), np.degrees(psi), v_tot, np.linalg.norm(w_vel)
        )
        sim.set_payload_attached(task.is_payload_attached)
        sim.step(tgt_x, tgt_y, tgt_L)
        global_time += sim.dt
        
        # [전체 데이터 저장] 프레임 내부 스텝마다 저장하면 너무 많으니 여기선 패스하고 프레임 단위로 저장하거나,
        # 정밀도를 위해 여기서 저장해도 됨. 메모리 절약을 위해 아래쪽(프레임 단위)에서 저장.
    
    x, _, th_x, _ = sim.state[0:4]
    y, _, th_y, _ = sim.state[4:8]
    psi = sim.state[8]
    
    # [데이터 수집]
    curr_sway = np.sqrt(np.degrees(th_x)**2 + np.degrees(th_y)**2)
    curr_wind = np.linalg.norm(sim.wind_velocity)
    curr_skew = np.degrees(psi)
    
    hx = x + sim.L * np.sin(th_x)
    hy = y + sim.L * np.sin(th_y)
    curr_err = np.sqrt((x - hx)**2 + (y - hy)**2)
    
    # 1. 애니메이션용 (Rolling Window)
    hist_t.append(global_time); hist_L.append(sim.L)
    hist_sway.append(curr_sway); hist_wind.append(curr_wind)
    hist_skew.append(curr_skew); hist_err.append(curr_err)
    hist_force.append(sim.last_force)
    
    if len(hist_t) > 200: 
        for lst in [hist_t, hist_L, hist_sway, hist_wind, hist_skew, hist_err, hist_force]: lst.pop(0)

    # 2. 전체 기록용 (Accumulation)
    all_t.append(global_time); all_L.append(sim.L)
    all_sway.append(curr_sway); all_wind.append(curr_wind)
    all_skew.append(curr_skew); all_err.append(curr_err)
    all_force.append(sim.last_force)

    # [화면 갱신]
    hz = -sim.L * np.cos(np.sqrt(th_x**2 + th_y**2))

    if frame % 10 == 0:
        prog = (frame / TOTAL_FRAMES) * 100
        sys.stdout.write(f"\rProgress: {prog:.1f}% | Time: {global_time:.1f}s | {msg}")
        sys.stdout.flush()

    cart.set_data([x], [y]); cart.set_3d_properties([0])
    target_pt.set_data([tgt_x], [tgt_y]); target_pt.set_3d_properties([0])
    
    spreader_corners, _ = get_box_corners(hx, hy, hz, th_x, th_y, psi, 1.2, 3.0, 0.1)
    trolley_w, trolley_l = 1.0, 1.0 
    t_corners = np.array([
        [x-trolley_w, y-trolley_l, 0], [x+trolley_w, y-trolley_l, 0],
        [x+trolley_w, y+trolley_l, 0], [x-trolley_w, y+trolley_l, 0]
    ])
    s_top_corners = spreader_corners[4:8]
    for k in range(4):
        cables[k].set_data([t_corners[k,0], s_top_corners[k,0]], [t_corners[k,1], s_top_corners[k,1]])
        cables[k].set_3d_properties([t_corners[k,2], s_top_corners[k,2]])
        
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    for line, (i, j) in zip(spreader_lines, edges):
        line.set_data([spreader_corners[i,0], spreader_corners[j,0]], [spreader_corners[i,1], spreader_corners[j,1]])
        line.set_3d_properties([spreader_corners[i,2], spreader_corners[j,2]])

    if task.is_payload_attached:
        cont_corners, _ = get_box_corners(hx, hy, hz-1.4, th_x, th_y, psi, 1.2, 3.0, 1.3)
        box_color = 'red'
    else:
        cx, cy = task.container_pos
        cont_corners, _ = get_box_corners(cx, cy, -23.7, 0, 0, 0, 1.2, 3.0, 1.3)
        box_color = 'gray'

    for line, (i, j) in zip(payload_lines, edges):
        line.set_data([cont_corners[i,0], cont_corners[j,0]], [cont_corners[i,1], cont_corners[j,1]])
        line.set_3d_properties([cont_corners[i,2], cont_corners[j,2]])
        line.set_color(box_color)

    shadow.set_data([x, hx], [y, hy]); shadow.set_3d_properties([-25, -25])
    error_line.set_data([x, tgt_x], [y, tgt_y]); error_line.set_3d_properties([0, 0])
    
    # 애니메이션 그래프 업데이트 (Rolling Window)
    l_len.set_data(hist_t, hist_L); l_sway.set_data(hist_t, hist_sway)
    l_wind.set_data(hist_t, hist_wind); l_skew.set_data(hist_t, hist_skew)
    l_err.set_data(hist_t, hist_err); l_force.set_data(hist_t, hist_force)
    
    ranges = (max(0, global_time-10), max(10, global_time))
    for ax in [ax_len, ax_sway, ax_skew, ax_err, ax_force]: ax.set_xlim(ranges)
    
    if len(hist_force) > 0:
        f_max = max(hist_force)
        ax_force.set_ylim(0, max(100, f_max * 1.2))

    txt_stat.set_text(msg); txt_stat.set_color('red' if stat=='danger' else 'blue')
    txt_info.set_text(f"Time: {global_time:.1f}s\nPhase: {msg}")
    
    return [cart, target_pt, error_line, shadow, l_err, l_force] + cables + payload_lines + spreader_lines

print(f"전체 기록 모드 렌더링 시작! (총 {TOTAL_FRAMES} 프레임)")

# 1. 애니메이션 생성 및 비디오 저장 (이 과정에서 시뮬레이션이 진행되며 all_ 리스트에 데이터가 쌓임)
anim = FuncAnimation(fig, animate_fast, frames=TOTAL_FRAMES, interval=1, blit=False, repeat=False)
writer = animation.FFMpegWriter(fps=FPS, bitrate=4000)
anim.save("MK13_Simulation_Video_최종.mp4", writer=writer, dpi=80) 
plt.close(fig) # 애니메이션 창 닫기

# 2. [핵심] 전체 데이터(t=0 ~ end)를 사용한 최종 결과 그래프 생성
print("\n>>> 최종 전체 그래프(2D) 생성 중...")

fig_final = plt.figure(figsize=(15, 12))
fig_final.suptitle(f"Simulation Result Summary (Total Time: {global_time:.1f}s)", fontsize=16, fontweight='bold')
# 레이아웃: 3행 2열
# [Length] [Force]
# [Sway]   [Skew]
# [Offset] [Empty/Legend]

ax_f1 = fig_final.add_subplot(3, 2, 1)
ax_f1.plot(all_t, all_L, 'g-')
ax_f1.set_title("Cable Length (m)"); ax_f1.grid(True); ax_f1.invert_yaxis()

ax_f2 = fig_final.add_subplot(3, 2, 2)
ax_f2.plot(all_t, all_force, color='darkorange')
ax_f2.set_title("Motor Control Force (N)"); ax_f2.grid(True)
ax_f2.set_ylim(0, 6000) # [수정됨] Force Limit 6000

ax_f3 = fig_final.add_subplot(3, 2, 3)
ax_f3_wind = ax_f3.twinx()
ax_f3.plot(all_t, all_sway, 'r-', label='Sway')
ax_f3_wind.plot(all_t, all_wind, 'b--', alpha=0.3, label='Wind')
ax_f3_wind.axhline(22, color='red', ls=':', alpha=0.5)
ax_f3.set_title("Sway (deg) & Wind (m/s)"); ax_f3.grid(True)
# 범례 통합
lines, labels = ax_f3.get_legend_handles_labels()
lines2, labels2 = ax_f3_wind.get_legend_handles_labels()
ax_f3.legend(lines + lines2, labels + labels2, loc='upper right')

ax_f4 = fig_final.add_subplot(3, 2, 4)
ax_f4.plot(all_t, all_skew, color='purple')
ax_f4.set_title("Skew Angle (deg)"); ax_f4.grid(True)
ax_f4.axhline(0, color='k', ls='-', alpha=0.3)

ax_f5 = fig_final.add_subplot(3, 2, 5)
ax_f5.plot(all_t, all_err, 'k-')
ax_f5.set_title("Payload Position Error (m)"); ax_f5.grid(True)
ax_f5.set_xlabel("Time (s)")
ax_f5.set_ylim(0, 0.35) # [수정됨] Error Limit 0.35

# 레이아웃 조정 및 저장
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig_final.savefig("MK13_Full_Simulation_Graphs_최종.png", dpi=150)
plt.close(fig_final)

print(">>> 모든 저장 완료!")
print("1. 비디오: MK13_Simulation_Video.mp4 (움직이는 창)")
print("2. 그래프: MK13_Full_Simulation_Graphs.png (t=0 ~ 끝 전체 데이터)")
