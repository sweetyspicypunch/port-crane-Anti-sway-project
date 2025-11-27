import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.linalg import solve_continuous_are

# --- 1. 물리 엔진 (동일) ---
class CranePhysics3D:
    def __init__(self):
        self.M = 100.0; self.m = 10.5; self.g = 9.81; self.dt = 0.02
        self.state = np.zeros(8)
        self.L = 1.0
        self.Q_sub = np.diag([50.0, 1.0, 2500.0, 10.0]) 
        self.R_sub = np.array([[0.1]])
        self.K_sub = None

    def compute_gain_1d(self):
        L_safe = max(self.L, 0.1)
        A = np.array([[0,1,0,0], [0,0,self.m*self.g/self.M,0], [0,0,0,1], [0,0,-(self.M+self.m)*self.g/(self.M*L_safe),0]])
        B = np.array([[0], [1/self.M], [0], [-1/(self.M*L_safe)]])
        P = solve_continuous_are(A, B, self.Q_sub, self.R_sub)
        return np.linalg.inv(self.R_sub) @ B.T @ P

    def physics_step_1d(self, state_sub, force, L, L_dot):
        x, v, th, w = state_sub
        m, M, g = self.m, self.M, self.g
        sin_th, cos_th = np.sin(th), np.cos(th)
        denom = L * (M + m - m * cos_th**2)
        th_acc = (-force*cos_th - m*L*(w**2)*cos_th*sin_th - (M+m)*g*sin_th) / denom
        th_acc -= (2 * L_dot * w) / L 
        x_acc = (force + m*sin_th*(L*w**2 - g*cos_th)) / (M + m - m * cos_th**2)
        return np.array([v, x_acc, w, th_acc])

    def step(self, target_x, target_y, target_L):
        self.K_sub = self.compute_gain_1d()
        L_dot = (target_L - self.L) / self.dt
        self.L = target_L

        # --- X축 제어 및 RK4 적분 ---
        state_x = self.state[0:4]
        error_x = state_x - np.array([target_x, 0, 0, 0])
        u_x = -self.K_sub @ error_x
        force_x = np.clip(u_x[0], -25, 25)

        k1 = self.physics_step_1d(state_x, force_x, self.L, L_dot)
        k2 = self.physics_step_1d(state_x + 0.5 * self.dt * k1, force_x, self.L, L_dot)
        k3 = self.physics_step_1d(state_x + 0.5 * self.dt * k2, force_x, self.L, L_dot)
        k4 = self.physics_step_1d(state_x + self.dt * k3, force_x, self.L, L_dot)
        
        self.state[0:4] += (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # --- Y축 제어 및 RK4 적분 ---
        state_y = self.state[4:8]
        error_y = state_y - np.array([target_y, 0, 0, 0])
        u_y = -self.K_sub @ error_y
        force_y = np.clip(u_y[0], -25, 25)

        k1_y = self.physics_step_1d(state_y, force_y, self.L, L_dot)
        k2_y = self.physics_step_1d(state_y + 0.5 * self.dt * k1_y, force_y, self.L, L_dot)
        k3_y = self.physics_step_1d(state_y + 0.5 * self.dt * k2_y, force_y, self.L, L_dot)
        k4_y = self.physics_step_1d(state_y + self.dt * k3_y, force_y, self.L, L_dot)

        self.state[4:8] += (self.dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

# --- 2. 작업 관리자 (동일) ---
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

class TaskManagerPickAndPlace:
    def __init__(self):
        self.phase_idx = 0
        self.timer = 0.0
        self.stable_counter = 0.0
        self.TH_DIST = 0.05; self.TH_DEG = 0.1; self.TH_TIME = 0.8
        self.L_TRANSPORT = 1.0; self.L_WORK = 3.0
        self.POS_HOME = (0.0, 0.0); self.POS_TARGET = (2.0, 2.0)

        self.phases = [
            ("READY",      *self.POS_HOME, self.L_TRANSPORT, 1.0),
            ("PICK_DOWN",  *self.POS_HOME, self.L_WORK,      2.0),
            ("PICK_UP",    *self.POS_HOME, self.L_TRANSPORT, 2.0),
            ("MOVE_TGT",   *self.POS_TARGET, self.L_TRANSPORT, 4.0),
            ("STABILIZE",  *self.POS_TARGET, self.L_TRANSPORT, 0.0),
            ("PLACE_DOWN", *self.POS_TARGET, self.L_WORK,      2.0),
            ("PLACE_UP",   *self.POS_TARGET, self.L_TRANSPORT, 2.0),
            ("MOVE_HOME",  *self.POS_HOME, self.L_TRANSPORT, 4.0),
            ("STABILIZE",  *self.POS_HOME, self.L_TRANSPORT, 0.0),
            ("DONE",       *self.POS_HOME, self.L_TRANSPORT, 0.0)
        ]

    def update(self, dt, cur_x, cur_y, cur_deg_x, cur_deg_y, cur_v_total):
        if self.phase_idx >= len(self.phases): 
            return 0,0,1,"Mission Complete", "done"
            
        p_type, p_tx, p_ty, p_L, p_dur = self.phases[self.phase_idx]
        
        if p_type == "DONE":
            return p_tx, p_ty, p_L, "Mission Complete", "done"

        prev_x, prev_y, prev_L = 0,0,1
        if self.phase_idx > 0: _, prev_x, prev_y, prev_L, _ = self.phases[self.phase_idx-1]

        tgt_x, tgt_y, tgt_L = p_tx, p_ty, p_L
        msg = p_type; status = "normal"

        if "MOVE" in p_type:
            progress = self.timer / p_dur
            s = smooth_step(progress)
            tgt_x = prev_x + (p_tx - prev_x)*s
            tgt_y = prev_y + (p_ty - prev_y)*s
        elif "PICK" in p_type or "PLACE" in p_type:
            progress = self.timer / p_dur
            tgt_L = prev_L + (p_L - prev_L)*progress
        elif p_type == "STABILIZE":
            dist = np.sqrt((cur_x-p_tx)**2 + (cur_y-p_ty)**2)
            deg = max(abs(cur_deg_x), abs(cur_deg_y))
            if dist < self.TH_DIST and deg < self.TH_DEG and cur_v_total < 0.1:
                self.stable_counter += dt
                status = "holding"
                msg = f"HOLD {self.stable_counter:.1f}s"
                if self.stable_counter >= self.TH_TIME:
                    status = "safe"; self.next_phase()
            else:
                self.stable_counter = 0; status = "unstable"; msg = "STABILIZING"
        
        if p_type != "STABILIZE" and p_type != "DONE":
            self.timer += dt
            if self.timer > p_dur: self.next_phase()
            
        return tgt_x, tgt_y, tgt_L, msg, status

    def next_phase(self):
        self.phase_idx += 1; self.timer = 0; self.stable_counter = 0

# --- 3. 시각화 및 로깅 ---
sim = CranePhysics3D()
task = TaskManagerPickAndPlace()

history_time = []
history_deg_x = []
history_deg_y = []

sim_time = 0.0

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 1)

ax3d = fig.add_subplot(gs[0:2, :], projection='3d')
ax3d.set_title("3D Crane Control (Video Saving Mode)")
ax3d.set_xlim(-1, 4); ax3d.set_ylim(-1, 4); ax3d.set_zlim(-3.5, 0.5)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y')

ax_graph = fig.add_subplot(gs[2, :])
ax_graph.set_title("Sway Angle vs Time")
ax_graph.set_xlabel("Time (s)")
ax_graph.set_ylabel("Angle (deg)")
ax_graph.grid(True)
ax_graph.set_ylim(-3, 3)

line_sway_x, = ax_graph.plot([], [], 'r-', label='Sway X', lw=1.5)
line_sway_y, = ax_graph.plot([], [], 'b-', label='Sway Y', lw=1.5)
ax_graph.legend(loc='upper right')

cart, = ax3d.plot([], [], [], 's-', markersize=10, color='cyan', markeredgecolor='k')
cable, = ax3d.plot([], [], [], 'k-', lw=2)
load, = ax3d.plot([], [], [], 'o', markersize=8, color='orange', markeredgecolor='k')
shadow, = ax3d.plot([], [], [], 'o', color='gray', alpha=0.2)
target_pt, = ax3d.plot([], [], [], 'rx', markersize=10, markeredgewidth=2)

status_text = fig.text(0.05, 0.92, "", fontsize=12, fontweight='bold')
time_text = fig.text(0.85, 0.92, "Time: 0.0s", fontsize=12, family='monospace')

def animate(frame):
    global sim_time
    
    x, vx, th_x, wx = sim.state[0:4]
    y, vy, th_y, wy = sim.state[4:8]
    cur_v = np.sqrt(vx**2 + vy**2)
    deg_x, deg_y = np.degrees(th_x), np.degrees(th_y)
    
    tgt_x, tgt_y, tgt_L, msg, status = task.update(sim.dt, x, y, deg_x, deg_y, cur_v)
    
    sim.step(tgt_x, tgt_y, tgt_L)
    
    if status != "done":
        sim_time += sim.dt
        history_time.append(sim_time)
        history_deg_x.append(deg_x)
        history_deg_y.append(deg_y)
        
        line_sway_x.set_data(history_time, history_deg_x)
        line_sway_y.set_data(history_time, history_deg_y)
        
        if sim_time > 5:
            ax_graph.set_xlim(0, sim_time + 0.5)
        else:
            ax_graph.set_xlim(0, 5)
            
        time_text.set_text(f"Time: {sim_time:.1f}s")
    else:
        time_text.set_text(f"Time: {sim_time:.1f}s (STOPPED)")
        time_text.set_color("red")

    lx = x + sim.L * np.sin(th_x)
    ly = y + sim.L * np.sin(th_y)
    lz = -sim.L * np.cos(np.sqrt(th_x**2 + th_y**2))
    
    cart.set_data([x], [y]); cart.set_3d_properties([0])
    cable.set_data([x, lx], [y, ly]); cable.set_3d_properties([0, lz])
    load.set_data([lx], [ly]); load.set_3d_properties([lz])
    shadow.set_data([lx], [ly]); shadow.set_3d_properties([-3.5])
    target_pt.set_data([tgt_x], [tgt_y]); target_pt.set_3d_properties([0])

    colors = {'normal':'blue', 'unstable':'red', 'holding':'orange', 'safe':'green', 'done':'black'}
    status_text.set_text(f"Task: {msg}")
    status_text.set_color(colors.get(status, 'black'))
    
    return cart, cable, load, shadow, target_pt, line_sway_x, line_sway_y



# 프레임 수 계산
SAVE_FRAMES = 1600 

anim = FuncAnimation(fig, animate, frames=SAVE_FRAMES, interval=20, blit=False)




plt.show()
