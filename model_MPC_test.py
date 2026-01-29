import mujoco
import mujoco.viewer
import numpy as np
import time
import MPC_Matrices
import Prediction

M = 1.0
m = 0.1
l_rod = 0.55
J = 0.0084
b = 0.0005
g = 9.81
T = 0.002

Q_d = (M + m) * (m * l_rod ** 2 + J)
denominator = Q_d - m ** 2 * l_rod ** 2
a_1 = -b * (m * l_rod ** 2 + J) / denominator
a_2 = -m ** 2 * l_rod ** 2 * g / denominator
a_3 = m * l_rod * b / denominator
a_4 = m * g * l_rod * (M + m) / denominator
b_1 = (m * l_rod ** 2 + J) / denominator
b_2 = -m * l_rod / denominator

A_c = np.array([[0, 1, 0, 0], [0, a_1, a_2, 0], [0, 0, 0, 1], [0, a_3, a_4, 0]])
B_c = np.array([[0], [b_1], [0], [b_2]])
A = np.asmatrix(np.eye(4) + A_c * T)
B = np.asmatrix(B_c * T)
n, p = A.shape[0], B.shape[1]

Q = np.asmatrix(np.diag([1.0, 1.0, 100.0, 1.0]))

R = np.asmatrix([0.01])

F = 1 * np.eye(n)
F = np.asmatrix(F)
N = 100

[E, H] = MPC_Matrices.MPC_Matrices(A, B, Q, R, F, N)

H = np.asmatrix((H + H.T) / 2)

# 用这个函数来进行角度转换，因为在mujoco中，传感器传出来的角度是一直增加的
def angle_convert(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

if __name__ == "__main__":
    # 加载mujoco环境
    try:
        model = mujoco.MjModel.from_xml_path('cartpole.xml')
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"图形加载失败: {e}")
        exit()

    model.opt.timestep = T

    # 定义初始条件
    init_x = 0.0
    init_v = 0.0
    init_theta = 0.3
    init_omega = 0.0

    data.qpos[0] = init_x
    data.qpos[1] = init_theta
    data.qvel[0] = init_v
    data.qvel[1] = init_omega
    mujoco.mj_forward(model, data)
    print("初始状态设置完毕。")

    # 编译一次，后续会加速，加与不加没有太大影响
    dummy_x = np.array([0, 0, 0, 0], dtype=np.float64)
    Prediction.Prediction(dummy_x, E, H, N, p)
    print("Numba编译完成，开始仿真")

    render_rate = 2
    step_counter = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 指定相机位置
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # 指定使用第几个固定相机 (XML中定义的第一个相机索引为0)
        viewer.cam.fixedcamid = 0

        while viewer.is_running():
            loop_start = time.time()

            # 获取实时状态
            raw_theta = data.qpos[1]
            wrapped_theta = angle_convert(raw_theta)

            x_k = np.array([
                data.qpos[0],
                data.qvel[0],
                wrapped_theta,
                data.qvel[1]
            ])

            u_val = Prediction.Prediction(x_k, E, H, N, p)
            #计算20步打印一次数据
            if step_counter % 20 == 0:
                print(f"X: {x_k[0]:.2f}, Theta: {x_k[2]:.2f} -> Force: {-u_val:.2f}")

            # 将计算出来的力给ctrl
            data.ctrl[0] = -u_val

            # 用来获取状态信息
            mujoco.mj_step(model, data)

            step_counter += 1
            #每2步刷新一次界面
            if step_counter % render_rate == 0:
                viewer.sync()

            # 计算从开始到现在一次循环执行的时间
            elapsed = time.time() - loop_start
            if elapsed < T:
                time.sleep(T - elapsed)