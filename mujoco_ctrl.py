#代码实现：给mujoco一个激励，看四个传感器的变化
#with语句：用来处理文件操作、数据库连接等需要明确释放资源的场景
import mujoco
from mujoco import viewer
import numpy as np
import time
# 绘图
import matplotlib.pyplot as plt

# 加载模型
model = mujoco.MjModel.from_xml_path('cartpole.xml')
data = mujoco.MjData(model)

sensor_data = []
timestamps = []

def process_sensor_data(sensor_data):
    # 打印传感器数据
    print("Processed Sensor Data: ", sensor_data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    #指定相机
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # 2. 指定使用第几个固定相机 (XML中定义的第一个相机索引为 0)
    # 你的 XML 里只有一个 camera 标签，所以 ID 是 0
    viewer.cam.fixedcamid = 0


    # 只要窗口在运行，下面的代码就会一直执行
    while viewer.is_running():

        #给初始力为-1
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)

        # 记录传感器数据和时间戳
        sensor_data.append(data.sensordata.copy())
        timestamps.append(data.time)

        # 实时打印数据 如果改成每100步打印可以改为：xxxx% 100 == 0
        if len(sensor_data) :
            process_sensor_data(data.sensordata)

        viewer.sync()
        time.sleep(model.opt.timestep)

# 转换为NumPy数组进行分析
sensor_array = np.array(sensor_data)
time_array = np.array(timestamps)

# 利用pyplot绘图
xpoints = time_array
ypoints = sensor_array

plt.plot(xpoints, ypoints[:, 0],     label='slider_pos')#位置
plt.plot(xpoints, ypoints[:, 1],     label='slider_vel')#速度
plt.plot(xpoints, ypoints[:, 2]%6.28,label='hinge_pos')#角度
plt.plot(xpoints, ypoints[:, 3],     label='hinge_vel')#角速度

plt.xlabel('time (s)')
plt.ylabel('sensor value')
plt.legend()          # 添加图例
plt.grid(True)        # 加网格
plt.show()