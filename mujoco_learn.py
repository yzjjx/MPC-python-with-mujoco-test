# import mujoco
# from mujoco import viewer
# import numpy as np
#
# # model = mujoco.MjModel.from_xml_path('ori.xml')
# model = mujoco.MjModel.from_xml_path('cartpole.xml')
# data = mujoco.MjData(model)
#
# # 新版本的mujoco需要单独导入viewer
# mujoco.viewer.launch(model)

import mujoco
from mujoco import viewer
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path('cartpole.xml')
data = mujoco.MjData(model)

# 创建传感器数据记录器
sensor_data = []
timestamps = []

mujoco.viewer.launch(model)

def process_sensor_data(sensor_data):
    # 打印传感器数据
    print("Processed Sensor Data: ", sensor_data)

# 仿真循环
for i in range(1000):
    mujoco.mj_step(model, data)

    # 记录传感器数据和时间戳
    sensor_data.append(data.sensordata.copy())
    timestamps.append(data.time)

    # 实时数据处理
    if i % 100 == 0:
        process_sensor_data(data.sensordata)

# 转换为NumPy数组进行分析
sensor_array = np.array(sensor_data)
time_array = np.array(timestamps)

#代码实现：给mujoco一个激励，看四个传感器的变化

