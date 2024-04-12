import numpy as np


class KalmanFilter:
    def __init__(self, dim_state=1, dim_meas=1, process_noise=1e-5, measurement_noise=1e-5, estimated_error=1e-5):
        self.dim_state = dim_state  # 状态维度
        self.dim_meas = dim_meas    # 测量维度

        # 状态向量和协方差矩阵的初始化
        self.state = np.zeros((dim_state, 1))  # 状态向量
        self.covariance = np.eye(dim_state) * estimated_error  # 误差协方差矩阵

        # 卡尔曼滤波器的核心矩阵
        self.F = np.eye(dim_state)  # 状态转移矩阵
        self.H = np.eye(dim_meas, dim_state)  # 测量矩阵
        self.Q = np.eye(dim_state) * process_noise  # 过程噪声协方差
        self.R = np.eye(dim_meas) * measurement_noise  # 测量噪声协方差

    def predict(self):
        # 预测下一时刻的状态
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(self.F, np.dot(self.covariance, self.F.T)) + self.Q

    def update(self, measurement):
        # 测量更新
        Y = measurement.reshape(self.dim_meas, 1) - np.dot(self.H, self.state)
        S = np.dot(self.H, np.dot(self.covariance, self.H.T)) + self.R
        K = np.dot(self.covariance, np.dot(self.H.T, np.linalg.inv(S)))
        self.state = self.state + np.dot(K, Y)
        size = self.state.shape[0]
        self.covariance = (np.eye(size) - np.dot(K, self.H)) * self.covariance

        return self.state.ravel().item()  # 返回平坦化后的状态数组，便于观察


class ExtendedKalmanFilter:
    def __init__(self, process_noise_pos, process_noise_vel, measurement_noise, initial_state=[0, 0, 0, 0]):
        # 初始化状态向量和协方差矩阵
        self.state = np.array(initial_state, dtype=float)  # 初始状态 [x, vx, y, vy]
        self.covariance = np.eye(4) * 0.1  # 初始误差协方差

        # 定义状态转移矩阵
        self.F = np.array([
            [1, 1, 0, 0],  # 下一个 x = 当前 x + 当前 vx
            [0, 1, 0, 0],  # 下一个 vx = 当前 vx
            [0, 0, 1, 1],  # 下一个 y = 当前 y + 当前 vy
            [0, 0, 0, 1]   # 下一个 vy = 当前 vy
        ], dtype=float)

        # 定义测量矩阵
        self.H = np.array([
            [1, 0, 0, 0],  # 测量 x
            [0, 0, 1, 0]   # 测量 y
        ], dtype=float)

        # 定义过程噪声协方差和测量噪声协方差
        self.Q = np.eye(4) * np.array([process_noise_pos, process_noise_vel, process_noise_pos, process_noise_vel])
        self.R = np.eye(2) * measurement_noise

    def predict(self):
        # 预测下一状态
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q

    def update(self, measurement):
        # 确保 measurement 是列向量
        measurement = np.array(measurement).reshape(-1, 1)
        # 计算测量残差
        y = measurement - np.dot(self.H, self.state.reshape(-1, 1))  # 保证 state 也是列向量
        # 计算残差协方差
        S = np.dot(self.H, np.dot(self.covariance, self.H.T)) + self.R
        # 计算卡尔曼增益
        K = np.dot(self.covariance, np.dot(self.H.T, np.linalg.inv(S)))
        # 更新状态
        self.state = self.state.reshape(-1, 1) + np.dot(K, y)  # 使用列向量进行计算
        self.state = self.state.flatten()  # 将更新后的状态向量转换回 1D 数组
        # 更新误差协方差
        I = np.eye(self.state.shape[0])  # 使用状态向量的长度创建单位矩阵
        self.covariance = np.dot((I - np.dot(K, self.H)), self.covariance)
        # 输出当前估计的位置 x, y
        return self.state[0], self.state[2]
