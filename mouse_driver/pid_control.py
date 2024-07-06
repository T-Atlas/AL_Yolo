# pid_control.py
import time

def clamp(value, limits):
    """限制值在指定的范围内。"""
    lower, upper = limits
    return max(lower, min(value, upper)) if all(limits) else value

class PIDControllerAdvanced:
    """
    PID控制器类，用于管理和计算基于PID的运动控制。
    参考：https://github.com/m-lundberg/simple-pid
    似乎有bug，不要使用，auto_tune_pid方法未实现。
    """

    def __init__(self, p_gain, i_gain, d_gain, output_limits=(None, None), auto_tune=False):
        """
        初始化PID控制器，并设置比例增益、积分增益和微分增益。
        
        参数:
        p_gain (float): 比例增益。
        i_gain (float): 积分增益。
        d_gain (float): 微分增益。
        output_limits (tuple): 输出的上下限。(min_output, max_output)
        """
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.output_limits = output_limits

        self.integral_x = 0
        self.integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_time = time.time()

        # Auto-tune properties
        self.auto_tune = auto_tune
        self.Kc = None  # Critical gain
        self.Tu = None  # Oscillation period

    def reset(self):
        """重置积分和上次误差的值为零。"""
        self.integral_x = 0
        self.integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_time = time.time()
        
    def auto_tune_pid(self):
        # TODO: Placeholder for auto-tuning logic, to be implemented
        pass

    def update(self, error_x, error_y):
        """
        根据提供的误差更新PID计算。
        
        参数:
        error_x (float): X方向的误差。
        error_y (float): Y方向的误差。

        返回:
        tuple: 包含PID输出值 (output_x, output_y) 的元组。
        """
        if self.auto_tune:
            self.auto_tune_pid()
            
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = 1e-16

        self.integral_x += self.i_gain * error_x * dt
        self.integral_y += self.i_gain * error_y * dt

        self.integral_x = clamp(self.integral_x, self.output_limits)
        self.integral_y = clamp(self.integral_y, self.output_limits)

        derivative_x = (error_x - self.last_error_x) / dt
        derivative_y = (error_y - self.last_error_y) / dt

        output_x = self.p_gain * error_x + self.integral_x + self.d_gain * derivative_x
        output_y = self.p_gain * error_y + self.integral_y + self.d_gain * derivative_y

        output_x = clamp(output_x, self.output_limits)
        output_y = clamp(output_y, self.output_limits)

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_time = current_time

        return output_x, output_y


class PIDController:
    """ PID控制器类，用于管理和计算基于PID的运动控制。 """

    def __init__(self, p_gain, i_gain, d_gain, output_limits=(None, None), smooth_factor=0.25):
        """
        初始化PID控制器，并设置比例增益、积分增益和微分增益。
        
        参数:
        p_gain (float): 比例增益。
        i_gain (float): 积分增益。
        d_gain (float): 微分增益。
        output_limits (tuple): 输出的上下限。(min_output, max_output)
        smooth_factor (float): 平滑因子，用于平滑输出。
        """
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.output_limits = output_limits
        self.smooth_factor = smooth_factor

        self.integral_x = 0
        self.integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_time = time.time()
        self.previous_output_x = 0
        self.previous_output_y = 0

    def reset(self):
        """ 重置积分和上次误差的值为零。 """
        self.integral_x = 0
        self.integral_y = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.previous_output_x = 0
        self.previous_output_y = 0
        self.last_time = time.time()

    def update(self, error_x, error_y):
        """
        根据提供的误差更新PID计算。
        
        参数:
        error_x (float): X方向的误差。
        error_y (float): Y方向的误差。

        返回:
        tuple: 包含PID输出值 (output_x, output_y) 的元组。
        """
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0.0:
            dt = 1e-16

        self.integral_x += self.i_gain * error_x * dt
        self.integral_y += self.i_gain * error_y * dt

        self.integral_x = clamp(self.integral_x, self.output_limits)
        self.integral_y = clamp(self.integral_y, self.output_limits)

        derivative_x = (error_x - self.last_error_x) / dt
        derivative_y = (error_y - self.last_error_y) / dt

        output_x = (
            self.p_gain * error_x +
            self.integral_x +
            self.d_gain * derivative_x
        )
        output_y = (
            self.p_gain * error_y +
            self.integral_y +
            self.d_gain * derivative_y
        )

        output_x = clamp(output_x, self.output_limits)
        output_y = clamp(output_y, self.output_limits)

        # 平滑输出
        output_x = self.smooth_factor * self.previous_output_x + (1 - self.smooth_factor) * output_x
        output_y = self.smooth_factor * self.previous_output_y + (1 - self.smooth_factor) * output_y

        self.previous_output_x = output_x
        self.previous_output_y = output_y

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_time = current_time

        return output_x, output_y

if __name__ == "__main__":
    pid = PIDController(p_gain=0.1, i_gain=0.01, d_gain=0.05, output_limits=(-100, 100))
    error_x, error_y = 10, -5
    output = pid.update(error_x, error_y)
    print("PID输出:", output)
    
    # 设置目标点和初始位置
    setpoint_x, setpoint_y = 100, 50  # 目标位置
    current_x, current_y = 0, 0  # 初始位置
    pid = PIDController(p_gain=0.1, i_gain=0.01, d_gain=0.05, output_limits=(-100, 100))

    # 模拟控制过程进行10次迭代
    print("Starting position: (0, 0)")
    for i in range(100):
        error_x = setpoint_x - current_x
        error_y = setpoint_y - current_y

        output_x, output_y = pid.update(error_x, error_y)

        # 更新当前位置
        current_x += output_x
        current_y += output_y

        print(f"Step {i+1}, Output (X, Y): ({output_x:.2f}, {output_y:.2f}), Position: ({current_x:.2f}, {current_y:.2f})")
