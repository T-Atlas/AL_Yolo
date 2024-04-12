import os
from pathlib import Path
import numpy as np
import torch
from models.common import DetectMultiBackend

from ultralytics.nn.autobackend import AutoBackend
import cv2

from mouse_driver.kaiman_filter import ExtendedKalmanFilter, KalmanFilter
from utils.general import (LOGGER, Profile, check_img_size, check_requirements, scale_boxes, strip_optimizer)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import time
from Capture import LoadScreen
from pynput.mouse import Listener
from mouse_driver.MouseMove import ghub_mouse_move
import pyautogui
from math import atan2
from pynput.mouse import Listener, Button
from mouse_driver.pid_control import PIDController,PIDControllerAdvanced
ROOT = os.getcwd()


class YOLOv5Detector:
    def __init__(
        self,
        weights='',
        data='',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300, #keep the default settings is enough, no nessary to use 1000
        device="cpu",
        view_img=False, #changed
        classes=None,
        agnostic_nms=False,
        augment=False,
        half=True,
        enemy_label=0, #add enemy label for future identification
        dnn=False
    ):
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.half = half
        self.dnn = dnn
        self.should_stop = False  # flag to stop
        self.enable_mouse_lock = False
        self.width, self.height = pyautogui.size()
        self.center = [self.width / 2, self.height / 2]
        self.size = 640
        self.offset = torch.tensor([self.size / 2, self.size / 2], device='cuda:0')
        self.mul = 0.4
        self.smooth = 0.25
        self.mouse_on_click = False
        self.showFPS = False
        self.listener = Listener(on_click=self.is_click)
        self.listener.start()
        self.enemy_label=enemy_label
        self.pid_enable = True
        self.pid_controller = PIDController(p_gain=0.2, i_gain=0.01, d_gain=0.05,output_limits=(-150, 150))
        self.target_history = []
        self.prediction_steps = 20  # 预测步长，根据需求调整
        self.predictor_type = 'none'  # 预测器类型，目前支持"none"、"linear"、"kalman"和"exkf"预测，效果都不太好，有待改进
        self.kf_x = KalmanFilter(process_noise=0.1, measurement_noise=0.1, estimated_error=0.1)
        self.kf_y = KalmanFilter(process_noise=0.1, measurement_noise=0.1, estimated_error=0.1)
        self.ekf = ExtendedKalmanFilter(process_noise_pos=0.1, process_noise_vel=0.1, measurement_noise=0.1)


    def is_click(self, x, y, button, pressed):
        if self.enable_mouse_lock:
            if button in [Button.left, Button.right]:
                if pressed:
                    self.mouse_on_click = True
                    print("鼠标锁定已开启")
                else:
                    self.mouse_on_click = False
                    print("鼠标锁定已关闭")

    def get_dis(self, vec): # must not null
        return (((vec[0] + vec[2] - self.size ) / 2) ** 2 + ((vec[1] + vec[3] - self.size) / 2) ** 2) ** (1 / 2)

    def lock_target(self, target):
        # 获得目标的当前中心坐标
        current_target_x = (target[0] + target[2] - self.size) / 2
        current_target_y = (target[1] + target[3] - self.size) / 2
        
        adjusted_target = [current_target_x, current_target_y]

        if self.pid_enable:
            move_rel_x, move_rel_y = [atan2(item, self.size) * self.size for item in adjusted_target]
            
            # 更新历史记录
            self.target_history.append((move_rel_x, move_rel_y))
            if len(self.target_history) > self.prediction_steps:
                self.target_history.pop(0)
            # 如果历史记录足够，进行位置预测
            if len(self.target_history) == self.prediction_steps and self.predictor_type != 'none' :
                move_rel_x, move_rel_y = self.predict_future_position()
                
            move_rel_x, move_rel_y = self.pid_controller.update(move_rel_x, move_rel_y)
            # 按理说要在合适的时机重置积分项，但是暂时没想好
        else:
            rel_target = [item * self.smooth for item in adjusted_target]
            # rel_target = [item * self.smooth for item in [(target[0] + target[2] - self.size) / 2, (target[1] + target[3] - self.size) / 2]]
            move_rel_x, move_rel_y = [atan2(item, self.size) * self.size for item in rel_target]
            # 更新历史记录
            self.target_history.append((move_rel_x, move_rel_y))
            if len(self.target_history) > self.prediction_steps:
                self.target_history.pop(0)
            # 如果历史记录足够，进行位置预测
            if len(self.target_history) == self.prediction_steps and self.predictor_type != 'none':
                move_rel_x, move_rel_y = self.predict_future_position()
            
        ghub_mouse_move(move_rel_x, move_rel_y)
        
    def predict_future_position(self):
        if self.predictor_type == 'linear':
            # 使用简单的线性预测模型
            deltas = [(self.target_history[i+1][0] - self.target_history[i][0],
                    self.target_history[i+1][1] - self.target_history[i][1]) for i in range(len(self.target_history) - 1)]
            # 计算平均变化量
            avg_delta_x = sum(delta[0] for delta in deltas) / len(deltas)
            avg_delta_y = sum(delta[1] for delta in deltas) / len(deltas)
            # 使用最后的位置和平均变化量预测下一位置
            last_x, last_y = self.target_history[-1]
            predicted_x = last_x + avg_delta_x
            predicted_y = last_y + avg_delta_y

        elif self.predictor_type == 'kalman':
            # Assuming self.kf_x and self.kf_y are instances of KalmanFilter for x and y respectively
            current_target_x, current_target_y = self.target_history[-1]
            self.kf_x.predict()
            self.kf_y.predict()
            predicted_x = self.kf_x.update(np.array([current_target_x]))
            predicted_y = self.kf_y.update(np.array([current_target_y]))
        
        elif self.predictor_type == 'exkf':
            current_target_x, current_target_y = self.target_history[-1]
            # You should combine x and y measurements into a single update step if possible
            measurement = np.array([current_target_x, current_target_y])  # Example with dummy velocity data
            self.ekf.predict()  # Predict the next state
            predicted_x, predicted_y = self.ekf.update(measurement)  # Update with the latest measurement Extract x and y positions
            
        else:
            #返回不预测的结果，有bug，会慢一拍
            predicted_x, predicted_y = self.target_history[-1]
            
        return predicted_x, predicted_y

    def run(self):
        # Load model
        device = select_device(0)
        
        imgsz = self.imgsz

        # Dataloader
        bs = 1  # batch_size
        
        dataset = LoadScreen()
        
        try:
            # since we use the ultralytics version, we need to change the import
            from ultralytics.utils.ops import non_max_suppression
            model = AutoBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
            stride, names, pt = model.stride, model.names, model.pt
            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        except TypeError as e:
            print("检测到旧版本的YOLOv5模型，正在尝试使用原始加载...")
            # two versions seems to have different results, I can't find a better way to solve this problem
            from utils.general import non_max_suppression
            model = DetectMultiBackend(self.weights, device=device, dnn=self.dnn, data=self.data, fp16=self.half)
            stride, names, pt = model.stride, model.names, model.pt
            # Run inference
            model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
            
        frame_cnt = 0
        that_time = 0

        for im, im0 in dataset: # main loop
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            # Inference
            pred = model(im, augment=self.augment, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            
            # Quit
            if self.should_stop:
                cv2.destroyAllWindows()
                break
            
            
            bound = pred[0].cpu().numpy()
            # print(bound)
            
            if self.enable_mouse_lock and len(bound) > 0:
                # chose target which is closest to center 
                target = bound[0]
                min_dis = self.get_dis(target)
                for vec in bound:
                    now_dis = self.get_dis(vec)
                    if now_dis < min_dis and vec[5] == self.enemy_label:
                        # only update target when it is enemy
                        target = vec
                        min_dis = now_dis
                
                if self.enable_mouse_lock and self.mouse_on_click and target[5] == self.enemy_label:
                    # only lock target when label is enemy and mouse is clicked
                    self.lock_target(target)
                
            # FPS calculate
            if self.showFPS:
                now_time = time.time()
                frame_cnt += 1
                duration_time = now_time - that_time
                fps = frame_cnt / duration_time
                if frame_cnt >= 100:
                    that_time = now_time
                    frame_cnt = 0

                print("Fps is ", fps)
            
        
    def work(self):
        check_requirements(exclude=('tensorboard', 'thop'))
        self.run()
        
    def stop(self):
        self.should_stop = True
    
    def start_mouse(self):
        self.enable_mouse_lock = True

    def stop_mouse(self):
        self.enable_mouse_lock = False
