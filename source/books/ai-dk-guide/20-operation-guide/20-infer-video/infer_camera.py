# coding=utf-8

import cv2  # 图片处理三方库，用于对图片进行前后处理
import numpy as np  # 用于对多维数组进行计算
import torch  # 深度学习运算框架，此处主要用来处理数据

from mindx.sdk import Tensor  # mxVision 中的 Tensor 数据结构
from mindx.sdk import base  # mxVision 推理接口

# 模型前后处理相关函数
from det_utils import get_labels_from_txt, letterbox, scale_coords, nms, draw_bbox

# jupyter 显示用
import ipywidgets as widgets
from IPython.display import display


# 变量初始化
base.mx_init()  # 初始化 mxVision 资源
DEVICE_ID = 0  # 设备id
model_path = "./yolov5s_bs1.om"  # 模型路径
# image_path = 'world_cup.jpg'  # 测试图片路径

# 利用手机ip摄像头
# url = 'rtsp://admin:password@192.168.0.102:8554/live'  # 这里需要替换为自己的链接
# cap = cv2.VideoCapture(url)


# 查找 USB camera 的 index 值
def find_camera_index():
    max_index_to_check = 10  # Maximum index to check for camera

    for index in range(max_index_to_check):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index

    # If no camera is found
    raise ValueError("No camera found.")


#
def img2bytes(image):
    """将图片转换为字节码"""
    return bytes(cv2.imencode(".jpg", image)[1])


# 获取摄像头
camera_index = find_camera_index()
cap = cv2.VideoCapture(camera_index)

# 获取保存视频相关变量
fps = 5  # 使用rtsp推流时，不能使用cap.get(cv2.CAP_PROP_FPS)来获取帧率，且由于延迟较高，手动指定帧率，可以根据实际情况调节
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

outfile = "video_result.mp4"
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(outfile, fourcc, fps, (video_width, video_height))

# jupyter 初始化视频显示界面
image_widget = widgets.Image(format="jpeg", width=video_width, height=video_height)
display(image_widget)

try:
    while cap.isOpened():  # 在摄像头打开的情况下循环执行
        ret, frame = cap.read()  # 此处 frame 为 bgr 格式图片

        # 数据前处理
        # img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读入图片
        # img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=[640, 640])  # 对图像进行缩放与填充，保持长宽比
        img, scale_ratio, pad_size = letterbox(
            frame, new_shape=[640, 640]
        )  # 对图像进行缩放与填充，保持长宽比
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.expand_dims(img, 0).astype(
            np.float32
        )  # 将形状转换为 channel first (1, 3, 640, 640)，即扩展第一维为 batchsize
        img = np.ascontiguousarray(img) / 255.0  # 转换为内存连续存储的数组
        img = Tensor(img)  # 将numpy转为转为Tensor类

        # 模型推理, 得到模型输出
        # 初始化 base.model 类
        model = base.model(modelPath=model_path, deviceId=DEVICE_ID)
        # 执行推理。输入数据类型：List[base.Tensor]， 返回模型推理输出的 List[base.Tensor]
        output = model.infer([img])[0]

        # 后处理
        output.to_host()  # 将 Tensor 数据转移到 Host 侧
        output = np.array(output)  # 将数据转为 numpy array 类型
        # 利用非极大值抑制处理模型输出，conf_thres 为置信度阈值，iou_thres 为iou阈值
        boxout = nms(torch.tensor(output), conf_thres=0.4, iou_thres=0.5)
        pred_all = boxout[0].numpy()  # 转换为numpy数组
        # 将推理结果缩放到原始图片大小
        scale_coords(
            [640, 640], pred_all[:, :4], frame.shape, ratio_pad=(scale_ratio, pad_size)
        )
        # 得到类别信息，返回序号与类别对应的字典
        labels_dict = get_labels_from_txt("./coco_names.txt")
        # 画出检测框、类别、概率
        img_dw = draw_bbox(pred_all, frame, (0, 255, 0), 2, labels_dict)

        # 将推理结果写入视频
        writer.write(img_dw)

        # 将推理结果显示在jupyter
        image_widget.value = img2bytes(img_dw)


except KeyboardInterrupt:
    cap.release()
    writer.release()
finally:
    cap.release()
    writer.release()
# 保存图片到文件
print("save infer result success")


# source /usr/local/Ascend/mxVision/set_env.sh
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# jupyter lab --ip=192.168.137.100 --allow-root --no-browser
