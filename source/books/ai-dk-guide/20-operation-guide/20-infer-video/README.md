# 推理-视频

> <small>（更新：25-06-17 | 发布：25-06-16）</small>

---

使用昇腾提供的MindSDK，编写推理程序，可使用NPU算力做推理。以下样例参考了《目标检测应用样例开发介绍（Python）》[^object_detect_python] 以及《目标检测样例》[^object_detect_example]。

[^object_detect_python]: 昇腾Atlas 200I DK A2 开发者套件官网文档，[目标检测应用样例开发介绍（Python）](https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/Getting%20Started%20with%20Application%20Development/gswmsad/gswmsad_0002.html)

[^object_detect_example]: 昇腾Atlas 200I DK A2 开发者套件官网文档，[目标检测样例](https://www.hiascend.com/document/detail/zh/Atlas200IDKA2DeveloperKit/23.0.RC2/Getting%20Started%20with%20Application%20Development/paqe/paqe_0003.html)

## 样例代码
```python
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
```
## 运行样例

### SSH root 账号登录开发板
在本地计算机终端上（比如 CMD 或 PowerShell 或其他 SSH程序）执行 `ssh ssh root@192.168.137.100`命令，以 `root` 账号登录开发板。

```bash
~ % ssh root@192.168.137.100
root@192.168.137.100's password: 
    _                                _             _               _     _  _   
   / \    ___   ___  ___  _ __    __| |         __| |  ___ __   __| | __(_)| |_ 
  / _ \  / __| / __|/ _ \| '_ \  / _` | _____  / _` | / _ \\ \ / /| |/ /| || __|
 / ___ \ \__ \| (__|  __/| | | || (_| ||_____|| (_| ||  __/ \ V / |   < | || |_ 
/_/   \_\|___/ \___|\___||_| |_| \__,_|        \__,_| \___|  \_/  |_|\_\|_| \__|
                                                                                
Welcome to Atlas 200I DK A2
This system is based on Ubuntu 22.04 LTS (GNU/Linux 5.10.0+ aarch64)
```
### 新建目录放程序
登录开发板后，可以新建一个目录存放样例程序，比如 `/root/ailab/camera`
```bash
(base) root@davinci-mini:~# mkdir -p ailab/camera
```
### 复制或上传相关程序到目录中

复制或上传相关程序到目录`/root/ailab/camera`中：
- infer_camera.py。
  - 点击[下载](./infer_camera.py)；或新建 `infer_camera.py`，复制上述样例程序代码到 `infer_camera.py`。
  - 然后在本地计算机终端上执行 `scp infer_camera.py root@192.168.137.100:/root/ailab/camera` 上传到开发板。
- coco_names.txt，det_utils.py，yolov5s_bs1.om
  - 来自《目标检测应用样例开发介绍（Python）》[^object_detect_python]中的样例代码。

复制或上传相关程序后，`/root/ailab/camera` 中的程序如下：
```
(base) root@davinci-mini:~/ailab/camera# tree
.
├── coco_names.txt
├── det_utils.py
├── infer_camera.py
└── yolov5s_bs1.om
```

### 确保在 conda 虚拟环境中

默认情况下是在 `conda` 的虚拟环境中（提示符前面有`（base）`）。需要用到 MindSDK（python 环境要安装 mindx，而 mindx 是在 MindSDK 安装时安装的，而非通过 pip3 安装），因此要借用已有的 `conda` 虚拟环境。可执行如下命令，能得到如下输出，就表明是在 `conda` 的虚拟环境中。
```bash
(base) root@davinci-mini:~# which python
/usr/local/miniconda3/bin/python
```
如果不是在 `conda` 虚拟环境中，可以执行命令 `conda activate` 激活（进入）。如下所示：
```bash
root@davinci-mini:~# conda activate
(base) root@davinci-mini:~# 
```
> 退出`conda`虚拟环境，可执行以下命令： 
> ```bash
> (base) root@davinci-mini:~# conda deactivate
> root@davinci-mini:~# 
> ```

### 设置环境变量，启动 jupyter

确保 USB 摄像头，已经插入开发板的 USB 接口上。开发板有2个USB接口，随便插哪个都可以。
然后依次执行以下命令，设置环境变量，启动 jupyter：
```bash
(base) root@davinci-mini:~/ailab/camera# source /usr/local/Ascend/ascend-toolkit/set_env.sh
(base) root@davinci-mini:~/ailab/camera# source /usr/local/Ascend/mxVision/set_env.sh
(base) root@davinci-mini:~/ailab/camera# jupyter lab --ip=192.168.137.100 --allow-root --no-browser
...
[C 2025-06-17 12:14:29.507 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-221784-open.html
    Or copy and paste one of these URLs:
        http://192.168.137.100:8888/lab?token=1a29216eb8b13f9ae0ad12dabb40addcb1153794ef06cb22
        http://127.0.0.1:8888/lab?token=1a29216eb8b13f9ae0ad12dabb40addcb1153794ef06cb22
...
```

复制屏幕上出现的 `http://192.168.137.100:8888/lab?token=1a29216eb8b13f9ae0ad12dabb40addcb1153794ef06cb22`，到本地计算机的浏览器中，按回车键，在浏览器就出来 jupyter 界面。
```alert type=caution
复制自己的开发板上显示的内容，不要复制此处的 http://192.168.137.100/...。因为token是不同的。
```

### 运行样例程序

在本地浏览器的 jupyter 界面上，打开 `infer_camera.py`，复制所有内容。再新建一个 notebook，将内容粘贴。

然后点击界面上方的双箭头（Restart the kernels and run all cells）。

稍等一会，将浏览器界面翻到最后，就可以看到推理结果：对USB摄像头画面中的物体做识别，比如人、杯子等。

### 查看 NPU 使用情况

在本机计算机再起一个终端，执行命令`ssh root@192.168.137.100`登录开发板。

然后执行 `npu-smi info watch`。移动 USB 摄像头，对新发现的物体做识别，可以看到如下 ` AI Core(%)`列出现了非0数值，表示 NPU 算力被用到了。

```bash
(base) root@davinci-mini:~# npu-smi info watch
NpuID(Idx)  ChipId(Idx) Pwr(W)      Temp(C)     AI Core(%)  AI Cpu(%)   Ctrl Cpu(%) Memory(%)   Memory BW(%)
0           0           8.7         55          0           2           64          71          7           
0           0           8.7         56          25          2           67          71          7           
0           0           8.8         56          0           2           64          73          13          
0           0           8.7         55          0           1           66          71          7           
0           0           8.7         55          0           1           63          73          12          
0           0           8.8         56          0           2           65          71          7           
0           0           8.8         55          0           2           63          72          14          
0           0           8.7         56          0           2           63          71          7           
0           0           8.7         56          24          2           64          71          7           
0           0           8.7         55          0           1           64          73          13          
0           0           8.7         55          0           2           62          71          7           
0           0           8.7         55          23          3           62          71          6        
```   