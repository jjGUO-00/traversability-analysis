# 系统配置

ubuntu 18.04 

ros melodic

# MMsegmentation

（1）创建并激活conda环境

```shell
conda create --name env_name python=3.7 -y
conda activate env_name
```

（2）安装pytorch

查看cuda版本

```shell 
nvcc --version
```

到pytorch官网查看安装指令

```
# 以CUDA 11.3为例
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

检查torch是否安装成功

```shell
python
>>> import torch
>>> torch.__version__
'1.10.1'
>>> torch.cuda.is_available()
True
```

（3）安装 mmcv

```
pip install -U openmim
mim install mmengine
mim install mmcv-full
```

（4）安装MMsegmentation

```
git clone -b v0.30.0 https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

# ros 依赖

```shell
pip install rospkg
```

# MMdeploy

（1）安装 MMDeploy 模型转换工具（含trt/ort自定义算子）

```shell
pip install mmdeploy==0.14.0
```

（2）安装 MMDeploy SDK推理工具

支持 onnxruntime-gpu tensorrt 推理

```shell
pip install mmdeploy-runtime-gpu==0.14.0
```

（3）安装推理引擎

从 [NVIDIA 官网](  https://developer.nvidia.com/nvidia-tensorrt-8x-download )下载 TensorRT-8.2.3.0 CUDA 11.x 安装包并解压到当前目录

[**TensorRT 8.2 GA Update 2 for Linux x86_64 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4 and 11.5 TAR Package**](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/tars/tensorrt-8.2.3.0.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz) 

```shell
tar -zxvf tensorrt-8.2.3.0.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp37-none-linux_x86_64.whl
```

（4）克隆 mmdeploy 仓库

转换时，需要使用 mmdeploy 仓库中的配置文件，建立转换流水线

```
git clone -b v0.14.0 --recursive https://github.com/open-mmlab/mmdeploy.git
```

（5）添加环境变量

进入到在有TensorRT-8.2.3.0和cuda文件夹的路径下

```shell
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH
```

# 模型转换

建议将以下指令写成一个 .sh文件

```shell
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

python mmdeploy/tools/deploy.py \
    mmdeploy/configs/mmseg/segmentation_tensorrt_static-1024x1024.py \
    mmsegmentation/configs/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes.py \
    mmsegmentation/checkpoints/segformer/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth \
    mmsegmentation/demo/demo.png \
    --work-dir mmdeploy_model/segformer/segformer_2 \
    --device cuda \
    --show \
    --dump-info
```

转换完成后，在 --work-dir 指定的路径下会有五个文件：

```
mmdeploy_models/segformer/segformer_2
├── deploy.json
├── detail.json
├── end2end.onnx
├── end2end.engine
└── pipeline.json
```

end2end.onnx: 推理引擎文件。可用 ONNX Runtime ，tensorRT推理
*.json: mmdeploy SDK 推理所需的 meta 信息

整个文件夹被定义为mmdeploy SDK model。换言之，mmdeploy SDK model既包括backend model，也包括inference meta 信息。

# 模型推理

在转换完成后，既可以使用 Model Converter 进行推理，也可以使用 Inference SDK。

### Backend model inference

```python
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = 'configs/mmseg/segmentation_onnxruntime_dynamic.py'
model_cfg = './unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
device = 'cpu'
backend_model = ['./mmdeploy_models/mmseg/ort/end2end.onnx']
image = './demo/resources/cityscapes.png'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='./output_segmentation.png')
```

### SDK model inference

```python
from mmdeploy_runtime import Segmentor
import cv2
import numpy as np

img = cv2.imread('./demo/resources/cityscapes.png')

# create a classifier
segmentor = Segmentor(model_path='./mmdeploy_models/mmseg/ort', device_name='cpu', device_id=0)
# perform inference
seg = segmentor(img)

# visualize inference result
## random a palette with size 256x3
palette = np.random.randint(0, 256, size=(256, 3))
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
  color_seg[seg == label, :] = color
# convert to BGR
color_seg = color_seg[..., ::-1]
img = img * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)
cv2.imwrite('output_segmentation.png', img)
```



# C++接口(未完成)

```shell
# 进入MMDeploy根目录
cd ${MMDEPLOY_DIR} 
 
## 以后可以从这部分开始运行
# 进入example文件夹
cd build/install/example
```

编译代码文件

```shell
# 编译object_detection.cpp
mkdir -p build && cd build
cmake -DMMDeploy_DIR=${MMDEPLOY_DIR}/build/install/lib/cmake/MMDeploy ..
make object_detection
```



在项目的CMakeLists中，增加：

```cmake
find_package(MMDeploy REQUIRED)
target_link_libraries(${name} PRIVATE mmdeploy ${OpenCV_LIBS})
```

编译时，使用 -DMMDeploy_DIR，传入MMDeloyConfig.cmake所在的路径。它在预编译包中的sdk/lib/cmake/MMDeloy下。



参考：

[安装](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/get_started.md)

[MMSegmentation 模型部署](https://github.com/open-mmlab/mmdeploy/blob/main/docs/zh_cn/04-supported-codebases/mmseg.md)

[LINUX-X86_64 下构建方式](https://mmdeploy.readthedocs.io/zh_CN/latest/01-how-to-build/linux-x86_64.html)
