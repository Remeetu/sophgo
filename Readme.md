# 飞桨自定义接入硬件后端(算能TPU)

请参考以下步骤进行硬件后端(算能TPU)的编译安装与验证

#### 请注意:本程序依赖于算能TPU板卡的 pcie 模式，请确保设备中安装了 libsophgo 板卡驱动和依赖包

## 环境准备与源码同步

```bash
# 1) 拉取镜像
docker pull sophgo/tpuc_dev:latest

# 2) 参考如下命令启动容器
docker run --privileged --name <YOUR_NAME> -v $PWD:/workspace -it sophgo/tpuc_dev:latest
# 若容器未启动，则手动启动并进入
docker start <YOUR_NAME>
docker exec -it <YOUR_NAME> bash

# 3) 克隆 PaddleCustomDevice 源码
git clone https://github.com/PaddlePaddle/PaddleCustomDevice

# 4) 请执行以下命令，以保证 checkout 最新的 PaddlePaddle 主框架源码
cd PaddleCustomDevice/
git submodule sync
git submodule update --remote --init --recursive
```

## PaddlePaddle 安装与运行
### 编译安装

```bash
# 1) 编译之前需要先保证环境下装有Paddle WHL包，可以直接安装CPU版本
pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/cpu-mkl/develop.html

# 2) 拉取 custom sophgo backend
cd backends/
git clone https://github.com/Remeetu/sophgo.git
cd sophgo/

# 3) 执行环境脚本
source envsetup.sh

# 4) 编译
mkdir build; cd build
cmake ..
make -j

# 5) 编译产出在 build/dist 路径下，使用 pip 安装
pip install build/dist/paddle_custom_sophgo*.whl
```

### 功能验证

```bash
# 1) 依赖于 Llama2-TPU 项目编译 bmodel，具体步骤参见 Llama2-TPU 项目的 Readme.md
# 请于 PaddleCustomDevice 项目的同级目录下，拉取 Llama2-TPU 代码
git clone https://github.com/sophgo/Llama2-TPU.git

# 2) 将生成的 bmodel 和 模型的解释文件 tokenizer.model 拷贝至 PaddleCustomDevice/backends/sophgo 目录下

# 3) 运行自定义算子单例测试
cd PaddleCustomDevice/backends/sophgo/tests/unittests/
python test_custom_llama_op_tpu.py
```
