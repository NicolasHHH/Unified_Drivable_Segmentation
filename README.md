# Unified Object Detection, Lane Regression and Drivable Area Segmentation.

### Project Description
INF573 Course Project: Unified Multi-class Object Detection, Lane line Regression and Drivable Area Detection
Team Member

### Table of Contents
- Demo Video
- Getting Started
- Trainin and Evaluate
- 3D map generation Demo
- Autoware Integration (New !!)

### Demo Video
https://user-images.githubusercontent.com/57991090/207166131-9fad8c0f-1b88-4bf0-b868-6104d9721be2.mp4

**Our work is developed from HybridNets (https://github.com/NicolasHHH/Unified_Drivable_Segmentation.git)**   


## Getting Started 

### Installation
The project was developed with [**Python>=3.7**](https://www.python.org/downloads/) and [**Pytorch>=1.10**](https://pytorch.org/get-started/locally/).
```bash
git clone https://github.com/NicolasHHH/Unified_Drivable_Segmentation.git
cd Unified_Drivable_Segmentation
pip install -r requirements.txt
```
 
### Demo - Unified Segmentation

#### 1. Default: car only detection + lane line + drivable area segmentation
```bash
# Download weights (cars only)
curl --create-dirs -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth

# Image inference
python hybridnets_test.py -w weights/hybridnets.pth --source demo/image --output demo_result

# Video inference
python hybridnets_test_videos.py -w weights/hybridnets.pth --source demo/video --output demo_result

# Result is saved in a new folder called demo_result
```
#### 2. Custom (ours): vehicle & pedestrain detection + lane line + drivable area segmentation

Download weights from google drive: https://drive.google.com/drive/folders/1kA16TJUVpswy6cb7EUVqN58J8ubLcytv?usp=sharing
Put them under `./weights/`

```bash

# Image inference
python hybridnets_test.py -w weights/xxx.pth --project bdd100k_person_car --source demo/image --output demo_result

# pictures of size 1280*720 are recommended

# Video inference
python hybridnets_test_videos.py -w weights/xxx.pth --project bdd100k_person_car --source demo/video --output demo_result
```
![5](https://user-images.githubusercontent.com/57991090/207167221-416cab1b-478c-465d-8806-0d25bc0a5582.jpg)

## Usage

### Data Preparation
Update your dataset paths in `projects/your_project_name.yml`.

For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)

For kitti Odometry, a tiny portion of data (10 frames) is provided in the ./sample folder.

### Training

#### 1) Edit or create a new project configuration, using bdd100k.yml as a template. Augmentation params are here.

#### 2) Train
```bash
python train.py -p bdd100k        # config filename
                -c 3              # coefficient of effnet backbone
                -n 4              # num_workers
                -b 6              # batch_size < 12G
                -w path/to/weight # use 'last' to resume training from previous session
                --freeze_det      # freeze detection head, others: --freeze_backbone, --freeze_seg
                --lr 1e-5         # learning rate
                --num_epochs 200
```
Please check `python train.py --help` for cheat codes.

#### 3) Evaluate

```bash
python val.py -w checkpoints/weight.pth
```

**Problem shooting: Validation process got killed! **

- Train on a high-RAM instance (RAM as in main memory, not VRAM in GPU). For your reference, we can only val the combined `car` class with 64GB RAM.
- Train with `python train.py --cal_map False` to not calculate metrics when validating. 

# 3D Drivable Map Construction
1) Convert the point cloud in `.bin` to `.pcd` using `kitti_bin_pcd.ipynb`
2) Colorize point clouds using `pcd_rgb.ipynb`
3) Visualize the result using the last bloc of pcd_rgb.ipynb`
<img width="457" alt="Road0C" src="https://user-images.githubusercontent.com/57991090/207167097-c211562e-26ca-4eff-8eda-8860082e5494.png">

# Autoware Integration

## ROS + Pytorch 环境配置记录

ros melodic + ubuntu 18.04 + cuda 11.3

### 1. 安装Anaconda

目前已知的支持同时import rospy 和 torch的办法.

首先去官网下载安装文件，在命令行中安装

```bash
sh Anaconda3-2022.10-Linux-x86_64.sh
# 根据提示完成安装
```

### 2.添加路径

在`~/.bashrc`中注释默认开启Anaconda 环境，并且将`ros` 默认的`python2`路径添加到文件

```bash
# 添加
export PYTHONPATH=$PYTHONPATH:/opt/ros/melodic/lib/python2.7/dist-packages

# 整段注释掉
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home/hty/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/hty/anaconda3/etc/profile.d/conda.sh" ]; then
#        . "/home/hty/anaconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/hty/anaconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<
```

这样确保了默认的python路径为ros的python。

### 3. 配置anaconda虚拟环境

从 terminal进入anaconda prompt，`~/anaconda3/bin/activate`  是第一步骤安装时设置的默认路径。

```bash
 xxx $: source ~/anaconda3/bin/activate

# 进入成功效果如下
(base) xxx $: 

# 推出
conda deactivate
```

创建独立的虚拟环境

```bash
(base) xxx $:  conda create -n rostorch python=3.9
```

安装pytorch 具体在官方archive中找对应的版本

```bash
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

通过pip安装ros依赖和其他模型相关库

```bash
pip install netifaces rospkg
```

测试

```bash
conda activate rostorch # 进入虚拟环境

python # 打开python 版本应该为3.9.x
>>> import torch
>>> import rospy 
>>> torch.cuda.is_available()

True
```

至此环境配置完成

## 运行

```
source ~/anaconda3/bin/activate

conda activate rostorch
```

### 默认配置（仅检测车辆[卡车，自行车，轿车，公交车均归类为car]）

```bash
# 已经开启roscore，且已下载hybridnets.pth 见上文“1. Default: car only ..."
python hybridnets_ros.py
```

### 自定义配置（汽车+行人，精度提高，对暗处和小物体检测回收率提高，）

Download weights from google drive: https://drive.google.com/drive/folders/1kA16TJUVpswy6cb7EUVqN58J8ubLcytv?usp=sharing
Put them under `./weights/`

```bash
# 已经开启roscore
python hybridnets_ros.py -w weights/xxx.pth --project bdd100k_person_car 
```
## 运行效果
![hybridnets_whole](https://user-images.githubusercontent.com/57991090/211127200-a804fb85-7be6-4ca6-ad00-2ee39a47181f.png)

