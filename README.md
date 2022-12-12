# Unified Object Detection, Lane Regression and Drivable Area Segmentation.


**Our work is developed from HybridNets (https://github.com/datvuthanh/HybridNets)**   

## Getting Started 

### Installation
The project was developed with [**Python>=3.7**](https://www.python.org/downloads/) and [**Pytorch>=1.10**](https://pytorch.org/get-started/locally/).
```bash
git clone https://github.com/datvuthanh/HybridNets
cd HybridNets
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
#### 2. Custom (ours): vihecle & pedestrain detection + lane line + drivable area segmentation

Download weights from google drive: https://drive.google.com/drive/folders/1kA16TJUVpswy6cb7EUVqN58J8ubLcytv?usp=sharing
Put them under `./weights/`

```bash

# Image inference
python hybridnets_test.py -w weights/xxx.pth --project bdd100k_person_car --source demo/image --output demo_result

# pictures of size 1280*720 are recommended

# Video inference
python hybridnets_test_videos.py -w weights/xxx.pth --project bdd100k_person_car --source demo/video --output demo_result
```

## Usage

### Data Preparation
Update your dataset paths in `projects/your_project_name.yml`.

For BDD100K: [imgs](https://bdd-data.berkeley.edu/), [det_annot](https://drive.google.com/file/d/1d5osZ83rLwda7mfT3zdgljDiQO3f9B5M/view), [da_seg_annot](https://drive.google.com/file/d/1yNYLtZ5GVscx7RzpOd8hS7Mh7Rs6l3Z3/view), [ll_seg_annot](https://drive.google.com/file/d/1BPsyAjikEM9fqsVNMIygvdVVPrmK1ot-/view)

For kitti Odometry, a tiny portion of data(20 frames) is provided in the ./sample folder.

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

