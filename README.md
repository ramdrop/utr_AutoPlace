### HERO (Hybrid-Estimate Radar Odometry)

This repo is developed based on [utiasASRL/hero_radar_odometry](https://github.com/utiasASRL/hero_radar_odometry).

| Methods         | Supervision       | Translational Error (%) | Rotational Error (1 x 10<sup>-3</sup> deg/m) |
|-----------------|-------------------|-------------------------|:--------------------------------------------:|
| [Under the Radar](https://arxiv.org/abs/2001.10789) | Supervised (L)    | 2.0583                  | 6.7                                          |
| [RO Cen](https://www.robots.ox.ac.uk/~mobile/Papers/2018ICRA_cen.pdf)          | Unsupervised (HC) | 3.7168                  | 9.5                                          |
| [MC-RANSAC](https://arxiv.org/abs/2011.03512)       | Unsupervised (HC) | 3.3204                  | 10.95                                        |
| [HERO](https://arxiv.org/abs/2105.14152) (Ours)     | Unsupervised (L)  | 1.9879                  | 6.524                                        |

### Trained results on nuScenes (ckpt)

![ckpt](/ckpt/tb.png)


### Build Instrucions
We provide a Dockerfile which can be used to build a docker image with all the required dependencies installed. It is possible to build and link all required dependencies using cmake, but we do not provide instrucions for this. To use NVIDIA GPUs within docker containers, you'll need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

1.Building Docker Image:
```bash
cd docker
docker build -t hero-image .
```

2.Launching NVIDIA-Docker Container:

```bash
cd utr_milliPlace

docker run --gpus all -it \
    -v /LOCAL/ramdrop/github/utr_milliPlace:/github/utr_milliPlace \
    --name utr_milliPlace \
    --shm-size 16G \
    --ipc=host \
    -p 6022:22 \
    hero-image:latest
```
3.Building CPP-STEAM: after launching the docker container,

```bash
git clone git@github.com:ramdrop/hero_nusc.git
cd hero_nusc
mkdir cpp/build
cd cpp/build
cmake .. && make
```

### Data Preprocessing

copy the processed nuScenes dataset (from milliPlace) to the following directory:
```
├── utr_milliPlace
│   ├── nuscenes_dataset
│   │   └── 7n5s_xy11
```

```bash
cd preprocess_nuscenes
# generate a .npy file for the sequences and transformations, executed outside the container
python preprocess.py --split='trainval' --nuscenes_datadir=/LOCAL/ramdrop/dataset/nuscenes
# write the transformation data in the format of oxford robot car, executed inside the container
python nuScenes_odom.py

python preprocess.py --split='test'  --nuscenes_datadir=/LOCAL/ramdrop/dataset/nuscenes
python nuScenes_odom.py

```

### Train

```bash
cd /github/utr_milliPlace
python3 train.py --config config/nuScenes.json
```

### Evaluate

```bash
cd /github/utr_milliPlace
python3 eval.py --pretrain ckpt/lastest.pt --config ckpt/nuScenes.json
```

### Generate descriptors for place recognition

```bash
cd /github/utr_milliPlace
python3 desc.py --config ckpt/nuScenes.json --pretrain ckpt/latest.pt
```
