# HERO (Hybrid-Estimate Radar Odometry)


| Methods         | Supervision       | Translational Error (%) | Rotational Error (1 x 10<sup>-3</sup> deg/m) |
|-----------------|-------------------|-------------------------|:--------------------------------------------:|
| [Under the Radar](https://arxiv.org/abs/2001.10789) | Supervised (L)    | 2.0583                  | 6.7                                          |
| [RO Cen](https://www.robots.ox.ac.uk/~mobile/Papers/2018ICRA_cen.pdf)          | Unsupervised (HC) | 3.7168                  | 9.5                                          |
| [MC-RANSAC](https://arxiv.org/abs/2011.03512)       | Unsupervised (HC) | 3.3204                  | 10.95                                        |
| [HERO](https://arxiv.org/abs/2105.14152) (Ours)     | Unsupervised (L)  | 1.9879                  | 6.524                                        |

# Build Instrucions
We provide a Dockerfile which can be used to build a docker image with all the required dependencies installed. It is possible to build and link all required dependencies using cmake, but we do not provide instrucions for this. To use NVIDIA GPUs within docker containers, you'll need to install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Building Docker Image:
```
cd docker
docker build -t hero-image .
```
## Launching NVIDIA-Docker Container:
```
docker run --gpus all --rm -it \
    --name hero_docker_oxford \
    -v /LOCAL/ramdrop/dataset/oxford-robocar:/workspace/robocar\
    --shm-size 16G \
    --ipc=host \
    hero-image:latest
```
## Building CPP-STEAM:
After launching the docker container, clone repo and build C++ code:
```
git clone git@github.com:ramdrop/hero_nusc.git
cd hero_nusc
mkdir cpp/build
cd cpp/build
cmake .. && make
```

# Getting Started
### Train
in `config/nuScenes.json` define dataset path:
```
"data_dir": [your_dataset_path],
```
To train a new model, we provide `train.py`, which can be used as follows:

```
python3 train.py --pretrain <optional_pretrained_model_path> python3 train.py --config config/nuScenes.json
```

### Evaluate
To evaluate a trained model(by Kaiwen), use `eval.py` with the following format (do not forget to define dataset path in `ckpt/nuScenes.json`):

```
python3 eval.py --pretrain ckpt/lastest.pt --config ckpt/nuScenes.json
```

### Generate descriptors for place recognition
Do not forget to define dataset path `IMG_DIR` and `CSV_DIR`
```
python3 desc.py --config logs/nuScenes.json --pretrain logs/latest.pt
```