docker run --gpus all --rm -it \
    --name hero_docker_oxford \
    -v /LOCAL/ramdrop/dataset/oxford-robocar:/workspace/robocar\
    --shm-size 16G \
    --ipc=host \
    hero-image:latest

docker exec -it 003eaae9122d /bin/bash

python3 train.py --config config/radar.json