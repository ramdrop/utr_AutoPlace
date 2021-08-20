# docker run --gpus all --rm -it \
#     --name hero_docker_oxford \
#     -v /LOCAL/ramdrop/dataset/oxford-robocar:/workspace/robocar \
#     --shm-size 16G \
#     --ipc=host \
#     hero-image:latest

# attach to a running container
# docker exec -it 047bc914da37 /bin/bash

cd utr_milliPlace
docker run --gpus all -it \
    -v /LOCAL/ramdrop/github/utr_milliPlace:/github/utr_milliPlace \
    --name utr_milliPlace \
    --shm-size 16G \
    --ipc=host \
    -p 6022:22 \
    hero-image:latest

