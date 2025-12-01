**команда запуска:**
docker run -v C:/vscode_projects/cmc_course/cuda-course:/root/cuda-course -it --rm --name ubuntu --gpus all nvidia/cuda:12.4.1-devel-ubuntu22.04 bash
docker run -v C:/vscode_projects/cmc_course/cuda-course:/root/cuda-course -it --rm --name ubuntu --gpus all nvidia/cuda:12.6.1-devel-ubuntu22.04 bash

docker run -v /path/to/local/folder:/path/in/container my_image
важно, чтобы был тэг devel, потому что в тэге runtime нет команды nvcc!
или --gpus '"device=0"'

# VSCode:
# Ctrl-Shift-P 'attach running container' -> development in container

**команда запуска докер образа вместе с nsight-compute:**
docker run -v C:/vscode_projects/cmc_course/cuda-course:/root/cuda-course -v /tmp/.X11-unix:/tmp/.X11-unix -it --rm --name ubuntu-ncu --gpus all --cap-add=SYS_ADMIN --network host -e DISPLAY=host.docker.internal:0.0 nsight-compute:12.4.1 bash

**update nsight-compute:**
docker run -v C:/vscode_projects/cmc_course/cuda-course:/root/cuda-course -v $HOME/.Xauthority:/root/.Xauthority -it --rm --name ubuntu-ncu --gpus all --cap-add=SYS_ADMIN --network host -e DISPLAY=localhost:0.0 nsight-compute:12.4.1 bash

docker run -it --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --cap-add=SYS_ADMIN --security-opt seccomp=unconfined -v $(pwd):/mnt -w /mnt --network host nsight-compute:12.4.1 bash
