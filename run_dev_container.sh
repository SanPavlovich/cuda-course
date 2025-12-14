#!/bin/bash
docker run -d -v C:/vscode_projects/cmc_course/cuda-course:/root/cuda-course -it --rm --name ubuntu --gpus all nvidia/cuda:12.4.1-devel-ubuntu22.04 bash