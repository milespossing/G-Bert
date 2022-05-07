#!/bin/bash
docker run -v /home/milespossing/repos/G-Bert:/opt/project -w /opt/project/code --rm --gpus all \
	mpossing/dlh-final:latest python $1
