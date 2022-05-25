#!/bin/bash

docker run -u $(id -u):$(id -g) --gpus device=$1 -v /home/gsakkas/seq2parse/src:/mnt -d -it gsakkas/tensorflow:1.13.2-gpu-py3 $2
