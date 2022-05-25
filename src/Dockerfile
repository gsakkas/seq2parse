FROM tensorflow/tensorflow:1.13.2-gpu-py3

# RUN apt-get -y update && \
#     apt-get -y upgrade

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

WORKDIR /mnt

# CMD ["python3", "train_block_fix.py"]
