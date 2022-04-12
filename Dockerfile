FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
WORKDIR /root
RUN pip install jupyter imageio matplotlib
RUN apt update && apt install ffmpeg libsm6 libxext6  -y
EXPOSE 8888
COPY . .
CMD /bin/bash
# CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root