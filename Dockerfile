FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04


## CLeanup
RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

ARG APT_INSTALL="apt-get install -y --no-install-recommends"
## Python3
# Install python3
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive ${APT_INSTALL} \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        wget && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y python3.6-tk zlib1g-dev libjpeg-dev libsm6 libxext6 git nano tmux default-jre ninja-build libopenblas-dev libomp-dev

# Link python to python3
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python

# Setuptools
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py
RUN rm get-pip.py

## Locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

## SSH
# Install openssh
RUN apt-get update &&  \
    ${APT_INSTALL} openssh-server && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*

# Setup environment for ssh session
RUN echo "export PATH=$PATH" >> /etc/profile && \
  echo "export LANG=$LANG" >> /etc/profile && \
  echo "export LANGUAGE=$LANGUAGE" >> /etc/profile && \
  echo "export LC_ALL=$LC_ALL" >> /etc/profile && \
  echo "export PYTHONIOENCODING=$PYTHONIOENCODING" >> /etc/profile

# Create folder for openssh fifos
RUN mkdir -p /var/run/sshd

# Disable password for root
RUN sed -i -re 's/^root:[^:]+:/root::/' /etc/shadow
RUN sed -i -re 's/^root:.*$/root::0:0:System Administrator:\/root:\/bin\/bash/' /etc/passwd

# Permit root login over ssh
RUN echo "Subsystem    sftp    /usr/lib/sftp-server \n\
PasswordAuthentication yes\n\
ChallengeResponseAuthentication yes\n\
PermitRootLogin yes \n\
PermitEmptyPasswords yes\n" > /etc/ssh/sshd_config

## Expose ports
# IPython
EXPOSE 8888
# ssh
EXPOSE 22

COPY requirements.txt requirements.txt
COPY Makefile Makefile
RUN pip install Cython
RUN pip install numpy
RUN pip install -r requirements.txt

RUN mkdir /app
COPY . /app
WORKDIR /app

## Setup PYTHONPATH
ENV PYTHONPATH="$PYTHONPATH:/app"
RUN echo "export PYTHONPATH=$PYTHONPATH" >> /etc/profile

CMD ["/usr/sbin/sshd", "-D"]