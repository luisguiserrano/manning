#The base image - The container is built on top of this image --# reference: https://hub.docker.com/_/ubuntu/
FROM ubuntu:18.04

MAINTAINER Christian Picon C <christian91mp@gmail.com>

USER root

# Install Ubuntu packages
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates build-essential curl git-core htop pkg-config unzip unrar tree freetds-dev vim \
sudo nodejs npm net-tools flex perl automake bison libtool byacc

#Installing Python and PIP
RUN apt-get update && apt-get install -y \
    python3.7 \
    python3-pip \
    python3-dev

#Installing Ubunntu packages for Graphviz
RUN apt-get update && apt-get install -y libv8-3.14-dev \
    libcurl4-gnutls-dev \
    libxml2-dev

#Install Graphviz for python and R
RUN apt-get update && apt-get install -y graphviz \
 && wget http://kr.archive.ubuntu.com/ubuntu/pool/universe/g/graphviz/graphviz_2.40.1-2_amd64.deb \
 && dpkg -i graphviz_2.40.1-2_amd64.deb

RUN apt-get update && apt-get install -y librsvg2-dev

#WORKDIR /usr/local/
RUN pip3 install --upgrade setuptools pip wheel

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

COPY notebooks ./usr/local/src/

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Adds metadata to the image as a key value pair
LABEL version="1.0"

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
WORKDIR /usr/local/src/
# Add Tini. Tini operates as a process subreaper for jupyter. This prevents kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]



