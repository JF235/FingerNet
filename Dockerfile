FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python2.7 python2.7-dev python-pip \
    python-numpy python-scipy python-matplotlib \
    libopencv-dev python-opencv \
    graphviz libgraphviz-dev pkg-config git wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Remove /usr/bin/python if it exists, then link python2.7
RUN if [ -e /usr/bin/python ]; then rm /usr/bin/python; fi && ln -s /usr/bin/python2.7 /usr/bin/python

RUN pip install --upgrade pip==20.3.4 setuptools==44.1.1

RUN pip install \
    numpy==1.16.6 \
    scipy==1.2.2 \
    matplotlib==2.2.5 \
    pydot==1.4.1 \
    graphviz==0.8.4 \
    tensorflow==1.0.1 \
    keras==2.0.2 \
    h5py==2.10.0

WORKDIR /workspace/FingerNet

COPY . /workspace/FingerNet

CMD ["bash"]