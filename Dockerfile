FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ noble-security main restricted universe multiverse" >> /etc/apt/sources.list

RUN apt update && \
    apt install -y \
    build-essential \
    cmake \
    git \
    gpg-agent \
    wget \
    libblas-dev \
    liblapack-dev \
    unzip \
    python3 \
    python3-pip \
    python3-venv \
    swig

# install gtest
RUN mkdir -p /tmp && \
    cd /tmp && \
    git clone https://github.com/google/googletest.git --depth=1 && \
    cd googletest && \
    cmake -B release -DCMAKE_BUILD_TYPE=Release . && \
    cmake --build release -j && \
    cmake --install release && \
    rm -rf /tmp/googletest

# install intel MKL
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-oneapi-mkl=2024.2.0-663 intel-oneapi-mkl-devel=2024.2.0-663

# install eigen
RUN cd /tmp && \
    wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar -zxvf eigen-3.4.0.tar.gz && \
    rm eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    cmake -B release -DCMAKE_BUILD_TYPE=Release . && \
    cmake --install release && \
    cmake --install release --prefix /usr && \
    rm -rf /tmp/eigen-3.4.0

WORKDIR /opt

RUN apt install -y ttf-mscorefonts-installer gdb passwd zip

# setup python3 environment
RUN python3 -m venv /opt/venv && \
    source /opt/venv/bin/activate && \
    pip install numpy pandas matplotlib

ENV PATH="/opt/venv/bin:$PATH"

RUN chmod 777 /opt/venv
RUN echo "root:123456" | chpasswd

# clean
RUN apt autoremove -y && \
    apt clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/tribase

WORKDIR /app/tribase

CMD [ "bash", "-c", "source /opt/venv/bin/activate && exec bash" ]