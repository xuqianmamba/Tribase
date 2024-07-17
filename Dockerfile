FROM ubuntu:24.04

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
    unzip

RUN mkdir -p /tmp

WORKDIR /tmp

RUN git clone https://github.com/google/googletest.git

WORKDIR /tmp/googletest

RUN cmake -B release -DCMAKE_BUILD_TYPE=Release . && \
    cmake --build release -j && \
    cmake --install release

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list && \
    apt update && \
    apt install -y intel-oneapi-mkl=2024.2.0-663 intel-oneapi-mkl-devel=2024.2.0-663

WORKDIR /tmp

RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar -zxvf eigen-3.4.0.tar.gz

WORKDIR /tmp/eigen-3.4.0

RUN cmake -B release -DCMAKE_BUILD_TYPE=Release . && \
    cmake --install release && \
    cmake --install release --prefix /usr

COPY . /app/tribase

WORKDIR /app/tribase

RUN cmake -B release -DCMAKE_BUILD_TYPE=Release . && \
    cmake --build release -j && \
    cmake -B build . && \
    cmake --build build -j

RUN apt install -y \
    python3 \
    python3-pip \
    python3-venv

RUN python3 -m venv venv && \
    source venv/bin/activate && \
    pip install gdown &&

# nuswide
RUN source venv/bin/activate && \
    gdown https://drive.google.com/file/d/1d0w5XchVZvuRcV9sDZtbWC6TnB20tODM/view?usp=sharing --fuzzy --output benchmarks/ && \
    unzip benchmarks/nuswide.zip -d benchmarks/nuswide && \
    rm benchmarks/nuswide.zip

# sift1m
# RUN source venv/bin/activate && \
#     gdown https://drive.google.com/file/d/1BcTuT4su77_Ue6Wi8EU340HSYoJeHwnD/view?usp=sharing --fuzzy --output benchmarks/ && \
#     unzip benchmarks/sift1m.zip -d benchmarks/sift1m && \
#     rm benchmarks/sift1m.zip

CMD [ "/bin/bash" ]