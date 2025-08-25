# Reproduce Guide

## Host environment requirements

- `pipx` installed (if you have `gdown` already, you can skip this step)
- `docker` installed and `docker` without `sudo`

[How to install pipx and docker on Linux](https://chatgpt.com/share/68ab0118-df3c-8010-bd95-db97d55926f0)

## step-by-step guide

We do have a one-click script, but we still recommend completing it step by step. If you need a one-click script, refer to the following instructions:

```bash
# Please make sure your environment has been prepared as the first step (Host environment requirements)
git clone https://github.com/panjd123/Tribase.git
cd Tribase
bash figures/one_click.sh
# The experiment may run for quite a long time (over 12 hours), so we recommend using tools like tmux.
```

### Setup environment using Docker / Download Dataset

This step includes everything that needs to be downloaded. Please pay attention to the network conditions to prevent download errors caused by unstable network.

```bash
# Please make sure your environment has been prepared as the first step (Host environment requirements)
git clone https://github.com/panjd123/Tribase.git
cd Tribase
# docker build -t panjd123/tribase-env:latest .
docker pull panjd123/tribase-env:latest

pipx install gdown
gdown https://drive.google.com/file/d/12wFLDNStJU02pEn7VcAs00LyS7uzcAbl/view?usp=sharing --fuzzy
unzip -o benchmarks.zip
```

### Setup docker container

```bash
docker run -d \
  --user "$(id -u):$(id -g)" \
  --name tribase-dev \
  -v .:/app/tribase \
  --restart always \
  panjd123/tribase-env \
  tail -f /dev/null
```

### Build project

```bash
docker exec -it tribase-dev bash script/build.sh
```

### Run all experiments

If you want to test the environment first, you can execute the following command:

```bash
docker exec -it tribase-dev ./release/bin/query --nprobes 0 --dataset HandOutlines --opt_levels OPT_TRIANGLE
```

The expected output is similar to this

```
nprobe:31 opt_level: 1 simi_ratio: 1
tri: 0(0.00%) tri_large: 0(0.00%) subnn_L2: 0(0.00%) subnn_IP: 0(0.00%)
simi_update_rate: 0.00% check_L2: 0 check_IP: 0
time_speedup: 123.20% pruning_speedup: 0.00% faiss_query_time: 0.012386 query_time: 0.010054 qps: 36801.225540
recall: 1 r2: 0
```

run all experiments

```bash
docker exec -it tribase-dev bash figures/run.sh
# The experiment may run for quite a long time (over 12 hours), so we recommend using tools like tmux.
```

The results and measurement indicators will be output in the `logs/` folder

### Draw all the figures in the paper

```bash
docker exec -it tribase-dev bash figures/draw.sh
```

The figures will be output in the `figures/` folder

### Download

```bash
scp user@server:/path/to/Tribase/figures/*.png /local/path/
# scp user@server:~/Tribase/figures/*.png .
```
