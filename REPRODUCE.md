# reproduce guide

### Setup environment using Docker

```bash
git clone https://github.com/panjd123/Tribase.git
cd Tribase
# docker build -t panjd123/tribase-env:latest .
docker pull panjd123/tribase-env:latest
docker run -d \
  --user 1000:1000 \
  --name tribase-dev \
  -v .:/app/tribase \
  --restart always \
  panjd123/tribase-env \
  tail -f /dev/null
```

### Download datasets

```bash
pipx install gdown
gdown https://drive.google.com/file/d/12wFLDNStJU02pEn7VcAs00LyS7uzcAbl/view?usp=sharing --fuzzy
unzio -o benchmarks.zip
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
```

The results and measurement indicators will be output in the `logs/` folder

### Draw all the figures in the paper

```bash
docker exec -it tribase-dev bash figures/draw.sh
```

The figures will be output in the `figures/` folder
