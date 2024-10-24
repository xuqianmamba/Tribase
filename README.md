# Tribase: A Vector Data Query Engine for Reliable and Lossless Pruning Compression using Triangle Inequalities

## Introduction

Tribase is an vector ANN query engine that employs a novel pruning technique based on the triangle inequality. This technique significantly reduces query time without sacrificing accuracy. Tribase supports various pruning strategies at different granularities, allowing it to achieve better performance across different datasets.

## Dataset

We have prepared some tiny datasets for testing in benchmarks folder. You can download the large datasets from the following links:

- nuswide: [GoogleDrive](https://drive.google.com/file/d/1d0w5XchVZvuRcV9sDZtbWC6TnB20tODM/view?usp=sharing)
- msong: [GoogleDrive](https://drive.google.com/file/d/1BcTuT4su77_Ue6Wi8EU340HSYoJeHwnD/view?usp=drive_link)
- sift1m: [GoogleDrive](https://drive.google.com/file/d/1BcTuT4su77_Ue6Wi8EU340HSYoJeHwnD/view?usp=drive_link)

If you build from docker, it will automatically download nuswide dataset from the link above.

### Dataset Format

You can prepare your own dataset in the following format, place it in the benchmarks folder.

```
benchmark
|-- nuswide
|   |-- origin
|   |   |-- nuswide_base.fvecs
|   |   |-- nuswide_query.fvecs
```

fvecs file format is as follows:

```
<4 bytes int representing num_dimension><num_dimension * sizeof(float) bytes raw data>
...
<4 bytes int representing num_dimension><num_dimension * sizeof(float) bytes raw data>
```

## Experimental Setup

Our server setup includes two Intel Xeon Gold 5318Y CPUs, each with 24 cores and 48 threads, totaling 96 CPU cores. The server boasts 2TB of memory and runs on CentOS Stream 8 operating system.

We also provide a dockerfile based on Ubuntu22.04 with all the dependencies installed.

## Quick Start

```bash
docker build -t tribase .
docker run -it tribase
./release/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --nprobes 50 100 300 1000 --run_faiss --verbose
./release/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
  --nprobes 50 100 300 1000 --cache --loop 3 --verbose
```

## How to Run

### Docker

We highly recommend using the provided Dockerfile to build the project, just run the following commands:

```bash
docker build -t tribase .
docker run -it tribase
```

### Manual Installation

If you want to build the project on your own machine, you should install the following dependencies:

- build-essential (g++ >= 13.2.0)
- cmake
- openblas
- intel-mkl = 2024.2.0-663
- Eigen3

### Build

Dockerfile will automatically build the project, but if you want to build it manually, you can use the following commands:

```bash
cmake -B release -DCMAKE_BUILD_TYPE=Release .
cmake --build release -j

cmake -B build .
cmake --build build -j
```

We only measure pruning rates in debug or standard mode, so if you need performance-related metrics, please use the release compiled version. If you require metrics related to pruning rates, use the debug or standard compiled version.

### Run

We have prepared a fully functional script named `query` for conducting benchmark tests and other tasks. Next, we will demonstrate how to replicate our experimental results.

#### Faiss Baseline

As a baseline and to generate ground truth, we use faiss-ivfflat. You may execute run_faiss once to obtain baseline values.

```bash
./release/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --nprobes 50 100 300 1000 --run_faiss --verbose
```

#### Tribase Index Generation

Subsequently, you can run our Tribase algorithm, which supports various combinations of three strategies. You can specify these by using the `--opt_levels` parameter, separating multiple strategies with spaces. During training, we will use the union of these strategies and individually test the query performance of each.

The available strategies are as follows:

- `OPT_NONE`
- `OPT_TRIANGLE`
- `OPT_SUBNN_L2`
- `OPT_SUBNN_IP`
- `OPT_TRI_SUBNN_L2`
- `OPT_TRI_SUBNN_IP`
- `OPT_ALL`

You can generate a Tribase index that supports various strategies with the following command, where the `--sub_nprobe_ratio` parameter is used to specify the nprobe ratio for the sub-index, affecting the index quality and construction speed. `1` denotes the highest quality.

```bash
./release/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --opt_levels OPT_ALL --sub_nprobe_ratio 0.3 --train_only --verbose
```

#### Tribase Query Performance

Next, you can test the query performance of the Tribase index with the following command, `--cache` is used to use the cached index, and `--loop` is used to specify the number of loops for each query.

```bash
./release/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
  --nprobes 50 100 300 1000 --cache --loop 3 --verbose
```

To obtain accurate pruning rates, it is necessary to introduce some atomic operations, which may result in a decrease in performance. You can run the following command in standard mode to output this information:

```bash
./build/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
  --nprobes 50 100 300 1000 --cache --verbose
```

After running the above commands, you can check the results in `benchmarks/nuswide/result/log.csv`.

#### Average Distance Ratio

r2 (or Average Distance Ratio in paper) is a metric that measures the average distance ratio between the query result and the ground truth. By adjusting the `--ratios` parameter, you can obtain the r2 results for different search pruning ratios.

```bash
./build/bin/query --benchmarks_path ./benchmarks --dataset nuswide \
  --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
  --nprobes 50 100 300 1000 --ratios 1.0 0.95 0.9 0.85 0.8 0.75 0.7 \
  --cache --verbose
```

#### Further Usage

Finally, you can use the following command to get a more comprehensive usage guide for this script:

```bash
./release/bin/query --help

Usage: tribase [--help] [--version] [--benchmarks_path VAR] [--dataset VAR] [--input_format VAR] [--output_format VAR] [--k VAR] [--nprobes VAR...] [--opt_levels VAR...] [--train_only] [--cache] [--sub_nprobe_ratio VAR] [--metric VAR] [--run_faiss] [--loop VAR] [--nlist VAR] [--verbose] [--ratios VAR...] [--csv VAR] [--dataset_info] [--early_stop]

Optional arguments:
  -h, --help          shows help message and exits 
  -v, --version       prints version information and exits 
  --benchmarks_path   benchmarks path [nargs=0..1] [default: "/home/xuqian/Triangle/benchmarks"]
  --dataset           dataset name [nargs=0..1] [default: "msong"]
  --input_format      format of the dataset [nargs=0..1] [default: "fvecs"]
  --output_format     format of the output [nargs=0..1] [default: "bin"]
  --k                 number of nearest neighbors [nargs=0..1] [default: 100]
  --nprobes           number of clusters to search [nargs=0..100] [default: {0}]
  --opt_levels        optimization levels [nargs=0..10] [default: {"OPT_NONE" "OPT_TRIANGLE" "OPT_SUBNN_L2" "OPT_SUBNN_IP"..."OPT_ALL"}]
  --train_only        train only 
  --cache             use cached index 
  --sub_nprobe_ratio  ratio of the number of subNNs to the number of clusters [nargs=0..1] [default: 1]
  --metric            metric type [nargs=0..1] [default: "l2"]
  --run_faiss         run faiss 
  --loop              [nargs=0..1] [default: 1]
  --nlist             [nargs=0..1] [default: 0]
  --verbose           verbose 
  --ratios            search ratio [nargs=0..100] [default: {1}]
  --csv               csv result file [nargs=0..1] [default: ""]
  --dataset_info      only output dataset-info to csv file 
  --early_stop        early stop
```

## Usage in Your Own Project

You can use the Tribase index in your own project by including the `src/tribase.h` header file and linking the `tribase` library. The following is a tiny example of how to use the Tribase index in your project.

```cpp
#include "tribase.h"
#include <cmath>
#include <memory>

int main(){
    auto [base, nb, d] = tribase::loadFvecs("base.fvecs");
    auto [query, nq, _] = tribase::loadFvecs("query.fvecs");
    int nlist = sqrt(nb);
    int nprobe = std::max(1, nlist / 10);
    tribase::Index index;
    // index.load_index("example.index");
    index = tribase::Index(d, nb, base, tribase::METRIC_L2, tribase::OPT_ALL);
    index.train(nb, d, base);
    index.add(nb, base.get());
    index.save_index("example.index");
    int k = 100; // number of nearest neighbors
    std::unique_ptr<float[]> distances = std::make_unique<float[]>(nq * k);
    std::unique_ptr<idx_t[]> labels = std::make_unique<idx_t[]>(nq * k);
    index.search(nq, query.get(), k, distances.get(), labels.get());
    return 0;
}
```

## trifaiss

We also provide a modified version of faiss that supports the triangle inequality pruning strategy. You can find the source code in the `trifaiss` folder. 

We still highly recommend using the provided Dockerfile to build the project, after entering the docker container, you can simply run the following commands:

```bash
source venv/bin/activate
cd trifaiss
python run.py
```

### Build

To build the trifaiss, you should install the following dependencies in addition to the ones mentioned above:

- swig = 4.2.0
- python packages: setuptools, numpy, pandas, tqdm

Then, you can build the trifaiss library with the following commands:

```bash
source venv/bin/activate
cd trifaiss
cmake -B build -DCMAKE_BUILD_TYPE=Release .
make -j -C build swigfaiss
python setup.py install
```

### Run

You can simply use `run.py` script to test our trifaiss.

You can modify the parameters in `main` function, most of the parameters are the same as the `query` script in Tribase.

```bash
source venv/bin/activate
cd trifaiss
python run.py
```

### Simple Usage

```python
import faiss # trifaiss version
import numpy as np

xb = load_vecs(base_path)
xq = load_vecs(query_path)
n = xb.shape[0]
d = xb.shape[1]
nlist = int(np.sqrt(n))
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFwithDistance(
    quantizer, d, nlist, faiss.METRIC_L2
)
index.train(xb)
index.add(xb)
index.nprobe = nlist // 10
k = 100
distances, labels = index.search(xq, k)
```

## TriHNSW

We also applied triangular pruning to the HNSW algorithm. It is important to note that this represents a completely different pruning logic and in practical testing, it can achieve lossless results. For more detailed conclusions and performance analysis, please refer to Section 4.6 in the paper. Below is the method for executing the TriHNSW tests.

```bash
./build/bin/hnswlib_test
```

## License

This project is licensed under the MIT License.
