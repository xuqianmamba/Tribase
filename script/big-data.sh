#!/bin/bash

# quick build
cmake --build ./release -j

loop=3
output_csv_file="./logs/big-recall-qps.csv"
faiss_output_csv_file="./logs/big-recall-qps-faiss.csv"

nprobes="1 3 5 7 10 30 50 70 100 200 300 500 700 1000 3000 5000 7000"

if [ -f $output_csv_file ]; then
    echo "Cannot create file $output_csv_file, file already exists."
    exit 1
fi

datasets=("deep100m")
for dataset in ${datasets[@]}; do

    ./release/bin/query --benchmarks_path ./benchmarks --dataset $test_dataset \
        --nprobes $nprobes \
        --run_faiss \
        --loop $loop \
        --csv $faiss_output_csv_file \
        --cache \
        --verbose

    # ./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
    #     --nprobes $nprobes \
    #     --sub_nprobe_ratio 1 \
    #     --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
    #     --loop $loop \
    #     --early_stop \
    #     --csv $output_csv_file \
    #     --verbose
done