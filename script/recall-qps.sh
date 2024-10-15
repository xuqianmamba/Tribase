#!/bin/bash

target="release"

# quick build
cmake --build ./$target -j

loop=3
output_csv_file="./logs/$target-recall-qps.csv"
faiss_output_csv_file="./logs/$target-recall-qps-faiss.csv"

nprobes="1 3 5 7 10 30 50 70 100 200 300 500 0"

# simple test
# test_dataset="sift1m"

# ./"$target"/bin/query --benchmarks_path ./benchmarks --dataset $test_dataset \
#     --nprobes $nprobes \
#     --run_faiss \
#     --loop $loop \
#     --csv $faiss_output_csv_file \
#     --cache \
#     --verbose

# ./"$target"/bin/query --benchmarks_path ./benchmarks --dataset $test_dataset \
#     --nprobes $nprobes \
#     --sub_nprobe_ratio 1 \
#     --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
#     --loop $loop \
#     --early_stop \
#     --csv $output_csv_file \
#     --verbose

# exit 0


if [ -f $output_csv_file ]; then
    echo "Cannot create file $output_csv_file, file already exists."
    exit 1
fi

if [ -f $faiss_output_csv_file ]; then
    echo "Cannot create file $faiss_output_csv_file, file already exists."
    exit 1
fi

datasets=("msong" "sift1m" "nuswide" "glove25" "fasion_mnist_784")
for dataset in ${datasets[@]}; do
    ./"$target"/bin/query --benchmarks_path ./benchmarks --dataset $test_dataset \
        --nprobes $nprobes \
        --run_faiss \
        --loop $loop \
        --csv $faiss_output_csv_file \
        --cache \
        --verbose

    ./"$target"/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
        --nprobes $nprobes \
        --sub_nprobe_ratio 1 \
        --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
        --loop $loop \
        --early_stop \
        --csv $output_csv_file \
        --verbose
done