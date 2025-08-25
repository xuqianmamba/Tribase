#!/bin/bash

target="standard"

loop=1
output_csv_file="./logs/$target-recall-qps-tribase.csv"
faiss_output_csv_file="./logs/$target-recall-qps-faiss.csv"

rm -f $output_csv_file $faiss_output_csv_file

declare -A nprobes_dict

nprobes_dict["nuswide"]="1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 450 500 510 513 516 0" # 518 518
nprobes_dict["fasion_mnist_784"]="1 3 5 7 10 20 30 40 50 60 70 80 90 150 200 0" # 100 244
nprobes_dict["msong"]="1 3 5 7 10 30 50 70 100 150 200 250 300 350 600 800 0" # 400
nprobes_dict["sift1m"]="1 3 5 7 10 30 50 70 100 150 200 250 300 400 600 800 0" # ?
nprobes_dict["glove25"]="1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 500 600 750 900 0" # ?
nprobes_dict["HandOutlines"]="1 3 5 7 10 12 14 16 18 20 22 24 26 28 0" # ?
nprobes_dict["StarLightCurves"]="1 3 5 7 10 12 14 16 18 0" # ?

export EDGE_DEVICE_ENABLED=1

datasets=("nuswide" "fasion_mnist_784" "msong" "sift1m" "glove25" "HandOutlines" "StarLightCurves")
for dataset in ${datasets[@]}; do
    nprobes="${nprobes_dict[$dataset]:-1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 450 500 550 600 650 700}"

    ./build/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
        --nprobes $nprobes \
        --sub_nprobe_ratio 1 \
        --opt_levels OPT_NONE OPT_TRIANGLE OPT_SUBNN_L2 OPT_TRI_SUBNN_L2 OPT_SUBNN_IP OPT_TRI_SUBNN_IP OPT_ALL \
        --loop $loop \
        --csv $output_csv_file \
        --verbose --cache
done