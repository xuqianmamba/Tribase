#!/bin/bash

target="build_ratio"

loop=10
dataset=StarLightCurves
subk=1000
force=0

PARSED_ARGS=$(getopt -o l:d:s:k:f: --long loop:,dataset:,subk:,force: -n "$0" -- "$@")
eval set -- "$PARSED_ARGS"
while true; do
    case "$1" in
        -l|--loop)
            loop="$2"
            shift 2 # 移过选项名和它的值
            ;;
        -d|--dataset)
            dataset="$2"
            shift 2 # 移过选项名和它的值
            ;;
        -s|-k|--subk)
            subk="$2"
            shift 2 # 移过选项名和它的值
            ;;
        -f|--force)
            force=1
            shift 1 # 移过选项名
            ;;
        --) # '--' 是 getopt 添加的标记, 表示选项处理结束
            shift
            break
            ;;
        *) # 此处不应被执行
            echo "内部解析错误!"
            exit 1
            ;;
    esac
done

output_csv_file="./logs/$target-recall-qps-tribase.csv"
output_log_file="./logs/$target-recall-qps-tribase.log"
faiss_output_csv_file="./logs/$target-recall-qps-faiss.csv"

if [ $force -eq 1 ]; then
    rm -rf $output_csv_file $output_log_file
fi

declare -A nprobes_dict

nprobes_dict["nuswide"]="1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 450 500 510 513 516 0" # 518 518
nprobes_dict["fasion_mnist_784"]="1 3 5 7 10 20 30 40 50 60 70 80 90 150 200 0" # 100 244
nprobes_dict["msong"]="1 3 5 7 10 30 50 70 100 150 200 250 300 350 600 800 0" # 400
nprobes_dict["sift1m"]="1 3 5 7 10 30 50 70 100 150 200 250 300 400 600 800 0" # ?
nprobes_dict["glove25"]="1 3 5 7 10 30 50 70 100 150 200 250 300 350 400 500 600 750 900 0" # ?
nprobes_dict["HandOutlines"]="1 3 5 7 10 12 14 16 18 20 22 24 26 28 0" # ?
nprobes_dict["StarLightCurves"]="1 3 5 7 10 12 14 16 18 0" # ?

sub_nprobe_ratios="1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1"

for ((i=1; i<=10; i++)); do
    for sub_nprobe_ratio in $sub_nprobe_ratios; do
        ./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
            --nprobes 0 \
            --sub_nprobe_ratio $sub_nprobe_ratio \
            --opt_levels OPT_SUBNN_L2 \
            --loop $loop \
            --csv $output_csv_file \
            --subk $subk \
            --verbose | tee -a $output_log_file
    done
done
