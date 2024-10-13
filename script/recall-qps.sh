# quick build
cmake --build ./release -j

loop=3
output_csv_file="./logs/recall-qps.csv"
faiss_output_csv_file="./logs/recall-qps-faiss.csv"

# simple test
# test_dataset="sift10k"

# ./release/bin/query --benchmarks_path ./benchmarks --dataset $test_dataset \
#     --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 \
#     --run_faiss \
#     --loop $loop \
#     --csv $faiss_output_csv_file \
#     --cache \
#     --verbose

# ./release/bin/query --benchmarks_path ./benchmarks --dataset $test_dataset \
#     --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 \
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

datasets=("msong" "sift1m" "nuswide" "glove25" "fasion_mnist_784")
for dataset in ${datasets[@]}; do
    ./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset \
        --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 0 \
        --sub_nprobe_ratio 1 \
        --opt_levels OPT_TRIANGLE OPT_TRI_SUBNN_L2 OPT_TRI_SUBNN_IP OPT_ALL \
        --loop $loop \
        --early_stop \
        --csv $output_csv_file \
        --verbose
done