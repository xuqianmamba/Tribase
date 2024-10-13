# ./release/bin/hnswlib_test --benchmarks_path ./benchmarks --dataset sift10k --k 1 --tag test
# ./release/bin/hnswlib_test --benchmarks_path ./benchmarks --dataset sift10k --build_only

datasets=("msong" "sift1m" "nuswide" "glove25" "fasion_mnist_784")

# k=1
# for dataset in ${datasets[@]}; do
#     ./release/bin/hnswlib_test --benchmarks_path ./benchmarks --dataset ${dataset} --k ${k} --tag "k=1"
# done

# k=10
# for dataset in ${datasets[@]}; do
#     ./release/bin/hnswlib_test --benchmarks_path ./benchmarks --dataset ${dataset} --k ${k} --tag "k=10"
# done

k=1
for dataset in ${datasets[@]}; do
    ./release/bin/hnswlib_test --benchmarks_path ./benchmarks --dataset ${dataset} --k ${k} --pure_hnsw --build_only --tag "size"
done

k=1
for dataset in ${datasets[@]}; do
    ./release/bin/hnswlib_test --benchmarks_path ./benchmarks --dataset ${dataset} --k ${k} --build_only --tag "size"
done