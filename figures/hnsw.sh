datasets=("nuswide" "fasion_mnist_784" "msong" "sift1m" "glove25") # 
for dataset in ${datasets[@]}; do
    ./release/bin/hnswlib_test --dataset $dataset --tag "release"
done
