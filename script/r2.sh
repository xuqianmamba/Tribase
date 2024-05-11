dataset=fasion_mnist_784
# ./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset --loop 2 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 700 0 --run_faiss
./release/bin/query --benchmarks_path ./benchmarks --dataset $dataset --sub_nprobe_ratio 1 --opt_levels 0 7 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 700 0 --ratios 1.0 0.95 0.9 0.85 0.8 0.75 0.7
