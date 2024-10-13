./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels OPT_TRIANGLE --loop 5 --nprobes 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels OPT_TRIANGLE --loop 5 --nprobes 0 --run_faiss

./build/bin/query --benchmarks_path ./benchmarks --dataset data_4d_mean --opt_levels OPT_ALL --nprobes 0 --ratios 0