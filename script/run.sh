./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --run_faiss --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0

./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratios 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0