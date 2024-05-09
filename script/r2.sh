./release/bin/query --benchmarks_path ./benchmarks --dataset sift10k --loop 5 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 700 --run_faiss
./release/bin/query --benchmarks_path ./benchmarks --dataset sift10k --sub_nprobe_ratio 1 --opt_levels 7 --loop 5 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 700 --ratios 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65
