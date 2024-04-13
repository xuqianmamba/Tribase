# ./debug/bin/query --benchmarks_path ./benchmarks --dataset sift1m --run_faiss --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 

# ./debug/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratios 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01 --loop 3 --nprobes 500 # 

# ./debug/bin/query --benchmarks_path ./benchmarks --dataset msong --run_faiss --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 

./debug/bin/query --benchmarks_path ./benchmarks --dataset msong --high_precision_subNN_index --opt_levels 1 --ratios 1.00 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500
