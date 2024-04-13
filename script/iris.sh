# conda deactivate
# rm -rf debug
# rm -rf release
# cmake -DCMAKE_BUILD_TYPE=Debug -S . -B debug
# cmake -DCMAKE_BUILD_TYPE=Release -S . -B release

cd debug
make -j &
cd ..

cd release
make -j &
cd ..

wait

# export ASAN_OPTIONS=abort_on_error=1
# gdb --args ./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --high_precision_subNN_index
# ./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --k 5 --nprobes 0 --opt_levels OPT_NONE --high_precision_subNN_index --metric ip
# ./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --k 5 --nprobes 0 --opt_levels OPT_SUBNN_IP --high_precision_subNN_index
# sudo ./debug/bin/query --dataset sift1m --nprobes 0 --high_precision_subNN_index
# sudo ./release/bin/query --dataset sift1m --nprobes 0 --opt_levels OPT_SUBNN_L2 --cache
# ./release/bin/query --benchmarks_path ./benchmarks --dataset msong --high_precision_subNN_index --opt_levels OPT_NONE --loop 10
# ./release/bin/query --benchmarks_path ./benchmarks --dataset msong --high_precision_subNN_index --opt_levels OPT_NONE --loop 10 --nlist 0
# ./release/bin/query --benchmarks_path ./benchmarks --dataset msong --high_precision_subNN_index --loop 10 --match
# ./release/bin/query --benchmarks_path ./benchmarks --dataset msong --high_precision_subNN_index --opt_levels OPT_NONE --loop 10
# gprof ./release/bin/query gmon.out > analysis.txt
# ./debug/bin/query --benchmarks_path ./benchmarks --dataset msong --high_precision_subNN_index --cache --opt_levels OPT_NONE
# gprof ./debug/bin/query gmon.out > analysis.txt
# sudo ./debug/bin/query --dataset msong --nprobes 0 --cache

# sudo cp -r /home/xuqian/Triangle/benchmarks/sift1m ./benchmarks

# /home/xuqian/Triangle/benchmarks
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 0 --high_precision_subNN_index --loop 1 # --run_faiss --verbose
# ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 1 --high_precision_subNN_index --loop 1 # --run_faiss --verbose
# ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 2 --high_precision_subNN_index --loop 1 # --run_faiss --verbose
# ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 4 --high_precision_subNN_index --loop 1 # --run_faiss --verbose
# ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 3 --high_precision_subNN_index --loop 1 # --run_faiss --verbose
# ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 5 --high_precision_subNN_index --loop 1 # --run_faiss --verbose
# ./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 7 --high_precision_subNN_index --loop 1 # --run_faiss --verbose

./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --opt_levels 1 --high_precision_subNN_index --loop 3 --ratio 0.1 --nprobes 10


./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --run_faiss --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0

./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 1.0 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.95 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.9 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.85 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.8 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.75 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.7 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0
./release/bin/query --benchmarks_path ./benchmarks --dataset sift1m --high_precision_subNN_index --opt_levels 1 --ratio 0.65 --loop 3 --nprobes 1 3 5 7 10 30 50 70 100 200 300 500 # 700 0