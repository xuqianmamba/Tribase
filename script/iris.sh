cd debug
make -j
cd ..
# export ASAN_OPTIONS=abort_on_error=1
# gdb --args ./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --high_precision_subNN_index
# ./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --k 5 --nprobes 0 --opt_levels OPT_NONE --high_precision_subNN_index --metric ip
./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --k 5 --nprobes 0 --opt_levels OPT_SUBNN_IP --high_precision_subNN_index