cd debug
make -j
cd ..
# export ASAN_OPTIONS=abort_on_error=1
# gdb --args ./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --high_precision_subNN_index
./debug/bin/query --benchmarks_path ./benchmarks --dataset iris --high_precision_subNN_index