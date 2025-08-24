./build/bin/uniform_dataset_generate --d 8 --nb 256 --nq 1024 --dataset 8d256s
./build/bin/uniform_dataset_generate --d 8 --nb 65536 --nq 1024 --dataset 8d65536s
./build/bin/uniform_dataset_generate --d 16 --nb 65536 --nq 1024 --dataset 16d65536s
./build/bin/uniform_dataset_generate --d 20 --nb 1048576 --nq 1024 --dataset 8d1048576s
./build/bin/query --dataset 8d256s --opt_levels OPT_ALL --nprobes 0 --csv logs/dim.csv
./build/bin/query --dataset 8d65536s --opt_levels OPT_ALL --nprobes 0 --csv logs/dim.csv
./build/bin/query --dataset 16d65536s --opt_levels OPT_ALL --nprobes 0 --csv logs/dim.csv
./build/bin/query --dataset 8d1048576s --opt_levels OPT_ALL --nprobes 0 --csv logs/dim.csv
