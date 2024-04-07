import os
import os.path as osp

root = "/home/xuqian/Triangle"
dataset = "sift1m"
file_format = "fvecs"

base_file = osp.join(
    root, "benchmarks", dataset, "origin", f"{dataset}_base.{file_format}"
)
query_file = osp.join(
    root, "benchmarks", dataset, "origin", f"{dataset}_query.{file_format}"
)
output_file = osp.join(root, "benchmarks", dataset, "result", f"result.txt")

os.system(
    "sudo ./release/bin/main --base_file {} --query_file {} --output_file {} --k 100".format(
        base_file, query_file, output_file
    )
)
