import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="msong")
parser.add_argument("--faiss", action="store_true")

args = parser.parse_args()

dataset = args.dataset
faiss = args.faiss

df = pd.read_csv(f"/home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/log.csv")

if faiss:
    df_f = pd.read_csv(
        f"/home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/faiss_result.txt",
        sep=" ",
        header=None,
    )
    df_f.columns = ["nprobes", "query_time", "recall", "r2"]
else:
    df_f = df[df["opt_level"] == 0]
    df = df[df["opt_level"] != 0]

plt.scatter(np.log10(df["r2"]), df["query_time"], c="blue", label="Tribase")
plt.scatter(np.log10(df_f["r2"]), df_f["query_time"], c="red", label="Faiss")
df_r0 = df[np.isclose(df["simi_ratio"], 1)]
plt.scatter(
    np.log10(df_r0["r2"]), df_r0["query_time"], c="green", label="Tribase (r=0)"
)
plt.legend()
plt.savefig(f"/home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/plot.png")

print(
    f"save plot to /home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/plot.png"
)
