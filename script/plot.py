import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = "sift10k"

df = pd.read_csv(f"/home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/log.csv")
df_f = pd.read_csv(
    f"/home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/faiss_result_nlist_100.txt",
    sep=" ",
    header=None,
)
df_f.columns = ["nprobes", "query_time", "recall", "r2"]
plt.scatter(np.log(df["r2"]), df["query_time"], c="blue", label="Tribase")
plt.scatter(np.log(df_f["r2"]), df_f["query_time"], c="red", label="Faiss")
df_r0 = df[np.isclose(df["simi_ratio"], 1)]
plt.scatter(np.log(df_r0["r2"]), df_r0["query_time"], c="green", label="Tribase (r=0)")
plt.legend()
plt.savefig(f"/home/panjunda/tribase/Tribase/benchmarks/{dataset}/result/plot.png")
