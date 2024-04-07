import numpy as np
import faiss
import time
import os.path as osp
import sys
import os
import pandas as pd
from tqdm import tqdm
from functools import reduce
from operator import or_
import json
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


xuqian = "/home/xuqian/Triangle"


def load_fvecs(file_path, bounds=None):
    if osp.exists(file_path + ".npy"):
        return np.load(file_path + ".npy")

    with open(file_path, "rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        vecsizeof = 1 * 4 + d * 4

        f.seek(0, 2)
        a = 1
        bmax = f.tell() // vecsizeof
        b = bmax

        if bounds is not None:
            if len(bounds) == 1:
                b = bounds[0]
            elif len(bounds) == 2:
                a = bounds[0]
                b = bounds[1]

        assert a >= 1
        if b > bmax:
            b = bmax

        if b == 0 or b < a:
            return np.array([])

        n = b - a + 1
        f.seek((a - 1) * vecsizeof, 0)
        v = np.fromfile(f, dtype=np.float32, count=(d + 1) * n)
        v = v.reshape((d + 1, n), order="F")
        assert np.sum(v[0, 1:] == v[0, 0]) == n - 1

    ret = v.T
    np.save(file_path + ".npy", ret)
    return ret


def load_bvecs(file_path, bounds=None):
    if osp.exists(file_path + ".npy"):
        return np.load(file_path + ".npy")

    with open(file_path, "rb") as f:
        # 读取向量的维度
        d = np.fromfile(f, dtype=np.int32, count=1)[0]
        vecsizeof = (
            1 * 4 + d
        )  # 每个向量的大小（包括维度信息），对于bvecs，每个元素占1字节

        # 获取向量的数量
        f.seek(0, 2)  # 移动到文件末尾
        a = 1
        bmax = f.tell() // vecsizeof
        b = bmax

        if bounds is not None:
            if len(bounds) == 1:
                b = bounds[0]
            elif len(bounds) == 2:
                a = bounds[0]
                b = bounds[1]

        assert a >= 1
        if b > bmax:
            b = bmax

        if b == 0 or b < a:
            return np.array([])

        # 计算实际读取的向量数量并移动到起始位置
        n = b - a + 1
        f.seek((a - 1) * vecsizeof, 0)

        # 读取n个向量，对于bvecs，我们读取无符号8位整数
        v = np.fromfile(f, dtype=np.uint8, count=d * n)
        v = v.reshape((n, d))  # 重塑数组，每行一个向量
        v = v.astype(np.float32)  # 将向量的元素类型转换为float3
        np.save(file_path + ".npy", v)

    return v


def load_vecs(file_path, bounds=None):
    if file_path.endswith(".fvecs"):
        return load_fvecs(file_path, bounds)
    elif file_path.endswith(".bvecs"):
        return load_bvecs(file_path, bounds)
    else:
        raise ValueError("Unknown file format")


def output_result(result_path, I, D):
    with open(result_path, "w") as f:
        for i in range(I.shape[0]):
            for k_index in range(I.shape[1]):
                f.write(f"{I[i, k_index]} {D[i, k_index]:.6f} ")
            f.write("\n")


def get_skip_info():
    return json.load(open(os.environ["SKIP_LOGGING_PATH"], "r"))


def set_opt_level(level=0b1111):
    text = ""

    if level & 0b0001:
        os.environ.pop("DISABLE_TRIANGLE", None)
        text += "Triangle "
    else:
        os.environ["DISABLE_TRIANGLE"] = "1"

    if level & 0b0010:
        os.environ.pop("DISABLE_INTERSECTION", None)
        text += "Intersection "
    else:
        os.environ["DISABLE_INTERSECTION"] = "1"

    if level & 0b0100:
        os.environ.pop("DISABLE_SUB_KNN_L2", None)
        text += "SubKnnL2 "
    else:
        os.environ["DISABLE_SUB_KNN_L2"] = "1"

    if level & 0b1000:
        os.environ.pop("DISABLE_SUB_KNN_COS", None)
        text += "SubKnnCos "
    else:
        os.environ["DISABLE_SUB_KNN_COS"] = "1"
    
    os.environ["OPT_LEVEL"] = str(level)

    return "Enabled:" + text


def test_main(
    dataset="",
    nlist=0,
    k_list=[],
    nprobe_list=[],
    opt_levels=[],
    sub_k=100,
    sub_nlist = 20,
    sub_nprobe = 1,
    sub_sample = 0.1,
    train_only=False,
    loops=10,
):
    file_format = "fvecs"
    if dataset in ["sift1b"]:
        file_format = "bvecs"
    base_path = osp.join(
        xuqian, "benchmarks", dataset, "origin", f"{dataset}_base.{file_format}"
    )
    query_path = osp.join(
        xuqian, "benchmarks", dataset, "origin", f"{dataset}_query.{file_format}"
    )

    # 对 opt_levels 求二进制或
    max_opt_level = reduce(or_, opt_levels)
    set_opt_level(max_opt_level)
    print(f"train_opt_level: {max_opt_level}")

    xb = load_vecs(base_path)
    xq = load_vecs(query_path)
    nb = xb.shape[0]
    db = xb.shape[1]
    nq = xq.shape[0]
    print(f"base shape: {xb.shape}, query shape: {xq.shape}")

    if nlist < 0:
        nlist = int(np.sqrt(nb))
    if nprobe_list:
        if isinstance(nprobe_list, int) and nprobe_list == -1:
            nprobe_list = [nlist]
        else:
            nprobe_list = [nprobe for nprobe in nprobe_list if nprobe <= nlist]
    else:
        nprobe_list = [round(1.6**i) for i in range(100) if 1.6**i <= nlist]
        if nprobe_list[-1] != nlist:
            nprobe_list.append(nlist)

    # print(nlist)
    # print(np.array(nprobe_list)/nlist)

    quantizer = faiss.IndexFlatL2(db)
    index = faiss.IndexIVFwithDistance(
        quantizer, db, nlist, faiss.METRIC_L2, sub_k, sub_nlist, sub_nprobe, sub_sample
    )

    print("Training index...")
    start = time.time()
    index.train(xb)
    end = time.time()
    print(f"Training time: {end - start} seconds")

    print("Adding vectors to index...")
    start = time.time()
    index.add(xb)
    end = time.time()
    print(f"Add time: {end - start} seconds")

    if train_only:
        return

    os.environ["SKIP_LOGGING_PATH"] = osp.join(xuqian, "skip.json")

    bar = tqdm(total=len(k_list) * len(opt_levels) * sum(nprobe_list) * loops)
    for _ in range(loops):
        for k in k_list:
            groundtruth_generate_time = None
            max_nprobe = max(nprobe_list)
            groundtruth_path = osp.join(
                xuqian, "benchmarks", dataset, "result", f"groundtruth_{k}.txt"
            )
            if not osp.exists(groundtruth_path):
                index.nprobe = nlist
                set_opt_level(0b0000)
                start = time.time()
                D, I = index.search(xq, k)
                end = time.time()
                output_result(groundtruth_path, I, D)
                result_path = osp.join(
                    xuqian,
                    "benchmarks",
                    dataset,
                    "result",
                    f"result_np={nlist}_nl={nlist}_k={k}_opt={0}.txt",
                )
                output_result(result_path, I, D)
                groundtruth_generate_time = end - start
            for opt_level in opt_levels:
                tqdm.write(set_opt_level(opt_level))
                for nprobe in nprobe_list:
                    result_path = osp.join(
                        xuqian,
                        "benchmarks",
                        dataset,
                        "result",
                        f"result_np={nprobe}_nl={nlist}_k={k}_opt={opt_level}.txt",
                    )
                    if nprobe > nlist:
                        continue
                    # if nprobe > max_nprobe and opt_level == 0b0000: # only skip faiss, not ours
                    #     continue
                    if not (nprobe == nlist and opt_level == 0 and groundtruth_generate_time):
                        index.nprobe = nprobe
                        start = time.time()
                        D, I = index.search(xq, k)
                        end = time.time()
                        query_time = end - start
                        output_result(result_path, I, D)
                    else:
                        query_time = groundtruth_generate_time

                    data_dict = get_skip_info()
                    data = list(data_dict.values())
                    avg_similarity, _ = run_similarity(result_path, groundtruth_path)
                    bar.update(nprobe)
                    total = data[-1]
                    skip_total = sum(data[:-1])

                    ret_dict = {
                        "dataset": dataset,
                        "k": k,
                        "nlist": nlist,
                        "nprobe": nprobe,
                        "opt_level": opt_level,
                        "avg_similarity": avg_similarity,
                        "query_time": query_time,
                        "query_num": nq,
                    }
                    ret_dict.update(data_dict)
                    pruning_speedup = (
                        1 if total == skip_total else total / (total - skip_total)
                    )
                    ret_dict.update({"pruning-speedup": pruning_speedup})
                    yield ret_dict
                    if avg_similarity == 1:
                        max_nprobe = min(max_nprobe, nprobe)


def parse_line(line):
    elements = line.strip().split()
    return [
        (int(elements[i]), float(elements[i + 1])) for i in range(0, len(elements), 2)
    ]


def calculate_similarity(line1, line2):
    list1 = parse_line(line1)
    list2 = parse_line(line2)
    match_count = 0

    # 获取文件2中所有距离的最大值
    dist2_max = max(dist for _, dist in list2) if list2 else float("0")
    # 使用一个非常小的值来比较距离
    epsilon = sys.float_info.epsilon

    # 创建一个集合，包含文件2中所有的ID
    ids2 = set(id for id, _ in list2)

    for id1, dist1 in list1:
        if id1 in ids2 or dist1 <= dist2_max or abs(dist1 - dist2_max) <= epsilon:
            match_count += 1

    return match_count / len(list1) if list1 else 0


def similarity(file1_path, file2_path):
    with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
        total_similarity = 0
        line_count = 0
        total_count = 0  # 添加一个变量来跟踪总数

        for line1, line2 in zip(f1, f2):
            list1 = parse_line(line1)
            list2 = parse_line(line2)
            total_similarity += calculate_similarity(line1, line2)
            line_count += 1
            total_count += len(list1)  # 更新总数

        # 计算平均相似度
        avg_similarity = total_similarity / line_count if line_count else 0
        return avg_similarity, total_count  # 返回平均相似度和总数


def run_similarity(file1_path, file2_path):
    avg_similarity, total_count = similarity(
        file1_path, file2_path
    )  # 获取平均相似度和总数
    return avg_similarity, total_count


# datasets = [
#     "deep1M",
#     "gist",
#     "tiny5m",
#     "glove2.2m",
#     "word2vec",
#     "msong",
#     "sift10k",
#     "sift1m",
#     "sift10m",
#     "20_mean",
#     "128_random",
#     "imagenet",
#     "ukbench",
#     "mnist_784",
#     "nuswide",
#     "nytimes",
#     "fasion_mnist_784",
#     "audio",
#     "cifar60k",
# ]

datasets = [
    # "sift10k",
    "msong",
    "sift1m",
    "imagenet",
    "ukbench",
    "nuswide",
    "fasion_mnist_784",
    "audio",
]


def analyze(df: pd.DataFrame):
    df = df.copy()
    df = df[np.isclose(df["avg_similarity"], 1)]

    idx = sum(df.groupby(["dataset", "nlist", "opt_level"], group_keys=False)
    .apply(
        lambda x: [x["nprobe"].idxmin(), x["nprobe"].idxmax()], include_groups=False
    )
    .reset_index(drop=True).values, [])

    df = df.loc[idx].reindex()

    skip_start_col_index = df.columns.get_loc("total_skip_count")
    skip_end_col_index = df.columns.get_loc("total_count")
    df.insert(
        skip_end_col_index,
        "skip_total",
        df.iloc[:, skip_start_col_index:skip_end_col_index].sum(axis=1),
    )
    
    df.iloc[:, skip_start_col_index:skip_end_col_index] = df.iloc[
        :, skip_start_col_index:skip_end_col_index
    ].values / df["skip_total"].apply(lambda x: x if x > 0 else 1).values.reshape(-1, 1)
    
    df.iloc[:, skip_start_col_index:skip_end_col_index] = \
    df.iloc[
        :, skip_start_col_index:skip_end_col_index
    ].map(lambda x: f"{x:.2%}" if x > 0 else "-")

    df.drop(["avg_similarity", "skip_total", "total_count"], axis=1, inplace=True)
    return df

def precheck_datasets(datasets):
    for dataset in datasets:
        base_path = osp.join(
            xuqian, "benchmarks", dataset, "origin", f"{dataset}_base.fvecs"
        )
        query_path = osp.join(
            xuqian, "benchmarks", dataset, "origin", f"{dataset}_query.fvecs"
        )
        if not osp.exists(base_path) or not osp.exists(query_path):
            print(f"Dataset {dataset} not found")
            return False
    return True

def main(loops = 10):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    LOG_PATH = osp.join(xuqian, "log3.txt")
    if False:
        os.environ["ENABLE_LITE_SUB_KNN"] = 1

    os.environ["LOG_INTERVAL"] = "2"  # seconds

    with open(LOG_PATH, "a+") as f:
        f.write(timestamp + "\n")
    datas = []
    for dataset in datasets:
        for nlist in [-1]:
            for data in test_main(
                dataset,
                nlist,
                k_list=[100],
                nprobe_list=None,
                opt_levels=[
                    0,
                    0b0001,
                    0b0100,
                    0b1000,
                    0b0101,
                    0b1001,
                    0b1101,
                ],
                sub_k=50,
                sub_nlist=1,
                sub_nprobe=1,
                sub_sample=0.02,
                # train_only=True
            ):
                for d in datas:
                    if (
                        d["dataset"] == data["dataset"]
                        and d["nlist"] == data["nlist"]
                        and d["k"] == data["k"]
                        and d["nprobe"] == data["nprobe"]
                        and d["opt_level"] == 0
                    ):
                        data.update(
                            {"time-speedup": d["query_time"] / data["query_time"]}
                        )
                        break
                if not "time-speedup" in data:
                    if data["opt_level"] == 0:
                        data.update({"time-speedup": 1})
                    else:
                        data.update({"time-speedup": -1})
                tqdm.write(str(data))
                with open(LOG_PATH, "a+") as f:
                    f.write(str(data) + "\n")
                datas.append(data)

    df = pd.DataFrame(datas)
    print(df)
    df.to_csv(f"{timestamp}_result.csv", index=False)
    df_a = analyze(df)
    print(df_a)
    df_a.to_csv(f"{timestamp}_result_a.csv", index=False)


main()
# precheck_datasets(datasets)
