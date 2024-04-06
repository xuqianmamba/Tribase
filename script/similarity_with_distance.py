import sys

def parse_line(line):
    # 修改以解析包含ID和距离的行
    elements = line.strip().split()
    return [(int(elements[i]), float(elements[i + 1])) for i in range(0, len(elements), 2)]


def calculate_similarity(line1, line2):
    list1 = parse_line(line1)
    list2 = parse_line(line2)
    match_count = 0

    # 获取文件2中所有距离的最大值
    dist2_max = max(dist for _, dist in list2) if list2 else float('0')
    # print (dist2_max)
    # 使用一个非常小的值来比较距离
    epsilon = sys.float_info.epsilon

    # 创建一个集合，包含文件2中所有的ID
    ids2 = set(id for id, _ in list2)

    for id1, dist1 in list1:
        if id1 in ids2 or dist1 <= dist2_max or abs(dist1 - dist2_max) <= epsilon:
            match_count += 1

    return match_count / len(list1) if list1 else 0




def main(file1_path, file2_path):
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
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

if __name__ == '__main__':
    # file1_path = '/home/xuqian/vector_dataset/script/resultsFAISSIVF_new.txt'
    
    #msong
    # file1_path = '/home/xuqian/Triangle/benchmarks/msong/10000truth.txt'
    # file2_path = '/home/xuqian/Triangle/benchmarks/msong/msong1000_1000_new.txt'

    file1_path = '/home/xuqian/Triangle/Tribase/results/results.txt'
    file2_path = '/home/xuqian/Triangle/benchmarks/sift1m/result/groundtruth_100.txt'

    # file1_path = '/home/xuqian/Triangle/benchmarks/msong/my100_1000_new.txt'
    # file2_path = '/home/xuqian/Triangle/benchmarks/msong/groundtruth_new.txt'
    # file2_path = '/home/xuqian/vector_dataset/script/groundtruth.txt'
    avg_similarity, total_count = main(file1_path, file2_path)  # 获取平均相似度和总数
    print(f"平均相似度: {avg_similarity * 100:.4f}%")
    print(f"总数: {total_count}")  # 打印总数
