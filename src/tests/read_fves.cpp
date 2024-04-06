#include <iostream>
#include <fstream>

int main() {
    std::string filePath = "/home/xuqian/Triangle/Tribase/src/tests/iris.fvecs"; // 替换为你的.fvecs文件路径

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "无法打开文件" << std::endl;
        return -1;
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    int dimension;
    if (!file.read(reinterpret_cast<char*>(&dimension), 4)) {
        std::cerr << "读取维度失败" << std::endl;
        return -1;
    }

    // 每个向量的总字节数 = 4字节的维度 + 维度*d的浮点数
    int bytesPerVector = 4 + dimension * 4;
    int numVectors = fileSize / bytesPerVector;

    std::cout << "文件中的向量数量: " << numVectors << std::endl;

    return 0;
}