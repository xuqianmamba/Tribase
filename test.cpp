#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
int main() {
    // std::cout << sizeof(bool) << std::endl;
    bool x[100];
    memset(x, 2, 100);
    for (int i = 0; i < 100; i++) {
        std::cout << x[i] << std::endl;
    }
}