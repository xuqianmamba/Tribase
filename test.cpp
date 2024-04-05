#include <queue>
#include <utility>
using Heap = std::priority_queue<std::pair<float, int>>;
float update1(Heap h) {
    auto [x, _] = h.top();
    if (x < 100) {
        return x;
    }
    return 10;
}

float update2(float* xs, int* ys) {
    if (xs[0] < 100) {
        return xs[0];
    }
    return 10;
}