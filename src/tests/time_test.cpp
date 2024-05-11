#include "gtest/gtest.h"
#include "tribase.h"

TEST(TIME, TEST) {
    using namespace tribase;
    Stopwatch sw;
    sleep(2);
    double s = sw.elapsedSeconds();
    std::cout << "Time elapsed: " << s << " s" << std::endl;
    EXPECT_FLOAT_EQ(s, 2.0);
}