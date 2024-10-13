#include "utils.h"

int main() {
    std::string filename = "test.csv";
    tribase::CsvWriter csv(filename, {"name", "age"});
    csv << "Tom" << 20 << std::endl;
    csv << "Jerry" << 22 << std::endl;
}