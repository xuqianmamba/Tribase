cmake -B release -DCMAKE_BUILD_TYPE=Release . && \
    cmake -B build . && \
    cmake -B debug -DCMAKE_BUILD_TYPE=Debug . && \
    cmake --build release -j && \
    cmake --build build -j && \
    cmake --build debug -j