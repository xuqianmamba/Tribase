## Quick Start

```bash
cmake -B debug .
# cmake -B release -DCMAKE_BUILD_TYPE=Release .
cd debug
make -j
./bin/main --help
```