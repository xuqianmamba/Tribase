cmake -B build -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF ..
make -C build -j swigfaiss
make -C build -j faiss


%extend faiss::IndexIVFwithDistance {
    %include "faiss/IndexIVFFlat.h"
}