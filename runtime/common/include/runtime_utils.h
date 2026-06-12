#ifndef SCULPTOR_RUNTIME_UTILS_H
#define SCULPTOR_RUNTIME_UTILS_H

#include <cstdint>

template <typename T, int Rank>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  std::int64_t offset;
  std::int64_t sizes[Rank];
  std::int64_t strides[Rank];
};

using MemRef2D = StridedMemRefType<float, 2>;
using MemRef3D = StridedMemRefType<float, 3>;
using MemRef4D = StridedMemRefType<float, 4>;
using MemRef5D = StridedMemRefType<float, 5>;

#endif // SCULPTOR_RUNTIME_UTILS_H
