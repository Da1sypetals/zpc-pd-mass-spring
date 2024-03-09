#include <mylib/foo.cuh>

namespace mylib {
// explicit instantiation of foo<int>
template void foo<int>();
} // namespace mylib