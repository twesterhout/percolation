#include "detail/shuffle.hpp"

TCM_NAMESPACE_BEGIN

template class shuffler_t<short, std::mt19937>;
template class shuffler_t<int, std::mt19937>;
template class shuffler_t<unsigned, std::mt19937>;
template class shuffler_t<long, std::mt19937>;
template class shuffler_t<unsigned long, std::mt19937>;

TCM_NAMESPACE_END

int main() {}
