#include <cstdlib>

void require(const bool cond) 
{
    if (!cond) std::abort();
}