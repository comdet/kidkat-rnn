#ifndef KIDKAT_CONFIG_H_INCLUDED
#define KIDKAT_CONFIG_H_INCLUDED

#include <list>
#include <vector>
#include <memory>
#include <string>
#include <cstring>
#include <sstream>
#include <map>
#include <unordered_map>
#include <math.h>
#include <climits>
#include <random>


#define TINYRNN_GRADIENT_CLIPPING_THRESHOLD 1.0


using Id = uint32_t;
using Index = uint32_t;
using Value = float;

#define TINYRNN_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;\
    TypeName &operator =(const TypeName &) = delete;


namespace Uuid
{
    
    static inline Id generateId()
    {
        static Id kRecentId = 0;
        return ++kRecentId;
    }
} // namespace Uuid;


#endif // KIDKAT_CONFIG_H_INCLUDED
