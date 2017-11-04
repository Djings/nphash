#include <stdio.h>


template<typename T>
unsigned long fnv1a(const T &number, unsigned long hash = 0xcbf29ce484222325) {
    const unsigned char *byte_of_data = reinterpret_cast<const unsigned char*>(&number);
    for (size_t i = 0; i < sizeof(T); ++i) {
        hash = hash * 0x100000001b3;
        hash = hash ^ (*byte_of_data);
        ++byte_of_data;
    }
    return hash;
}

