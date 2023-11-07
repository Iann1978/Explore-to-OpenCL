#ifndef RAY_H
#define RAY_H
#include <cuda.h>
#include <cuda_runtime.h>

#include "vec3.h"

class ray {
public:
    __device__ ray() {}

    __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    //point3 origin() const { return orig; }
    __device__ vec3 direction() const { return dir; }

    //point3 at(double t) const {
    //    return orig + t * dir;
    //}

private:
    point3 orig;
    vec3 dir;
};

#endif