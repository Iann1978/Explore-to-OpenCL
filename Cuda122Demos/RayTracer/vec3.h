#ifndef VEC3_H
#define VEC3_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using std::sqrt;

class vec3 {
public:
    double e[3];

    __device__ vec3() {}
    //__host__ vec3() : e{ 0,0,0 } {}
    __device__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    __device__ double x() const { return e[0]; }
    __device__ double y() const { return e[1]; }
    __device__ double z() const { return e[2]; }

    __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    //double operator[](int i) const { return e[i]; }
    //double& operator[](int i) { return e[i]; }

    //vec3& operator+=(const vec3& v) {
    //    e[0] += v.e[0];
    //    e[1] += v.e[1];
    //    e[2] += v.e[2];
    //    return *this;
    //}

    __device__ vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ vec3& operator/=(double t) {
        return *this *= 1 / t;
    }

    __device__ double length() const {
        return sqrt(length_squared());
    }

    __device__ double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions
//
//inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
//    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
//}

__device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

//inline vec3 operator*(const vec3& u, const vec3& v) {
//    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
//}
//
__device__ inline vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__device__ inline vec3 operator/(vec3 v, double t) {
    return (1 / t) * v;
}

__device__ inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}
//
//inline vec3 cross(const vec3& u, const vec3& v) {
//    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
//        u.e[2] * v.e[0] - u.e[0] * v.e[2],
//        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
//}
//
__device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif