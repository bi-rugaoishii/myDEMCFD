#pragma once
#include <cuda_runtime.h>

template <class T,int Dim>
struct MyArray;

template<class T>
struct MyArray<T,2>{
    T* data_=nullptr;
    int sizex_=0;
    int sizey_=0;
    int size_=0;

    __host__ __device__ __forceinline__
        int ind(int x, int y)const{
            return y*sizex_+x;
        }

    __host__ __device__ __forceinline__
        T& operator[](int id)const{
            return data_[id];
        }

    __host__ __device__ __forceinline__
        T& operator()(int x, int y)const{
            return data_[y*sizex_+x];
        }

    __host__ __device__ __forceinline__
        T& operator()(int x)const{
            return data_[x];
        }
};

template<class T>
struct MyArray<T,3>{
    T* data_=nullptr;
    int sizex_=0;
    int sizey_=0;
    int sizez_=0;
    int size_=0;

    __host__ __device__ __forceinline__
        int ind(int x, int y, int z)const{
            return (z*sizey_+y)*sizex_+x;
        }

    __host__ __device__ __forceinline__
        T& operator ()(int x, int y, int z)const{
            return data_[(z*sizey_+y)*sizex_+x];
        }

    __host__ __device__ __forceinline__
        T& operator()(int x)const{
            return data_[x];
        }
};
