#pragma once

#ifndef __USE_OMP__
    #define OMP_GET_MAX_THREADS() 1
#else
    #include <omp.h>
    #define OMP_GET_MAX_THREADS() omp_get_max_threads()
#endif

#ifndef __USE_OMP__
#define OMP_GET_THREAD_NUM() 0
#else
#include <omp.h>
#define OMP_GET_THREAD_NUM() omp_get_thread_num()
#endif

#ifdef _MSC_VER
    #define forceinline __forceinline
#elif defined(__GNUC__)
    #define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
    #if __has_attribute(__always_inline__)
        #define forceinline inline __attribute__((__always_inline__))
    #else
        #define forceinline inline
    #endif
#else
    #define forceinline inline
#endif

#ifndef __USE_OMP__
#define OMP_GET_WTIME() 0
#else
#define OMP_GET_WTIME() omp_get_wtime()
#endif


#ifdef _MSC_VER
    #define PRAGMA(opt) __pragma(opt)
#else
    #define PRAGMA(opt) _Pragma(#opt)
#endif


#if _OPENMP >= 201307
    #define OMP_FOR()  PRAGMA(omp parallel for)
    #define OMP_FOR_COLLAPSE()  PRAGMA(omp parallel for collapse(2))
    #define OMP_FOR_SIMD()  PRAGMA(omp parallel for simd)
    #define OMP_SIMD()  PRAGMA(omp simd)
#else
    #define OMP_FOR()  PRAGMA(omp parallel for)
    #define OMP_FOR_COLLAPSE()  PRAGMA(omp parallel for)
    #define OMP_FOR_SIMD()  PRAGMA(omp parallel for)
    #define OMP_SIMD()  PRAGMA(ivdep)
#endif
    
