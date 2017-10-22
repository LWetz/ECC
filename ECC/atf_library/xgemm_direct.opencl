
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is a generic GEMM kernel that works for all sizes and configurations: it doesn't require any
// pre and and post-processing kernels.
//
// This kernel is seperated into three files. This is part 1 out of 3.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif

// =================================================================================================

// Enable support for double-precision
#if PRECISION == 16
  #pragma OPENCL EXTENSION cl_khr_fp16: enable
#endif

// Enable support for double-precision
#if PRECISION == 64 || PRECISION == 6464
  #if __OPENCL_VERSION__ <= CL_VERSION_1_1
     #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #endif
#endif

// Half-precision
#if PRECISION == 16
  typedef half real;
  typedef half2 real2;
  typedef half4 real4;
  typedef half8 real8;
  typedef half16 real16;
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Double-precision 
#elif PRECISION == 64
  typedef double real;
  typedef double2 real2;
  typedef double4 real4;
  typedef double8 real8;
  typedef double16 real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37

// Complex single-precision
#elif PRECISION == 3232
  typedef struct cfloat {float x; float y;} real;
  typedef struct cfloat2 {real x; real y;} real2;
  typedef struct cfloat4 {real x; real y; real z; real w;} real4;
  typedef struct cfloat8 {real s0; real s1; real s2; real s3;
                          real s4; real s5; real s6; real s7;} real8;
  typedef struct cfloat16 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;
                           real s8; real s9; real sA; real sB;
                           real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Complex double-precision
#elif PRECISION == 6464
  typedef struct cdouble {double x; double y;} real;
  typedef struct cdouble2 {real x; real y;} real2;
  typedef struct cdouble4 {real x; real y; real z; real w;} real4;
  typedef struct cdouble8 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;} real8;
  typedef struct cdouble16 {real s0; real s1; real s2; real s3;
                            real s4; real s5; real s6; real s7;
                            real s8; real s9; real sA; real sB;
                            real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37
#endif

// Single-element version of a complex number
#if PRECISION == 3232
  typedef float singlereal;
#elif PRECISION == 6464
  typedef double singlereal;
#else
  typedef real singlereal;
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
  typedef float real_arg;
  #define GetRealArg(x) (half)x
#else
  typedef real real_arg;
  #define GetRealArg(x) x
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cc).
#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
  #define IsZero(a) (a == ZERO)
#endif

// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)
#else
  #define AbsoluteValue(value) value = fabs(value)
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c, a, b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c, a, b) c = a + b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a, b) a.x*b.x - a.y*b.y
  #define MulImag(a, b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c, a, b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c, a, b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c, a, b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #if USE_CL_MAD == 1
    #define MultiplyAdd(c, a, b) c = mad(a, b, c)
  #else
    #define MultiplyAdd(c, a, b) c += a * b
  #endif
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e, a, b, c, d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e, a, b, c, d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value) 
#endif

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
  #define USE_STAGGERED_INDICES 0
#endif

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1
  inline size_t GetGroupIDFlat() {
    return get_group_id(0) + get_num_groups(0) * get_group_id(1);
  }
  inline size_t GetGroupID1() {
    return (GetGroupIDFlat()) % get_num_groups(1);
  }
  inline size_t GetGroupID0() {
    return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
  }
#else
  inline size_t GetGroupID1() { return get_group_id(1); }
  inline size_t GetGroupID0() { return get_group_id(0); }
#endif

// =================================================================================================


// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library. Note that all parameters here have a
// suffix 'D' to denote that they are for the 'direct' version of the GEMM kernel.
#ifndef WGD
  #define WGD 8      // Tile-size in dimension M, N, and K (e.g. 8, 16, 32, 64)
#endif
#ifndef MDIMCD
  #define MDIMCD 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMCD
  #define NDIMCD 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMAD
  #define MDIMAD 8    // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#endif
#ifndef NDIMBD
  #define NDIMBD 8    // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#endif
#ifndef KWID
  #define KWID 1      // Unroll factor of the WGD loop (smaller or equal than WGD)
#endif
#ifndef VWMD
  #define VWMD 1      // Vector width of matrices A and C
#endif
#ifndef VWND
  #define VWND 1      // Vector width of matrix B
#endif
#ifndef PADA
  #define PADA 1      // Local memory padding for matrix A
#endif
#ifndef PADB
  #define PADB 1      // Local memory padding for matrix B
#endif

// Helper parameters based on the above tuning parameters
#define MWID (WGD/MDIMCD)                // Work per work-item (M-dimension)
#define NWID (WGD/NDIMCD)                // Work per work-item (N-dimension)
#define KDIMAD ((MDIMCD*NDIMCD)/(MDIMAD)) // Re-shaped tile dimension of matrix A: KDIMAD * MDIMAD
#define KDIMBD ((MDIMCD*NDIMCD)/(NDIMBD)) // Re-shaped tile dimension of matrix B: KDIMBD * NDIMBD
#define MWAD (WGD/MDIMAD)                // Amount of loads-per-thread for matrix A (M-dimension)
#define KWAD (WGD/KDIMAD)                // Amount of loads-per-thread for matrix A (K-dimension)
#define KWBD (WGD/KDIMBD)                // Amount of loads-per-thread for matrix B (K-dimension)
#define NWBD (WGD/NDIMBD)                // Amount of loads-per-thread for matrix B (N-dimension)

// =================================================================================================

// Data-widths in dimension M
#if VWMD == 1
    typedef real realMD;
#elif VWMD == 2
    typedef real2 realMD;
#elif VWMD == 4
    typedef real4 realMD;
#elif VWMD == 8
    typedef real8 realMD;
#elif VWMD == 16
    typedef real16 realMD;
#endif

// Data-widths in dimension N
#if VWND == 1
    typedef real realND;
#elif VWND == 2
    typedef real2 realND;
#elif VWND == 4
    typedef real4 realND;
#elif VWND == 8
    typedef real8 realND;
#elif VWND == 16
    typedef real16 realND;
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
inline void InitAccRegistersDirect(real cpm[NWID][MWID]) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    #pragma unroll
    for (int ni=0; ni<NWID; ++ni) {
      SetToZero(cpm[ni][mi]);
    }
  }
}

// =================================================================================================

// Performs the actual computation: Cpm += Apm * Bpm
inline void MultiplyAccumulateDirect(real cpm[NWID][MWID], real apm[MWID], real bpm[NWID]) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {
      MultiplyAdd(cpm[ni][mi], apm[mi], bpm[ni]);
    }
  }
}

// =================================================================================================

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix.
inline void GlobalToPrivateDirectA(const __global real* restrict agms, real apm[MWID],
                                   const int a_ld, const int a_offset, const int idm, const int idk,
                                   const int a_transpose, const int a_conjugate) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    const int a_index = (a_transpose) ? (idm + mi)*a_ld + idk : idk*a_ld + (idm + mi);
    apm[mi] = agms[a_index + a_offset];
    if (a_conjugate) { COMPLEX_CONJUGATE(apm[mi]); }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToPrivateDirectB(const __global real* restrict bgms, real bpm[NWID],
                                   const int b_ld, const int b_offset, const int idn, const int idk,
                                   const int b_transpose, const int b_conjugate) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    const int b_index = (b_transpose) ? (idn + ni)*b_ld + idk : idk*b_ld + (idn + ni);
    bpm[ni] = bgms[b_index + b_offset];
    if (b_conjugate) { COMPLEX_CONJUGATE(bpm[ni]); }
  }
}

// Loads global off-chip memory into thread-private register files. This function is specific for
// loading the A input matrix. This is the same as above but now includes a bounds check.
inline void GlobalToPrivateCheckedA(const __global real* restrict agms, real apm[MWID],
                                    const int a_ld, const int a_offset, const int idm, const int idk,
                                    const int a_transpose, const int a_conjugate,
                                    const int kSizeM) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    if (idm + mi < kSizeM) {
      const int a_index = (a_transpose) ? (idm + mi)*a_ld + idk : idk*a_ld + (idm + mi);
      apm[mi] = agms[a_index + a_offset];
      if (a_conjugate) { COMPLEX_CONJUGATE(apm[mi]); }
    }
    else {
      SetToZero(apm[mi]);
    }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToPrivateCheckedB(const __global real* restrict bgms, real bpm[NWID],
                                    const int b_ld, const int b_offset, const int idn, const int idk,
                                    const int b_transpose, const int b_conjugate,
                                    const int kSizeN) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    if (idn + ni < kSizeN) {
      const int b_index = (b_transpose) ? (idn + ni)*b_ld + idk : idk*b_ld + (idn + ni);
      bpm[ni] = bgms[b_index + b_offset];
      if (b_conjugate) { COMPLEX_CONJUGATE(bpm[ni]); }
    }
    else {
      SetToZero(bpm[ni]);
    }
  }
}

// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
inline void LocalToPrivateDirectA(__local real* alm, real apm[MWID], const int kg,
                                  const int a_transpose) {
  #pragma unroll
  for (int mi=0; mi<MWID; ++mi) {
    const int mg = mi + get_local_id(0)*MWID;
    const int index = (a_transpose) ? mg*(WGD + PADA) + kg : kg*(WGD + PADA) + mg;
    apm[mi] = alm[index];
  }
}

// Same as above, but now for the B input matrix
inline void LocalToPrivateDirectB(__local real* blm, real bpm[NWID], const int kg,
                                  const int b_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    const int ng = ni + get_local_id(1)*NWID;
    const int index = (b_transpose) ? ng*(WGD + PADB) + kg : kg*(WGD + PADB) + ng;
    bpm[ni] = blm[index];
  }
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResultsDirect(__global real* cgm, real cpm[NWID][MWID],
                               const int idm, const int idn,
                               const real alpha, const real beta,
                               const int c_ld, const int c_offset, const int c_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {

      // Determines the destination index
      int c_index = (c_transpose) ? (idm + mi)*c_ld + (idn + ni) : (idn + ni)*c_ld + (idm + mi);

      // The final multiplication with alpha (in case beta == 0)
      real result;
      if (IsZero(beta)) {
        Multiply(result, alpha, cpm[ni][mi]);
      }
      // The final multiplication with alpha and the addition with beta*C
      else {
        AXPBY(result, alpha, cpm[ni][mi], beta, cgm[c_index + c_offset]);
      }
      cgm[c_index + c_offset] = result;
    }
  }
}

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
inline void StoreResultsChecked(__global real* cgm, real cpm[NWID][MWID],
                                const int idm, const int idn, const int kSizeM, const int kSizeN,
                                const real alpha, const real beta,
                                const int c_ld, const int c_offset, const int c_transpose) {
  #pragma unroll
  for (int ni=0; ni<NWID; ++ni) {
    #pragma unroll
    for (int mi=0; mi<MWID; ++mi) {
      if ((idm + mi) < kSizeM && (idn + ni) < kSizeN) {

        // Determines the destination index
        int c_index = (c_transpose) ? (idm + mi)*c_ld + (idn + ni) : (idn + ni)*c_ld + (idm + mi);

        // The final multiplication with alpha (in case beta == 0)
        real result;
        if (IsZero(beta)) {
          Multiply(result, alpha, cpm[ni][mi]);
        }
        // The final multiplication with alpha and the addition with beta*C
        else {
          AXPBY(result, alpha, cpm[ni][mi], beta, cgm[c_index + c_offset]);
        }
        cgm[c_index + c_offset] = result;
      }
    }
  }
}

// =================================================================================================


// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
inline void GlobalToLocalDirectA(const __global realMD* restrict agm, __local real* alm,
                                 const int a_ld, const int a_offset, const int kwg,
                                 const int a_transpose, const int a_conjugate) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int mia=0; mia<MWAD/VWMD; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWAD; ++kia) {

      // Computes the indices for the global memory
      int mg = mia + la0*(MWAD/VWMD);
      int kg = kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg/VWMD : mg + GetGroupID0()*(WGD/VWMD);
      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realMD avec = agm[idk*(a_ld/VWMD) + idm + a_offset];
      #if VWMD == 1
         alm[kg*(WGD + PADA) + mg] = avec;
      #elif VWMD == 2
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;
      #elif VWMD == 4
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.x;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.y;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.z;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.w;
      #elif VWMD == 8
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;
         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;
         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;
         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;
         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;
      #elif VWMD == 16
         alm[kg*(WGD + PADA) + mg*VWMD + 0] = avec.s0;
         alm[kg*(WGD + PADA) + mg*VWMD + 1] = avec.s1;
         alm[kg*(WGD + PADA) + mg*VWMD + 2] = avec.s2;
         alm[kg*(WGD + PADA) + mg*VWMD + 3] = avec.s3;
         alm[kg*(WGD + PADA) + mg*VWMD + 4] = avec.s4;
         alm[kg*(WGD + PADA) + mg*VWMD + 5] = avec.s5;
         alm[kg*(WGD + PADA) + mg*VWMD + 6] = avec.s6;
         alm[kg*(WGD + PADA) + mg*VWMD + 7] = avec.s7;
         alm[kg*(WGD + PADA) + mg*VWMD + 8] = avec.s8;
         alm[kg*(WGD + PADA) + mg*VWMD + 9] = avec.s9;
         alm[kg*(WGD + PADA) + mg*VWMD + 10] = avec.sA;
         alm[kg*(WGD + PADA) + mg*VWMD + 11] = avec.sB;
         alm[kg*(WGD + PADA) + mg*VWMD + 12] = avec.sC;
         alm[kg*(WGD + PADA) + mg*VWMD + 13] = avec.sD;
         alm[kg*(WGD + PADA) + mg*VWMD + 14] = avec.sE;
         alm[kg*(WGD + PADA) + mg*VWMD + 15] = avec.sF;
      #endif
      if (a_conjugate) {
        for (int vm=0; vm<VWMD; ++vm) {
          COMPLEX_CONJUGATE(alm[kg*(WGD + PADA) + mg*VWMD + vm]);
        }
      }
    }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToLocalDirectB(const __global realND* restrict bgm, __local real* blm,
                                 const int b_ld, const int b_offset, const int kwg,
                                 const int b_transpose, const int b_conjugate) {
  #if MDIMCD == NDIMBD
    const int lb0 = get_local_id(0);
    const int lb1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int kib=0; kib<KWBD; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWBD/VWND; ++nib) {

      // Computes the indices for the global memory
      int ng = nib + lb0*(NWBD/VWND);
      int kg = kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg/VWND : ng + GetGroupID1()*(WGD/VWND);
      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      const realND bvec = bgm[idk*(b_ld/VWND) + idn + b_offset];
      #if VWND == 1
         blm[kg*(WGD + PADB) + ng] = bvec;
      #elif VWND == 2
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;
      #elif VWND == 4
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.x;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.y;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.z;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.w;
      #elif VWND == 8
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;
         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;
         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;
         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;
         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;
      #elif VWND == 16
         blm[kg*(WGD + PADB) + ng*VWND + 0] = bvec.s0;
         blm[kg*(WGD + PADB) + ng*VWND + 1] = bvec.s1;
         blm[kg*(WGD + PADB) + ng*VWND + 2] = bvec.s2;
         blm[kg*(WGD + PADB) + ng*VWND + 3] = bvec.s3;
         blm[kg*(WGD + PADB) + ng*VWND + 4] = bvec.s4;
         blm[kg*(WGD + PADB) + ng*VWND + 5] = bvec.s5;
         blm[kg*(WGD + PADB) + ng*VWND + 6] = bvec.s6;
         blm[kg*(WGD + PADB) + ng*VWND + 7] = bvec.s7;
         blm[kg*(WGD + PADB) + ng*VWND + 8] = bvec.s8;
         blm[kg*(WGD + PADB) + ng*VWND + 9] = bvec.s9;
         blm[kg*(WGD + PADB) + ng*VWND + 10] = bvec.sA;
         blm[kg*(WGD + PADB) + ng*VWND + 11] = bvec.sB;
         blm[kg*(WGD + PADB) + ng*VWND + 12] = bvec.sC;
         blm[kg*(WGD + PADB) + ng*VWND + 13] = bvec.sD;
         blm[kg*(WGD + PADB) + ng*VWND + 14] = bvec.sE;
         blm[kg*(WGD + PADB) + ng*VWND + 15] = bvec.sF;
      #endif
      if (b_conjugate) {
        for (int vn=0; vn<VWND; ++vn) {
          COMPLEX_CONJUGATE(blm[kg*(WGD + PADB) + ng*VWND + vn]);
        }
      }
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs doesn't
// use the vector data-types.
inline void GlobalToLocalScalarA(const __global real* restrict agms, __local real* alm,
                                 const int a_ld, const int a_offset, const int kwg,
                                 const int a_transpose, const int a_conjugate) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int mia=0; mia<MWAD; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWAD; ++kia) {

      // Computes the indices for the global memory
      int mg = mia + la0*MWAD;
      int kg = kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;
      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      real result = agms[idk*a_ld + idm + a_offset];
      if (a_conjugate) { COMPLEX_CONJUGATE(result); }
      alm[kg*(WGD + PADA) + mg] = result;
    }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToLocalScalarB(const __global real* restrict bgms, __local real* blm,
                                 const int b_ld, const int b_offset, const int kwg,
                                 const int b_transpose, const int b_conjugate) {
  #if MDIMCD == NDIMBD
    const int lb0 = get_local_id(0);
    const int lb1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int kib=0; kib<KWBD; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWBD; ++nib) {

      // Computes the indices for the global memory
      int ng = nib + lb0*NWBD;
      int kg = kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;
      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      real result = bgms[idk*b_ld + idn + b_offset];
      if (b_conjugate) { COMPLEX_CONJUGATE(result); }
      blm[kg*(WGD + PADB) + ng] = result;
    }
  }
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix. In contrast to the functions above, this function performs bounds
// checks and doesn't use the vector data-types.
inline void GlobalToLocalCheckedA(const __global real* restrict agms, __local real* alm,
                                  const int a_ld, const int a_offset, const int kwg,
                                  const int a_transpose, const int a_conjugate,
                                  const int kSizeM, const int kSizeK) {
  #if MDIMCD == MDIMAD
    const int la0 = get_local_id(0);
    const int la1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int la0 = tid % MDIMAD;
    const int la1 = tid / MDIMAD;
  #endif
  #pragma unroll
  for (int mia=0; mia<MWAD; ++mia) {
    #pragma unroll
    for (int kia=0; kia<KWAD; ++kia) {

      // Computes the indices for the global memory
      int mg = mia + la0*MWAD;
      int kg = kia + la1*KWAD;
      int idm = (a_transpose) ? mg + kwg : mg + GetGroupID0()*WGD;
      int idk = (a_transpose) ? kg + GetGroupID0()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      int condition = (a_transpose) ? idm < kSizeK : idm < kSizeM;
      if (condition) {
        real result = agms[idk*a_ld + idm + a_offset];
        if (a_conjugate) { COMPLEX_CONJUGATE(result); }
        alm[kg*(WGD + PADA) + mg] = result;
      }
      else {
        SetToZero(alm[kg*(WGD + PADA) + mg]);
      }
    }
  }
}

// Same as above, but now for the B input matrix
inline void GlobalToLocalCheckedB(const __global real* restrict bgms, __local real* blm,
                                  const int b_ld, const int b_offset, const int kwg,
                                  const int b_transpose, const int b_conjugate,
                                  const int kSizeN, const int kSizeK) {
  #if MDIMCD == NDIMBD
    const int lb0 = get_local_id(0);
    const int lb1 = get_local_id(1);
  #else
    const int tid = get_local_id(0) + MDIMCD*get_local_id(1);
    const int lb0 = tid % NDIMBD;
    const int lb1 = tid / NDIMBD;
  #endif
  #pragma unroll
  for (int kib=0; kib<KWBD; ++kib) {
    #pragma unroll
    for (int nib=0; nib<NWBD; ++nib) {

      // Computes the indices for the global memory
      int ng = nib + lb0*NWBD;
      int kg = kib + lb1*KWBD;
      int idn = (b_transpose) ? ng + kwg : ng + GetGroupID1()*WGD;
      int idk = (b_transpose) ? kg + GetGroupID1()*WGD : kg + kwg;

      // Loads the data from global memory into the local memory
      int condition = (b_transpose) ? idn < kSizeK : idn < kSizeN;
      if (condition) {
        real result = bgms[idk*b_ld + idn + b_offset];
        if (b_conjugate) { COMPLEX_CONJUGATE(result); }
        blm[kg*(WGD + PADB) + ng] = result;
      }
      else {
        SetToZero(blm[kg*(WGD + PADB) + ng]);
      }
    }
  }
}

// =================================================================================================


// =================================================================================================

// Main body of the kernel. This is the direct version without pre/post processing and restrictions.
inline void XgemmDirect(const int kSizeM, const int kSizeN, const int kSizeK,
                        const real_arg arg_alpha,
                        const real_arg arg_beta,
                        const __global realMD* restrict agm, const int a_offset, const int a_ld,
                        const __global realND* restrict bgm, const int b_offset, const int b_ld,
                        __global real* cgm, const int c_offset, const int c_ld,
                        __local real* alm, __local real* blm,
                        const int a_transpose, const int b_transpose, const int c_transpose,
                        const int a_conjugate, const int b_conjugate) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Extra pointers to scalar versions of global memory
  const __global real* restrict agms = (const __global real* restrict) agm;
  const __global real* restrict bgms = (const __global real* restrict) bgm;

  // Allocates workitem-private memory (registers)
  real apm[MWID];
  real bpm[NWID];
  real cpm[NWID][MWID];

  // Initializes the accumulation registers
  InitAccRegistersDirect(cpm);

  // The faster version of GEMM is not allowed on the (incomplete) borders. Therefore, this section
  // processes only the main parts: output blocks of WGD by WGD.
  const int idm = get_local_id(0) * MWID + GetGroupID0() * WGD;
  const int idn = get_local_id(1) * NWID + GetGroupID1() * WGD;
  if ((idm < (kSizeM/WGD)*WGD) && (idn < (kSizeN/WGD)*WGD)) {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      if (a_ld % VWMD == 0) {
        GlobalToLocalDirectA(agm, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      else {
        GlobalToLocalScalarA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate);
      }
      if (b_ld % VWND == 0) {
        GlobalToLocalDirectB(bgm, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      else {
        GlobalToLocalScalarB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi=0; pwi<WGD; pwi+=KWID) {
        #pragma unroll
        for (int pit=0; pit<KWID; ++pit) {
          int kg = pwi + pit;

          // Loads data: local --> private (matrix A and B)
          LocalToPrivateDirectA(alm, apm, kg, a_transpose);
          LocalToPrivateDirectB(blm, bpm, kg, b_transpose);

          // Performs the accumulation (Cpm += Apm * Bpm)
          MultiplyAccumulateDirect(cpm, apm, bpm);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      GlobalToPrivateDirectA(agms, apm, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate);
      GlobalToPrivateDirectB(bgms, bpm, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate);

      // Performs the accumulation (Cpm += Apm * Bpm)
      MultiplyAccumulateDirect(cpm, apm, bpm);
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    StoreResultsDirect(cgm, cpm, idm, idn, alpha, beta, c_ld, c_offset, c_transpose);
  }

  // Simple but slower version for the parts on the edge (incomplete tiles in M and N-dimensions)
  else {

    // Loops over all complete workgroup tiles (K-dimension)
    int kwg = 0;
    for (; kwg < (kSizeK/WGD) * WGD; kwg+=WGD) {

      // Loads data: off-chip --> local (matrix A and B)
      GlobalToLocalCheckedA(agms, alm, a_ld, a_offset, kwg, a_transpose, a_conjugate, kSizeM, kSizeK);
      GlobalToLocalCheckedB(bgms, blm, b_ld, b_offset, kwg, b_transpose, b_conjugate, kSizeN, kSizeK);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Loops over all workitem tiles, unrolled by a factor KWID
      for (int pwi=0; pwi<WGD; pwi+=KWID) {
        #pragma unroll
        for (int pit=0; pit<KWID; ++pit) {
          int kg = pwi + pit;

          // Loads data: local --> private (matrix A and B)
          LocalToPrivateDirectA(alm, apm, kg, a_transpose);
          LocalToPrivateDirectB(blm, bpm, kg, b_transpose);

          // Performs the accumulation (Cpm += Apm * Bpm)
          MultiplyAccumulateDirect(cpm, apm, bpm);
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Loop over the remaining part (incomplete tile in K-dimension)
    for (; kwg < kSizeK; ++kwg) {

      // Loads data: off-chip --> private (matrix A and B)
      GlobalToPrivateCheckedA(agms, apm, a_ld, a_offset, idm, kwg, a_transpose, a_conjugate, kSizeM);
      GlobalToPrivateCheckedB(bgms, bpm, b_ld, b_offset, idn, kwg, b_transpose, b_conjugate, kSizeN);

      // Performs the accumulation (Cpm += Apm * Bpm)
      MultiplyAccumulateDirect(cpm, apm, bpm);
    }

    // Stores a tile of results and performs the multiplication with alpha and beta
    StoreResultsChecked(cgm, cpm, idm, idn, kSizeM, kSizeN, alpha, beta, c_ld, c_offset, c_transpose);
  }
}

// =================================================================================================

// Direct version of the GEMM kernel with [A, B] = [non-transposed, non-transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectNN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [non-transposed, transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectNT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 0, 1, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, non-transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectTN(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 0, c_transpose, a_conjugate, b_conjugate);
}

// Direct version of the GEMM kernel with [A, B] = [transposed, transposed]
__attribute__((reqd_work_group_size(MDIMCD, NDIMCD, 1)))
__kernel void XgemmDirectTT(const int kSizeM, const int kSizeN, const int kSizeK,
                            const real_arg arg_alpha, const real_arg arg_beta,
                            const __global realMD* restrict agm, const int a_offset, const int a_ld,
                            const __global realND* restrict bgm, const int b_offset, const int b_ld,
                            __global real* cgm, const int c_offset, const int c_ld,
                            const int c_transpose, const int a_conjugate, const int b_conjugate) {
  __local real alm[WGD * (WGD + PADA)];
  __local real blm[WGD * (WGD + PADB)];
  XgemmDirect(kSizeM, kSizeN, kSizeK, arg_alpha, arg_beta,
              agm, a_offset, a_ld, bgm, b_offset, b_ld, cgm, c_offset, c_ld,
              alm, blm, 1, 1, c_transpose, a_conjugate, b_conjugate);
}

// =================================================================================================


// End of the C++11 raw string literal
)"

// =================================================================================================
