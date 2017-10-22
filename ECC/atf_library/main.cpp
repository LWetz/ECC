//
//  main.cpp
//  
//
//  Created by Ari Rasch on 28/10/16.
//
//

#include <stddef.h>
#include <string>
#include <array>
#include <math.h>
#include <type_traits>
#include <utility>
#include <random>


#include "atf.h"


#define CLTUNE_CONV_RICHARD
//#define CLTUNE_MATMULT_RICHARD

//#define TEST

//#define HARALD

//#define MD_HOM_GEMV
//#define MD_HOM_GEMM

//#define CLBLAST_SAXPY
//#define CLBLAST_GEMM_DIRECT
//#define CLBLAST_GEMV

//#define CLTUNE_CONV
//#define CLTUNE_MATMULT

//#define SEARCH_SPACE_GENERATION_TEST_USING_GEMM

//#define MD_HOM_GEMV

//#define OT_EXAMPLE




#ifdef CLTUNE_CONV_RICHARD

#define kSizeX  8192
#define kSizeY  4096

// Settings (synchronise these with "conv.opencl")
#define HFS (3)        // Half filter size
#define FS (HFS+HFS+1) // Filter size

int main() {
    const atf::cf::device_info device(
            "Apple",
            atf::cf::device_info::GPU,
            0
    );

    const std::string source =
#include "conv.opencl"
    ;

    // get device specific boundaries for work items and local memory
    cl_uint max_wi_dimensions = 0;
    device.device().getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &max_wi_dimensions);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << max_wi_dimensions << std::endl;
    size_t max_wi_sizes[3];
    device.device().getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_wi_sizes);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << max_wi_sizes[0] << ", " << max_wi_sizes[1] << ", " << max_wi_sizes[2] << std::endl;
    const size_t max_wi_size_0 = max_wi_sizes[0];
    const size_t max_wi_size_1 = max_wi_sizes[1];
    size_t max_wg_size = 0;
    device.device().getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_wg_size);
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << max_wg_size << std::endl;
    cl_ulong max_local_mem_size = 0;
    device.device().getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &max_local_mem_size);
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << max_local_mem_size << std::endl;

#if 1
    // TP ranges and constraints as defined in CLTune conv sample
    auto TBX            = atf::tp("TBX", {8, 16, 32, 64});
    auto TBY            = atf::tp("TBY", {8, 16, 32, 64});
    auto LOCAL          = atf::tp("LOCAL", {0, 1, 2});
    auto WPTX           = atf::tp("WPTX", {1, 2, 4, 8});
    auto WPTY           = atf::tp("WPTY", {1, 2, 4, 8});
    auto VECTOR         = atf::tp("VECTOR", {1, 2, 4},
                                  [&] (auto VECTOR) -> bool {
                                      if (LOCAL == 2) {
                                          return bool(WPTX % VECTOR == 0) && bool((2 * HFS) % VECTOR == 0);
                                      } else {
                                          return WPTX % VECTOR == 0;
                                      }
                                  });
    auto UNROLL_FACTOR  = atf::tp("UNROLL_FACTOR", {1, FS});
    auto PADDING        = atf::tp("PADDING", {0, 1},
                                  [&] (auto PADDING) {
                                      return PADDING == 0 || LOCAL != 0;
                                  }
                                  && [&](auto PADDING) -> bool {
                                      if (LOCAL != 0) {
                                          return ((TBY * WPTY + 2 * HFS) * (TBX * WPTX + 2 * HFS + PADDING)) * sizeof(float) <= max_local_mem_size;
                                      } else {
                                          return true;
                                      }
                                  });
    auto TBX_XL         = atf::tp("TBX_XL", {8, 9, 10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                             64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
                                  atf::less_than_or_eq(max_wi_size_0)
                                  && [&] (auto TBX_XL) -> bool {
                                      if (LOCAL == 2) {
                                          return TBX_XL == TBX + ((2 * HFS) + WPTX - 1) / WPTX;
                                      } else {
                                          return TBX_XL == TBX;
                                      }
                                  });
    auto TBY_XL         = atf::tp("TBY_XL", {8, 9, 10, 11, 12, 13, 14, 15,
                                             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                             64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74},
                                  atf::less_than_or_eq(max_wi_size_1)
                                  && [&] (auto TBY_XL) { return TBX_XL * TBY_XL <= max_wg_size; }
                                  && [&] (auto TBY_XL) -> bool {
                                      if (LOCAL == 2) {
                                          return TBY_XL == TBY + ((2 * HFS) + WPTY - 1) / WPTY;
                                      } else {
                                          return TBY_XL == TBY;
                                      }
                                  });
#elif 0
    // TODO larger TP ranges

#else
    auto TBX            = atf::tp("TBX",           {});
    auto TBY            = atf::tp("TBY",           {});
    auto LOCAL          = atf::tp("LOCAL",         {});
    auto WPTX           = atf::tp("WPTX",          {});
    auto WPTY           = atf::tp("WPTY",          {});
    auto VECTOR         = atf::tp("VECTOR",        {});
    auto UNROLL_FACTOR  = atf::tp("UNROLL_FACTOR", {});
    auto PADDING        = atf::tp("PADDING",       {});
    auto TBX_XL         = atf::tp("TBX_XL",        {});
    auto TBY_XL         = atf::tp("TBY_XL",        {});
#endif

    const auto kExtraSize = size_t{FS * 8};
    std::vector<float> mat_a((kExtraSize + kSizeX) * (kExtraSize + kSizeY)); //fill_ints(mat_a);
    std::vector<float> mat_b(kSizeX * kSizeY);                               //zero(mat_b);
    std::vector<float> coeff(FS * FS);                                       //zero(coeff);
    // Creates the filter coefficients (gaussian blur)
    auto sigma = 1.0f;
    auto mean = FS/2.0f;
    auto sum = 0.0f;
    for (auto x=size_t{0}; x<FS; ++x) {
        for (auto y=size_t{0}; y<FS; ++y) {
            auto exponent = -0.5f * (pow((x-mean)/sigma, 2.0f) + pow((y-mean)/sigma, 2.0f));
            coeff[y*FS + x] = static_cast<float>(exp(exponent) / (2.0f * 3.14159265f * sigma * sigma));
            sum += coeff[y*FS + x];
        }
    }
    for (auto &item: coeff) { item = item / sum; }

    auto kernel = atf::cf::ocl(device,
                               {source, "conv"},
                               atf::inputs(atf::scalar<int>(kSizeX),
                                           atf::scalar<int>(kSizeY),
                                           atf::buffer(mat_a),
                                           atf::buffer(coeff),
                                           atf::buffer(mat_b)),
                               atf::cf::GS(kSizeX * TBX_XL / TBX / WPTX, kSizeY * TBY_XL / TBY / WPTY),
                               atf::cf::LS(TBX_XL, TBY_XL)
                               );

#if 1
    auto tuner = atf::annealing_tree(atf::cond::evaluations(53));
//    auto tuner = atf::open_tuner(atf::cond::duration<std::chrono::minutes>(2850));
    // use new database for every run
//    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
//            std::chrono::system_clock::now().time_since_epoch()).count();
//    tuner.set_path_to_database("/scratch/tmp/r_schu41/opentuner_" + std::to_string(millis));
#elif 0
//    auto tuner = atf::open_tuner_flat(atf::cond::evaluations(117));
    auto tuner = atf::open_tuner_flat(atf::cond::duration<std::chrono::minutes>(2850));
    // use new database for every run
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    tuner.set_path_to_database("/scratch/tmp/r_schu41/opentuner_" + std::to_string(millis));
#elif 0
//    auto tuner = atf::annealing_tree(atf::cond::evaluations(117));
    auto tuner = atf::annealing_tree(atf::cond::duration<std::chrono::minutes>(2850));
#else
    auto tuner = atf::exhaustive(atf::cond::evaluations(1));
#endif

    tuner(G(TBX, TBY, LOCAL, WPTX, WPTY, VECTOR, PADDING, TBX_XL, TBY_XL))(UNROLL_FACTOR)(kernel);
}
#endif


#ifdef CLTUNE_MATMULT_RICHARD

#define N 1024
#define kSizeM  N //2048
#define kSizeN  N //2048
#define kSizeK  N //2048

int main()
{


#if 1 // CPU
  auto max_wi_size_0      = 8192;
  auto max_wi_size_1      = 8192;
  auto max_wg_size        = 8192;
  auto max_local_mem_size = 32768;
#else // GPU
  auto max_wi_size_0      = 1024;
  auto max_wi_size_1      = 1024;
  auto max_wg_size        = 1024;
  auto max_local_mem_size = 49152;
#endif


#if 0 // CLTune
    // TP ranges and constraints as defined in CLTune gemm sample
    auto MDIMC = atf::tp("MDIMC", {8, 16, 32},
                         atf::less_than_or_eq(max_wi_size_0));          // WI size constraint
    auto NDIMC = atf::tp("NDIMC", {8, 16, 32},
                         atf::less_than_or_eq(max_wi_size_1)            // WI size constraint
                         && [&](auto NDIMC) {
                             return MDIMC * NDIMC <= max_wg_size;       // WG size constraint
                         });
    auto MDIMA = atf::tp("MDIMA", {8, 16, 32});
    auto NDIMB = atf::tp("NDIMB", {8, 16, 32});
    auto KWG = atf::tp("KWG", {16, 32},
                       atf::multiple_of(MDIMC * NDIMC / MDIMA)
                       && atf::multiple_of(MDIMC * NDIMC / NDIMB));
    auto KWI = atf::tp("KWI", {2, 8}, atf::divides(KWG));
    auto VWM = atf::tp("VWM", {1, 2, 4, 8});
    auto VWN = atf::tp("VWN", {1, 2, 4, 8});
    auto STRM = atf::tp("STRM", {0, 1});
    auto STRN = atf::tp("STRN", {0, 1});
    auto SA = atf::tp("SA", {0, 1});
    auto SB = atf::tp("SB", {0, 1});
    auto MWG = atf::tp("MWG", {16, 32, 64, 128},
                       atf::multiple_of(MDIMC * VWM)
                       && atf::multiple_of(MDIMA * VWM));
    auto NWG = atf::tp("NWG", {16, 32, 64, 128},
                       atf::multiple_of(NDIMC * VWN)
                       && atf::multiple_of(NDIMB * VWN)
                       && [&] (auto NWG) {  // local memory constraint
                           return ((SA * KWG * MWG / VWM) + (SB * KWG * NWG / VWN)) * sizeof(float)
                                  <= max_local_mem_size;
                       });
    auto PRECISION = atf::tp("PRECISION", {32});
#elif 1 // CLTune
    // larger TP ranges
    auto MDIMC = atf::tp("MDIMC", atf::interval<size_t>(0, std::log2(kSizeM), atf::pow_2),
                         atf::less_than_or_eq(max_wi_size_0));          // WI size constraint
    auto NDIMC = atf::tp("NDIMC", atf::interval<size_t>(0, std::log2(kSizeN), atf::pow_2),
                         atf::less_than_or_eq(max_wi_size_1)            // WI size constraint
                         && [&](auto NDIMC) {
                             return MDIMC * NDIMC <= max_wg_size;       // WG size constraint
                         });
    auto MDIMA = atf::tp("MDIMA", atf::interval<size_t>(0, std::log2(kSizeM), atf::pow_2),
                         atf::divides(MDIMC * NDIMC));
    auto NDIMB = atf::tp("NDIMB", atf::interval<size_t>(0, std::log2(kSizeN), atf::pow_2),
                         atf::divides(MDIMC * NDIMC));
    auto KWG   = atf::tp("KWG", atf::interval<size_t>(0, std::log2(kSizeK), atf::pow_2),
                       atf::multiple_of(MDIMC * NDIMC / MDIMA)
                       && atf::multiple_of(MDIMC * NDIMC / NDIMB));
    auto KWI   = atf::tp("KWI", atf::interval<size_t>(0, std::log2(kSizeK), atf::pow_2),
                       atf::divides(KWG));
    auto VWM   = atf::tp("VWM", {1, 2, 4, 8});
    auto VWN   = atf::tp("VWN", {1, 2, 4, 8});
    auto STRM  = atf::tp("STRM", {0, 1});
    auto STRN  = atf::tp("STRN", {0, 1});
    auto SA    = atf::tp("SA", {0, 1});
    auto SB    = atf::tp("SB", {0, 1});
    auto MWG   = atf::tp("MWG", atf::interval<size_t>(0, std::log2(kSizeM), atf::pow_2),
                       atf::multiple_of(MDIMC * VWM)
                       && atf::multiple_of(MDIMA * VWM));
    auto NWG   = atf::tp("NWG", atf::interval<size_t>(0, std::log2(kSizeN), atf::pow_2),
                       atf::multiple_of(NDIMC * VWN)
                       && atf::multiple_of(NDIMB * VWN)
                       && [&] (auto NWG) {  // local memory constraint
                           return ((SA * KWG * MWG / VWM) + (SB * KWG * NWG / VWN)) * sizeof(float)
                                  <= max_local_mem_size;
                       });
    auto PRECISION = atf::tp("PRECISION", {32});
#else // testing
    auto MDIMC = atf::tp("MDIMC", {32});
    auto NDIMC = atf::tp("NDIMC", {4});
    auto MDIMA = atf::tp("MDIMA", {8});
    auto NDIMB = atf::tp("NDIMB", {16});
    auto KWG = atf::tp("KWG", {64});
    auto KWI = atf::tp("KWI", {8});
    auto VWM = atf::tp("VWM", {4});
    auto VWN = atf::tp("VWN", {8});
    auto STRM = atf::tp("STRM", {1});
    auto STRN = atf::tp("STRN", {1});
    auto SA = atf::tp("SA", {1});
    auto SB = atf::tp("SB", {1});
    auto MWG = atf::tp("MWG", {128});
    auto NWG = atf::tp("NWG", {128});
    auto PRECISION = atf::tp("PRECISION", {32});
#endif
 
  auto cf = [](auto){ return 0; };

  auto tuner = atf::open_tuner( atf::cond::evaluations(0) )
                              ( G(MDIMC, NDIMC, MDIMA, NDIMB, KWG, KWI, VWM, VWN, SA, SB, MWG, NWG),
                                G(STRM, STRN, PRECISION)
                              )
                              ( cf );
}
#endif


#ifdef TEST
int main()
{
  auto start = std::chrono::high_resolution_clock::now();
  auto TP_1 = atf::tp( "TP_1", atf::interval(1,10) );
  auto TP_2 = atf::tp( "TP_2", atf::interval(1,10) );
  auto TP_3 = atf::tp( "TP_3", atf::interval(1,10) );
  auto TP_4 = atf::tp( "TP_4", atf::interval(1,10) );
  
  auto cf = []( auto config )
            {
              for( auto& c : config )
                std::cout << c.second << std::endl;
              return 0;
            };
  auto best_config = atf::annealing_tree( atf::cond::evaluations(10) )
                                    ( TP_1, TP_2, TP_3, TP_4 )
                                    ( cf ) ;
  
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
std::cout << "Time to create search space: " << elapsed_milliseconds << std::endl;
}
#endif




#ifdef HARALD

#define CUDA   1
#define OpenCL 2

#define Linear1D  1
#define Linear2D  2
#define Array2D   3
#define Ldg       4
#define void_elem 5

int main()
{
  auto Padding        = atf::tp( "Padding"       , atf::interval<int>(1,16, [](auto i ){ return i*32; } )                                                                       );
  auto PixelPerThread = atf::tp( "PixelPerThread", atf::interval<int>(1,4)                                                                                                      );
  auto API            = atf::tp( "API"           , { CUDA, OpenCL }                                                                                                             );
  auto LocalMemory    = atf::tp( "LocalMemy"     , atf::interval<bool>()                                                                                                        );
  auto Blocksize_1    = atf::tp( "Blocksize_1"   , { 32, 64, 128, 256, 512, 1024 }                                                                                              );
  auto Blocksize_2    = atf::tp( "Blocksize_2"   , { 1, 2, 4, 8, 16, 32 }                                , [&](auto Blocksize_2){ return (Blocksize_1 * Blocksize_2) <= 1024; } );
  auto TextureMemory  = atf::tp( "TextureMemory" , { Linear1D, Linear2D, Array2D, Ldg, void_elem }       ,
  [&](auto TextureMemory)
  {
    return (!LocalMemory || !(Blocksize_1 == 1024) || !(Blocksize_2 == 1)) &&
           !( LocalMemory && Blocksize_1 == 1024 && Blocksize_2 == 1  && PixelPerThread == 2 ) &&
           !( LocalMemory && Blocksize_1 == 32   && Blocksize_2 == 32 && PixelPerThread == 3 ) &&
           !( LocalMemory && Blocksize_1 == 64   && Blocksize_2 == 16 && PixelPerThread == 3 ) &&
           !( LocalMemory && Blocksize_1 == 1024 && Blocksize_2 == 1  && PixelPerThread == 3 ) &&
           !( LocalMemory && Blocksize_1 == 32   && Blocksize_2 == 32 && PixelPerThread == 4 ) &&
           !( LocalMemory && Blocksize_1 == 64   && Blocksize_2 == 16 && PixelPerThread == 4 ) &&
    
           !( API == OpenCL && TextureMemory == Linear1D ) &&
           !( API == OpenCL && TextureMemory == Linear2D ) &&
           !( API == OpenCL && TextureMemory == Ldg      ) &&
    
           !( LocalMemory && Blocksize_1 == 128  && Blocksize_2 == 8  && PixelPerThread == 3 ) &&
           !( LocalMemory && Blocksize_1 == 256  && Blocksize_2 == 4  && PixelPerThread == 3 ) &&
           !( LocalMemory && Blocksize_1 == 512  && Blocksize_2 == 2  && PixelPerThread == 3 ) &&
           !( LocalMemory && Blocksize_1 == 128  && Blocksize_2 == 8  && PixelPerThread == 4 ) &&
           !( LocalMemory && Blocksize_1 == 256  && Blocksize_2 == 4  && PixelPerThread == 4 ) &&
           !( LocalMemory && Blocksize_1 == 512  && Blocksize_2 == 2  && PixelPerThread == 4 ) &&
           !( LocalMemory && Blocksize_1 == 1024 && Blocksize_2 == 1  && PixelPerThread == 4 ) &&
    
           !( TextureMemory == Array2D && !(Padding==0) );
  } );


  auto cf = [](auto){ return 1; };

  auto best_config = atf::open_tuner( atf::cond::speedup(1, 200) || atf::cond::evaluations(1000) )
                                    (
                                      Padding,
                                      PixelPerThread,
                                      API,
                                      LocalMemory,
                                      Blocksize_1,
                                      Blocksize_2,
                                      TextureMemory
                                    )
                                    ( cf );
}
#endif


#ifdef MD_HOM_GEMV
void fill( std::vector<float>& mat )
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0, 10.0);

  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = dist(mt);
}


void zero( std::vector<float>& mat )
{
  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = 0;//dist(mt);
}


int main()
{
  auto start = std::chrono::system_clock::now();
  
  std::string gemm_ocl_source =
                                #include "md_hom_gemv.cl"
                              ;
  
  const size_t M_exp = 2;
  const size_t N_exp = 2;
  
  const size_t M = pow(2, M_exp );
  const size_t N = pow(2, N_exp );

  std::vector<float> A( M * N ); fill( A );
  std::vector<float> x( N ); fill( x );
  std::vector<float> y( M ); zero( y );
  
  std::vector<float> y_gold( M );
  for( int i = 0 ; i < M ; ++i )
    for( int j = 0 ; j < N ; ++j )
        y_gold[ i ] += A[ i * M + j ] * x[ j ];


#if 0 // only for testing
  for( int i = 0 ; i < N_3 ; ++i )
    for( int j = 0 ; j < N_2 ; ++j )
      std::cout << "C_gold[ i * kSizeN + j ] = " << C_gold[ i * N_2 + j ] << std::endl;
#endif


// TPs
#if 1
  auto NUM_WG_2 = atf::tp( "NUM_WG_2", { 1 } ); //atf::interval<size_t>(0,M_exp, atf::pow_2), atf::divides( N ) );
  auto NUM_WG_1 = atf::tp( "NUM_WG_1", atf::interval<size_t>(0,M_exp, atf::pow_2), atf::divides( M ) );

  auto NUM_WI_2 = atf::tp( "NUM_WI_2", atf::interval<size_t>(0,M_exp, atf::pow_2), atf::divides( N / NUM_WG_2 ) );
  auto NUM_WI_1 = atf::tp( "NUM_WI_1", atf::interval<size_t>(0,N_exp, atf::pow_2), atf::divides( M / NUM_WG_1 ) );

  auto N_1 = atf::tp( "N_1", { M } );
  auto N_2 = atf::tp( "N_2", { N } );
#endif

// only for testing
#if 0
  auto NUM_WG_3 = atf::tp( "NUM_WG_3", {1} );
  auto NUM_WG_2 = atf::tp( "NUM_WG_2", {1} );
  auto NUM_WG_1 = atf::tp( "NUM_WG_1", {1} );

  auto NUM_WI_3 = atf::tp( "NUM_WI_3", {2} );
  auto NUM_WI_2 = atf::tp( "NUM_WI_2", {1} );
  auto NUM_WI_1 = atf::tp( "NUM_WI_1", {1} );

  auto LM_SIZE_3 = atf::tp( "LM_SIZE_3", {4} );
  auto LM_SIZE_2 = atf::tp( "LM_SIZE_2", {4} );
  auto LM_SIZE_1 = atf::tp( "LM_SIZE_1", {4} );

  auto PM_SIZE_3 = atf::tp( "PM_SIZE_3", {2} );
  auto PM_SIZE_2 = atf::tp( "PM_SIZE_2", {4} );
  auto PM_SIZE_1 = atf::tp( "PM_SIZE_1", {4} );
  
  auto GM_SIZE_3 = atf::tp( "GM_SIZE_3", { N_3 } );
  auto GM_SIZE_2 = atf::tp( "GM_SIZE_2", { N_2 } );
  auto GM_SIZE_1 = atf::tp( "GM_SIZE_1", { N_1 } );
#endif

  // OpenCL kernel wrapper
  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::GPU,
                                  0},
                                  { gemm_ocl_source , "func" },
                                  inputs( atf::buffer<float>( A ),
                                          atf::buffer<float>( x ),
                                          atf::buffer<float>( y )
                                        ),
                                  atf::cf::GS( NUM_WG_2 * NUM_WI_2, NUM_WG_1 * NUM_WI_1 ),
                                  atf::cf::LS( NUM_WI_2           , NUM_WI_1            )
                                ).check_result( A, x, y_gold );

  
  auto best_config =
atf::exhaustive//<NO_CONSTRAINTS>()
//atf::open_tuner//<NO_CONSTRAINTS>
//atf::open_tuner_flat<NO_CONSTRAINTS>
//atf::annealing_tree<NO_CONSTRAINTS>
//atf::annealing<NO_CONSTRAINTS>
()
//( atf::cond::evaluations(1) )
//( atf::cond::speedup(1, 200) || atf::cond::evaluations(1000) )
//( atf::cond::speedup(1, 200) )
(
  G( N_2, NUM_WG_2, NUM_WI_2 ),
  G( N_1, NUM_WG_1, NUM_WI_1 )
)
( ocl_kernel );

 
 std::cout << "\nbest found configuration: ";
 for( auto& tp : best_config )
   std::cout << tp.first << " = " << tp.second << std::endl;

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  return 0;
}
#endif



#ifdef MD_HOM_GEMM
void fill( std::vector<float>& mat )
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0, 10.0);

  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = i;//dist(mt);
}


void zero( std::vector<float>& mat )
{
  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = 0;
}


int main()
{
  auto start = std::chrono::system_clock::now();
  
  std::string gemm_ocl_source =
                                #include "md_hom_gemm.cl"
                              ;
  
  const size_t N = 2;
  const int N_3 = pow(2,N);
  const int N_2 = pow(2,N);
  const int N_1 = pow(2,N);
  
  std::vector<float> A( N_3 * N_1 ); fill( A );
  std::vector<float> B( N_1 * N_2 ); fill( B );
  std::vector<float> C( N_3 * N_2 ); zero( C );
  
  std::vector<float> C_gold( N_3 * N_2 );
  for( int i = 0 ; i < N_3 ; ++i )
    for( int j = 0 ; j < N_2 ; ++j )
      for( int k = 0 ; k < N_1 ; ++k )
        C_gold[ i * N_2 + j ] += A[ i * N_1 + k ] * B[ j * N_1 + k ];


#if 0 // only for testing
  for( int i = 0 ; i < N_3 ; ++i )
    for( int j = 0 ; j < N_2 ; ++j )
      std::cout << "C_gold[ i * kSizeN + j ] = " << C_gold[ i * N_2 + j ] << std::endl;
#endif


// TPs -- full search space
#if 1
  auto NUM_WG_3 = atf::tp( "NUM_WG_3", atf::interval<size_t>(0,N, atf::pow_2) );// , atf::divides( N_3 ) );
  auto NUM_WG_2 = atf::tp( "NUM_WG_2", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_2 ) );
  auto NUM_WG_1 = atf::tp( "NUM_WG_1", { 1 } );// atf::interval<size_t>(0,N, atf::pow_2) );

  auto NUM_WI_3 = atf::tp( "NUM_WI_3", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_3 / NUM_WG_3 ) );
  auto NUM_WI_2 = atf::tp( "NUM_WI_2", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_2 / NUM_WG_2 ) );
  auto NUM_WI_1 = atf::tp( "NUM_WI_1", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_1 / NUM_WG_1 ) ); // has to be a power of 2 (currently required for parallel reduction)

  auto LM_SIZE_3 = atf::tp( "LM_SIZE_3", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto LM_SIZE_3) { return N_3 % (NUM_WG_3 * LM_SIZE_3) == 0; } );
  auto LM_SIZE_2 = atf::tp( "LM_SIZE_2", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto LM_SIZE_2) { return N_2 % (NUM_WG_2 * LM_SIZE_2) == 0; } );
  auto LM_SIZE_1 = atf::tp( "LM_SIZE_1", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto LM_SIZE_1) { return N_1 % (NUM_WG_1 * LM_SIZE_1) == 0; } );

  auto PM_SIZE_3 = atf::tp( "PM_SIZE_3", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto PM_SIZE_3) { return LM_SIZE_3 % (NUM_WI_3 * PM_SIZE_3) == 0; } );
  auto PM_SIZE_2 = atf::tp( "PM_SIZE_2", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto PM_SIZE_2) { return LM_SIZE_2 % (NUM_WI_2 * PM_SIZE_2) == 0; } );
  auto PM_SIZE_1 = atf::tp( "PM_SIZE_1", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto PM_SIZE_1) { return LM_SIZE_1 % (NUM_WI_1 * PM_SIZE_1) == 0; } );
  
  auto GM_SIZE_3 = atf::tp( "GM_SIZE_3", { N_3 } );
  auto GM_SIZE_2 = atf::tp( "GM_SIZE_2", { N_2 } );
  auto GM_SIZE_1 = atf::tp( "GM_SIZE_1", { N_1 } );
#endif


// TPs -- partial search space
#if 0
  auto NUM_WG_3 = atf::tp( "NUM_WG_3", atf::interval<size_t>(0,N, atf::pow_2) );// , atf::divides( N_3 ) );
  auto NUM_WG_2 = atf::tp( "NUM_WG_2", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_2 ) );
  auto NUM_WG_1 = atf::tp( "NUM_WG_1", { 1 } );// atf::interval<size_t>(0,N, atf::pow_2) );

  auto NUM_WI_3 = atf::tp( "NUM_WI_3", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_3 / NUM_WG_3 ) );
  auto NUM_WI_2 = atf::tp( "NUM_WI_2", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_2 / NUM_WG_2 ) );
  auto NUM_WI_1 = atf::tp( "NUM_WI_1", atf::interval<size_t>(0,N, atf::pow_2) );//, atf::divides( N_1 / NUM_WG_1 ) ); // has to be a power of 2 (currently required for parallel reduction)

  auto LM_SIZE_3 = atf::tp( "LM_SIZE_3", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto LM_SIZE_3) { return N_3 % (NUM_WG_3 * LM_SIZE_3) == 0; } && atf::equal( NUM_WI_3 ) );
  auto LM_SIZE_2 = atf::tp( "LM_SIZE_2", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto LM_SIZE_2) { return N_2 % (NUM_WG_2 * LM_SIZE_2) == 0; } && atf::equal( NUM_WI_2 ) );
  auto LM_SIZE_1 = atf::tp( "LM_SIZE_1", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto LM_SIZE_1) { return N_1 % (NUM_WG_1 * LM_SIZE_1) == 0; } && atf::equal( NUM_WI_1 ) );

  auto PM_SIZE_3 = atf::tp( "PM_SIZE_3", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto PM_SIZE_3) { return LM_SIZE_3 % (NUM_WI_3 * PM_SIZE_3) == 0; } && atf::equal(1) );
  auto PM_SIZE_2 = atf::tp( "PM_SIZE_2", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto PM_SIZE_2) { return LM_SIZE_2 % (NUM_WI_2 * PM_SIZE_2) == 0; } );
  auto PM_SIZE_1 = atf::tp( "PM_SIZE_1", atf::interval<size_t>(0,N, atf::pow_2), [&]( auto PM_SIZE_1) { return LM_SIZE_1 % (NUM_WI_1 * PM_SIZE_1) == 0; } && atf::equal(1) );
  
  auto GM_SIZE_3 = atf::tp( "GM_SIZE_3", { N_3 } );
  auto GM_SIZE_2 = atf::tp( "GM_SIZE_2", { N_2 } );
  auto GM_SIZE_1 = atf::tp( "GM_SIZE_1", { N_1 } );
#endif


// only for testing
#if 0
  auto NUM_WG_3 = atf::tp( "NUM_WG_3", {1} );
  auto NUM_WG_2 = atf::tp( "NUM_WG_2", {1} );
  auto NUM_WG_1 = atf::tp( "NUM_WG_1", {1} );

  auto NUM_WI_3 = atf::tp( "NUM_WI_3", {2} );
  auto NUM_WI_2 = atf::tp( "NUM_WI_2", {1} );
  auto NUM_WI_1 = atf::tp( "NUM_WI_1", {1} );

  auto LM_SIZE_3 = atf::tp( "LM_SIZE_3", {4} );
  auto LM_SIZE_2 = atf::tp( "LM_SIZE_2", {4} );
  auto LM_SIZE_1 = atf::tp( "LM_SIZE_1", {4} );

  auto PM_SIZE_3 = atf::tp( "PM_SIZE_3", {2} );
  auto PM_SIZE_2 = atf::tp( "PM_SIZE_2", {4} );
  auto PM_SIZE_1 = atf::tp( "PM_SIZE_1", {4} );
  
  auto GM_SIZE_3 = atf::tp( "GM_SIZE_3", { N_3 } );
  auto GM_SIZE_2 = atf::tp( "GM_SIZE_2", { N_2 } );
  auto GM_SIZE_1 = atf::tp( "GM_SIZE_1", { N_1 } );
#endif

  // OpenCL kernel wrapper
  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::GPU,
                                  0},
                                  { gemm_ocl_source , "gemm_1" },
                                  inputs( atf::buffer<float>( A ),
                                          atf::buffer<float>( B ),
                                          atf::buffer<float>( C )
                                        ),
                                  atf::cf::GS( NUM_WG_1 * NUM_WI_1, NUM_WG_2 * NUM_WI_2, NUM_WG_3 * NUM_WI_3 ),
                                  atf::cf::LS( NUM_WI_1           , NUM_WI_2           , NUM_WI_3            )
                                ).check_result( A, B, C_gold );

  
  auto best_config =
atf::exhaustive()//<NO_CONSTRAINTS>()
//atf::open_tuner//<NO_CONSTRAINTS>
//atf::open_tuner_flat<NO_CONSTRAINTS>
//atf::annealing_tree<NO_CONSTRAINTS>
//atf::annealing<NO_CONSTRAINTS>
//()
//( atf::cond::evaluations(1000) )
//( atf::cond::speedup(1, 200) || atf::cond::evaluations(1000) )
//( atf::cond::speedup(1, 200) )

(
  G( NUM_WG_3, NUM_WI_3, LM_SIZE_3, PM_SIZE_3, GM_SIZE_3 ),
  G( NUM_WG_2, NUM_WI_2, LM_SIZE_2, PM_SIZE_2, GM_SIZE_2 ),
  G( NUM_WG_1, NUM_WI_1, LM_SIZE_1, PM_SIZE_1, GM_SIZE_1 )
)
( ocl_kernel );

 
 std::cout << "\nbest found configuration: ";
 for( auto& tp : best_config )
   std::cout << tp.first << " = " << tp.second << std::endl;

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  return 0;
}
#endif




#ifdef CLBLAST_SAXPY

int main()
{
  auto start = std::chrono::system_clock::now();

  std::string saxpy_ocl_source =
                                  #include "saxpy.cl" // TODO: USE_CL_MAD set?
                               ;

  // input size
  const int n = 400000;

  // alpha
  const float alpha = 1;

  // x (lhs)
  //auto      x        = std::vector<float>( n, 1 );
  const int x_offset = 0;
  const int x_inc    = 1;

  // y (rhs)
  //auto      y        = std::vector<float>( n, 2 );
  const int y_offset = 0;
  const int y_inc    = 1;

  // tuning parameters
#if 1
  auto WPT = atf::tp( "WPT", atf::interval<size_t>(1,n) );//, atf::divides( n       ) );
  auto WGS = atf::tp( "WGS", atf::interval<size_t>(1,n) );//, atf::divides( n / WPT ) );
  auto VW  = atf::tp( "VW" , {1}                    );
#else
  auto WGS = atf::tp( "WGS", {8} );
  auto WPT = atf::tp( "WPT", {6250}  );
  auto VW  = atf::tp( "VW" , {1}  );
#endif

  // cost function (TODO: CLBlast wrapper)
  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::CPU,
                                  0},
                                  saxpy_ocl_source,
                                  inputs( atf::scalar<int>( n ),

                                          atf::scalar<float>( alpha ),

                                          atf::buffer<float>( n ),
                                          atf::scalar<int>( x_offset ),
                                          atf::scalar<int>( x_inc ),

                                          atf::buffer<float>( n ),
                                          atf::scalar<int>( y_offset ),
                                          atf::scalar<int>( y_inc )
                                        ),
                                   atf::cf::GS( n / WPT ), atf::cf::LS( WGS )
                                );

  // tuning
  atf::exhaustive() //open_tuner_flat( atf::cond::valid_test_count(10) ).set_TPs( WGS,
                 (
                   WPT,
                   WGS,
                   VW
                 )
                 ( ocl_kernel );

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  return 0;
}
#endif





#ifdef CLBLAST_GEMM_DIRECT
size_t CeilDiv(const size_t x, const size_t y) {
  return 1 + ((x - 1) / y);
}
size_t Ceil(const size_t x, const size_t y) {
  return CeilDiv(x,y)*y;
}


void fill( std::vector<float>& mat )
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0, 10.0);

  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = 1;//dist(mt);
}


int main()
{
  auto start = std::chrono::system_clock::now();
  
  std::string gemm_ocl_source =
                                #include  "xgemm_direct.opencl" //"gemm.cl" //
                              ;
  
//  size_t N = 8;
  
  const int kSizeM = 20;//pow(2,N);
  const int kSizeN = 576;//pow(2,N);
  const int kSizeK = 25;//pow(2,N);
  
  std::vector<float> A( kSizeM * kSizeK ); fill( A );
  std::vector<float> B( kSizeK * kSizeN ); fill( B );
  std::vector<float> C( kSizeM * kSizeN );
  
  std::vector<float> C_gold( kSizeM * kSizeN );
  for( int i = 0 ; i < kSizeM ; ++i )
    for( int j = 0 ; j < kSizeN ; ++j )
      for( int k = 0 ; k < kSizeK ; ++k )
      {
        C_gold[ i * kSizeN + j ] += A[ i * kSizeK + k ] * B[ k * kSizeN + j ];
        //std::cout << "C_gold[ i * kSizeN + j ] = " << C_gold[ i * kSizeN + j ] << std::endl;
      }

#if 0 // kernel's default values
  auto WGD    = atf::tp( "WGD",    {8} );
  auto MDIMCD = atf::tp( "MDIMCD", {8}, atf::divides( WGD ) /*&& atf::divides(kSizeM)*/ );
  auto NDIMCD = atf::tp( "NDIMCD", {8}, atf::divides( WGD ) /*&& atf::divides(kSizeN)*/ );
  auto MDIMAD = atf::tp( "MDIMAD", {8}, atf::divides( WGD ) && [&](auto MDIMAD) { return (MDIMCD*NDIMCD) % MDIMAD == 0; } && [&](auto MDIMAD) { return WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0; } );
  auto NDIMBD = atf::tp( "NDIMBD", {8}, atf::divides( WGD ) && [&](auto NDIMBD) { return (MDIMCD*NDIMCD) % NDIMBD == 0; } && [&](auto NDIMBD) { return WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0; } );
  auto KWID   = atf::tp( "KWID",   {1}, atf::divides( WGD ) );
  auto VWMD   = atf::tp( "VWMD",   {1}, atf::divides(WGD / MDIMCD) && atf::divides(WGD / MDIMAD) );
  auto VWND   = atf::tp( "VWND",   {1}, atf::divides(WGD / NDIMCD) && atf::divides(WGD / NDIMBD) );
  auto PADA   = atf::tp( "PADA",   {1} );
  auto PADB   = atf::tp( "PADB",   {1} );
#endif


#if 0 // only for testing
  auto WGD    = atf::tp( "WGD",    {64} );
  auto MDIMCD = atf::tp( "MDIMCD", {16}, atf::divides( WGD ) /*&& atf::divides(kSizeM)*/ );
  auto NDIMCD = atf::tp( "NDIMCD", {16}, atf::divides( WGD ) /*&& atf::divides(kSizeN)*/ );
  auto MDIMAD = atf::tp( "MDIMAD", {16}, atf::divides( WGD ) && [&](auto MDIMAD) { return (MDIMCD*NDIMCD) % MDIMAD == 0; } && [&](auto MDIMAD) { return WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0; } );
  auto NDIMBD = atf::tp( "NDIMBD", {16}, atf::divides( WGD ) && [&](auto NDIMBD) { return (MDIMCD*NDIMCD) % NDIMBD == 0; } && [&](auto NDIMBD) { return WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0; } );
  auto KWID   = atf::tp( "KWID",   {2}, atf::divides( WGD ) );
  auto VWMD   = atf::tp( "VWMD",   {2}, atf::divides(WGD / MDIMCD) && atf::divides(WGD / MDIMAD) );
  auto VWND   = atf::tp( "VWND",   {1}, atf::divides(WGD / NDIMCD) && atf::divides(WGD / NDIMBD) );
  auto PADA   = atf::tp( "PADA",   {0} );
  auto PADB   = atf::tp( "PADB",   {0} );
#endif


#if 0 // OpenTuner
  auto WGD    = atf::tp( "WGD",    {8, 16, 32, 64, 128} );
  auto MDIMCD = atf::tp( "MDIMCD", {8, 16, 32}          );
  auto NDIMCD = atf::tp( "NDIMCD", {8, 16, 32}          );
  auto MDIMAD = atf::tp( "MDIMAD", {8, 16, 32}          );
  auto NDIMBD = atf::tp( "NDIMBD", {8, 16, 32}          );
  auto KWID   = atf::tp( "KWID",   {2, 8, 16}           );
  auto VWMD   = atf::tp( "VWMD",   {1, 2, 4, 8}         );
  auto VWND   = atf::tp( "VWND",   {1, 2, 4, 8}         );
  auto PADA   = atf::tp( "PADA",   {0, 1}               );
  auto PADB   = atf::tp( "PADB",   {0, 1}               );
#endif



#if 0 // CLTune
  auto WGD    = atf::tp( "WGD",    {8, 16, 32, 64, 128} );
  auto MDIMCD = atf::tp( "MDIMCD", {8, 16, 32},     atf::divides( WGD ) /*&& atf::divides(kSizeM)*/ );
  auto NDIMCD = atf::tp( "NDIMCD", {8, 16, 32},     atf::divides( WGD ) /*&& atf::divides(kSizeN)*/ );
  auto MDIMAD = atf::tp( "MDIMAD", {8, 16, 32},     atf::divides( WGD ) && [&](auto MDIMAD) { return (MDIMCD*NDIMCD) % MDIMAD == 0; } && [&](auto MDIMAD) { return WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0; } );
  auto NDIMBD = atf::tp( "NDIMBD", {8, 16, 32},     atf::divides( WGD ) && [&](auto NDIMBD) { return (MDIMCD*NDIMCD) % NDIMBD == 0; } && [&](auto NDIMBD) { return WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0; } );
  auto KWID   = atf::tp( "KWID",   {2, 8, 16},      atf::divides( WGD ) );
  auto VWMD   = atf::tp( "VWMD",   {1, 2, 4, 8},    atf::divides(WGD / MDIMCD) && atf::divides(WGD / MDIMAD) );
  auto VWND   = atf::tp( "VWND",   {1, 2, 4, 8},    atf::divides(WGD / NDIMCD) && atf::divides(WGD / NDIMBD) );
  auto PADA   = atf::tp( "PADA",   {0, 1}  ); 
  auto PADB   = atf::tp( "PADB",   {0, 1}  );
#endif

 
#if 1 // -> ATF
  const int kSizeMax = std::max( kSizeM, kSizeN );

  auto WGD    = atf::tp( "WGD",    atf::interval<int>(1,kSizeMax) );
  auto MDIMCD = atf::tp( "MDIMCD", atf::interval<int>(1,kSizeM),   atf::divides( WGD ) && atf::divides(kSizeM) );
  auto NDIMCD = atf::tp( "NDIMCD", atf::interval<int>(1,kSizeN),   atf::divides( WGD ) && atf::divides(kSizeN) );
  auto MDIMAD = atf::tp( "MDIMAD", atf::interval<int>(1,kSizeM),   atf::divides( WGD ) && [&](auto MDIMAD) { return (MDIMCD*NDIMCD) % MDIMAD == 0; } && [&](auto MDIMAD) { return WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0; } );
  auto NDIMBD = atf::tp( "NDIMBD", atf::interval<int>(1,kSizeN),   atf::divides( WGD ) && [&](auto NDIMBD) { return (MDIMCD*NDIMCD) % NDIMBD == 0; } && [&](auto NDIMBD) { return WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0; } );
  auto KWID   = atf::tp( "KWID",   atf::interval<int>(1,kSizeK),   atf::divides( WGD ) );
  auto VWMD   = atf::tp( "VWMD",   {1, 2, 4, 8},                   atf::divides(WGD / MDIMCD) && atf::divides(WGD / MDIMAD) );
  auto VWND   = atf::tp( "VWND",   {1, 2, 4, 8},                   atf::divides(WGD / NDIMCD) && atf::divides(WGD / NDIMBD) && [&](auto){ return (((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD)+(((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD) <= 1024; } );
  auto PADA   = atf::tp( "PADA",   {0, 1}  ); 
  auto PADB   = atf::tp( "PADB",   {0, 1}  );
#endif
  
  auto PRECISION = atf::tp( "PRECISION", {32} );

  atf::cf::thread_configurations_t thread_configurations;

#if 0 // XgemmDirectNN
  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::GPU,
                                  0},
                                  { gemm_ocl_source , "XgemmDirectNN" },
                                  inputs( atf::scalar<int>(kSizeM),
                                          atf::scalar<int>(kSizeN),
                                          atf::scalar<int>(kSizeK),
                                          atf::scalar<float>(1),                // alpha
                                          atf::scalar<float>(0), //1),                // beta
                                          atf::buffer<float>( A ),
                                          atf::scalar<int>(0),                  // offset M
                                          atf::scalar<int>(kSizeM),                  // a_ld
                                          atf::buffer<float>( B ),
                                          atf::scalar<int>(0),                  // offset N
                                          atf::scalar<int>(kSizeN),                 // b_ld
                                          atf::buffer<float>( C ),
                                          atf::scalar<int>(0),                  // offset K
                                          atf::scalar<int>(kSizeN),                 // c_ld
                                          atf::scalar<int>(1),                  // c_transpose
                                          atf::scalar<int>(0),                  // a_conjugate
                                          atf::scalar<int>(0)                   // b_conjugate
                                        ),
                                  atf::cf::GS( ((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD, ((1 + ((kSizeN - 1) / WGD))*WGD * NDIMCD) / WGD ),
                                  atf::cf::LS(  MDIMCD                ,  NDIMCD                 )
                                ).check_result( A, B, C_gold );
#endif

#if 1 // XgemmDirectTN
  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::GPU,
                                  0},
                                  { gemm_ocl_source , "XgemmDirectTN" },
                                  inputs( atf::scalar<int>(kSizeM),
                                          atf::scalar<int>(kSizeN),
                                          atf::scalar<int>(kSizeK),
                                          atf::scalar<float>(1),                // alpha
                                          atf::scalar<float>( 0 ), //1),                // beta
                                          atf::buffer<float>( A ),
                                          atf::scalar<int>(0),                  // offset M
                                          atf::scalar<int>(kSizeK),                  // a_ld
                                          atf::buffer<float>( B ),
                                          atf::scalar<int>(0),                  // offset N
                                          atf::scalar<int>(kSizeN),                 // b_ld
                                          atf::buffer<float>( C ),
                                          atf::scalar<int>(0),                  // offset K
                                          atf::scalar<int>(kSizeN),                 // c_ld
                                          atf::scalar<int>(1),                  // c_transpose
                                          atf::scalar<int>(0),                  // a_conjugate
                                          atf::scalar<int>(0)                   // b_conjugate
                                        ),
                                  atf::cf::GS( ((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD, ((1 + ((kSizeN - 1) / WGD))*WGD * NDIMCD) / WGD ),
                                  atf::cf::LS(  MDIMCD                ,  NDIMCD                 )
                                ).check_result( A, B, C_gold );
#endif


#if 0 // XgemmDirectTT
  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::GPU,
                                  0},
                                  { gemm_ocl_source , "XgemmDirectTT" },
                                  inputs( atf::scalar<int>(kSizeM),
                                          atf::scalar<int>(kSizeN),
                                          atf::scalar<int>(kSizeK),
                                          atf::scalar<float>(1),                // alpha
                                          atf::scalar<float>(0),                // beta
                                          atf::buffer<float>( A ), //kSizeM * kSizeK),
                                          atf::scalar<int>(0),                  // offset M
                                          atf::scalar<int>(kSizeK),                  // a_ld
                                          atf::buffer<float>( B ),
                                          atf::scalar<int>(0),                  // offset N
                                          atf::scalar<int>(kSizeK),                 // b_ld
                                          atf::buffer<float>( C ),
                                          atf::scalar<int>(0),                  // offset K
                                          atf::scalar<int>(kSizeN),                 // c_ld
                                          atf::scalar<int>(1),                  // c_transpose
                                          atf::scalar<int>(0),                  // a_conjugate
                                          atf::scalar<int>(0)                   // b_conjugate
                                        ),
                                  atf::cf::GS( ((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD, ( (1 + ((kSizeN - 1) / WGD))*WGD * NDIMCD) / WGD ),
                                  atf::cf::LS(  MDIMCD                ,  NDIMCD                 )
                                ).check_result( A, B, C_gold );
#endif

  auto best_config =
//atf::exhaustive//<NO_CONSTRAINTS>
//atf::open_tuner//<NO_CONSTRAINTS>
//atf::open_tuner_flat<NO_CONSTRAINTS>
atf::annealing_tree//<NO_CONSTRAINTS>
//atf::annealing<NO_CONSTRAINTS>
( atf::cond::evaluations(10) )
//( atf::cond::speedup(1, 200) || atf::cond::evaluations(1000) )
//( atf::cond::speedup(1, 200) )

#if 1 // with G(...)
  ( G(WGD,
      MDIMCD,
      NDIMCD,
      MDIMAD,
      NDIMBD,
      KWID,
      VWMD,
      VWND
     ),
    G( PADA ),
    G( PADB ),
    G( PRECISION )
  )
  ( ocl_kernel );
#else // without G(...)
  ( WGD,
    MDIMCD,
    NDIMCD,
    MDIMAD,
    NDIMBD,
    KWID,
    VWMD,
    VWND,
    PADA,
    PADB,
    PRECISION
  )
  ( ocl_kernel );
#endif
 
 std::cout << "\nbest found configuration: ";
 for( auto& tp : best_config )
   std::cout << tp.first << " = " << tp.second << std::endl;

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  return 0;
}
#endif



#ifdef CLBLAST_GEMV
size_t CeilDiv(const size_t x, const size_t y) {
  return 1 + ((x - 1) / y);
}
size_t Ceil(const size_t x, const size_t y) {
  return CeilDiv(x,y)*y;
}


void fill( std::vector<float>& mat )
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0, 10.0);

  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = 1;//dist(mt);
}


int main()
{
  auto start = std::chrono::system_clock::now();
  
  std::string gemv_ocl_source =
                                #include  "xgemv.opencl" //"gemm.cl" //
                              ;
  
  //size_t N = 9;
  const int kSizeM = 20;// pow(2,N);
  const int kSizeN = 576;//pow(2,N);
  
  std::vector<float> A( kSizeM * kSizeN ); fill( A );
  std::vector<float> x(          kSizeN ); fill( x );
  std::vector<float> y( kSizeM          );
  
  std::vector<float> y_gold( kSizeM  );
  for( int i = 0 ; i < kSizeM ; ++i )
    for( int j = 0 ; j < kSizeN ; ++j )
      {
        y_gold[ i  ] += A[ i * kSizeN + j ] * x[ j ];
        //std::cout << "C_gold[ i * kSizeN + j ] = " << C_gold[ i * kSizeN + j ] << std::endl;
      }

 
#if 1 // -> ATF
  const int kSizeMax = std::max( kSizeM, kSizeN );
  
  auto WGS = atf::tp( "WGS", atf::interval<int>(1,kSizeMax)     );
  auto WPT = atf::tp( "WPT", atf::interval<int>(1,kSizeMax), atf::divides(kSizeM) && atf::less_than_or_eq(WGS) && [&](auto WPT){ return kSizeM/WPT % WGS == 0; } );
  auto VW  = atf::tp( "VW" , {1, 2, 4, 8}, atf::divides(WPT) );
#endif
  


  auto ocl_kernel = atf::cf::ocl( {"Apple",
                                  atf::cf::device_info::GPU,
                                  0},
                                  { gemv_ocl_source , "Xgemv" },
                                  inputs( atf::scalar<int>(kSizeM),
                                          atf::scalar<int>(kSizeN),
                                          atf::scalar<float>(1),                // alpha
                                          atf::scalar<float>(0),                // beta
                                          atf::scalar<int>(1),                // rotated
                                          atf::buffer<float>( A ),            // A
                                          atf::scalar<int>(0),                  // A offset
                                          atf::scalar<int>(576),                  // a_ld
                                          atf::buffer<float>( x ),               // x
                                          atf::scalar<int>(0),                  // x offset
                                          atf::scalar<int>(1),                 // x inc
                                          atf::buffer<float>( y ),   // y
                                          atf::scalar<int>(0),                  // y offset
                                          atf::scalar<int>(1),                 //  y inc
                                          atf::scalar<int>(0),                  // do_conjugate
                                          atf::scalar<int>(0),                  // parameter
                                          atf::scalar<int>(0),                  // kl
                                          atf::scalar<int>(0)                   // ku
                                        ),
                                  atf::cf::GS( kSizeM / WPT ),
                                  atf::cf::LS( WGS    )
                                ).check_result( A, x, y_gold );


  auto best_config =
atf::open_tuner
  ( atf::cond::speedup(1, 200) || atf::cond::evaluations(1000) )
  ( G(WGS, WPT, VW)                                            )
  ( ocl_kernel                                                 );

 
 std::cout << "\nbest found configuration: ";
 for( auto& tp : best_config )
   std::cout << tp.first << " = " << tp.second << std::endl;

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  return 0;
}
#endif






#ifdef CLTUNE_MATMULT
void mat_mult( std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, size_t N )
{
  for( size_t i = 0 ; i < N ; ++i )
    for( size_t j = 0 ; j < N ; ++j )
      for( size_t k = 0 ; k < N ; ++k )
        C[ i * N + j ] = A[ i * N + k ] * B[ k * N + j ];
}


void fill( std::vector<float>& mat )
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0, 10.0);
  
  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = dist(mt);
}


int main()
{
  auto start = std::chrono::system_clock::now();
  
  std::string gemm_ocl_source =
                                #include "gemm.cl"
                              ;

  const int DIM_M = 7; // 100 -> 128
  const int DIM_N = 5; // 10  -> 16
  const int DIM_K = 9; // 500 -> 512
  
  const int kSizeM = pow(2,DIM_M);
  const int kSizeN = pow(2,DIM_N);
  const int kSizeK = pow(2,DIM_K);
  
  std::vector<float> C( kSizeM * kSizeN );
  std::vector<float> A( kSizeM * kSizeK );
  std::vector<float> B( kSizeK * kSizeN );
  
  fill( A );
  fill( B );


#if 0 // testing
  auto MWG   = atf::tp( "MWG"  , {16}  );
  auto NWG   = atf::tp( "NWG"  , {16}  );
  auto KWG   = atf::tp( "KWG"  , {16}                                  );
  auto MDIMC = atf::tp( "MDIMC", {8},       atf::divides(MWG)  );
  auto NDIMC = atf::tp( "NDIMC", {8},       atf::divides(NWG)  );
  auto MDIMA = atf::tp( "MDIMA", {8},       atf::divides(MWG) && [&](auto MDIMA) { return (MDIMC*NDIMC) % MDIMA == 0; } && [&](auto MDIMA) { return KWG % ( (MDIMC*NDIMC) / MDIMA ) == 0; }  );
  auto NDIMB = atf::tp( "NDIMB", {8},       atf::divides(NWG) && [&](auto NDIMB) { return (MDIMC*NDIMC) % NDIMB == 0; } && [&](auto NDIMB) { return KWG % ( (MDIMC*NDIMC) / NDIMB ) == 0; }  );
  auto KWI   = atf::tp( "KWI"  , {2},            atf::divides( KWG ) );
  auto VWM   = atf::tp( "VWM"  , {1},         atf::divides(MWG / MDIMC) && atf::divides(MWG / MDIMA) );
  auto VWN   = atf::tp( "VWN"  , {1},         atf::divides(NWG / NDIMC) && atf::divides(NWG / NDIMB) && [&](auto){ return ( (kSizeM * MDIMC) / MWG ) + ( (kSizeN * NDIMC) / NWG ) <= 1024; } );
  auto STRM  = atf::tp( "STRM" , {0}             );
  auto STRN  = atf::tp( "STRN" , {0}             );
  auto SA    = atf::tp( "SA"   , {0}             );
  auto SB    = atf::tp( "SB"   , {0}             );
#endif

#if 0 // CLTune
  auto MWG   = atf::tp( "MWG"  , {16, 32, 64, 128}  );
  auto NWG   = atf::tp( "NWG"  , {16, 32, 64, 128}  );
  auto KWG   = atf::tp( "KWG"  , {16, 32}                                  );
  auto MDIMC = atf::tp( "MDIMC", {8, 16, 32},       atf::divides(MWG)  );
  auto NDIMC = atf::tp( "NDIMC", {8, 16, 32},       atf::divides(NWG)  );
  auto MDIMA = atf::tp( "MDIMA", {8, 16, 32},       atf::divides(MWG) && [&](auto MDIMA) { return (MDIMC*NDIMC) % MDIMA == 0; } && [&](auto MDIMA) { return KWG % ( (MDIMC*NDIMC) / MDIMA ) == 0; }  );
  auto NDIMB = atf::tp( "NDIMB", {8, 16, 32},       atf::divides(NWG) && [&](auto NDIMB) { return (MDIMC*NDIMC) % NDIMB == 0; } && [&](auto NDIMB) { return KWG % ( (MDIMC*NDIMC) / NDIMB ) == 0; }  );
  auto KWI   = atf::tp( "KWI"  , {2, 8},            atf::divides( KWG ) );
  auto VWM   = atf::tp( "VWM"  , {1,2,4,8},         atf::divides(MWG / MDIMC) && atf::divides(MWG / MDIMA) );
  auto VWN   = atf::tp( "VWN"  , {1,2,4,8},         atf::divides(NWG / NDIMC) && atf::divides(NWG / NDIMB) && [&](auto){ return ( (kSizeM * MDIMC) / MWG ) + ( (kSizeN * NDIMC) / NWG ) <= 1024; } );
  auto STRM  = atf::tp( "STRM" , {0, 1}             );
  auto STRN  = atf::tp( "STRN" , {0, 1}             );
  auto SA    = atf::tp( "SA"   , {0, 1}             );
  auto SB    = atf::tp( "SB"   , {0, 1}             );
#endif



#if 0 // OT
  auto MWG   = atf::tp( "MWG"  , atf::interval<int>(0,DIM_M, atf::pow_2) );
  auto NWG   = atf::tp( "NWG"  , atf::interval<int>(0,DIM_N, atf::pow_2) );
  auto KWG   = atf::tp( "KWG"  , atf::interval<int>(0,DIM_K, atf::pow_2) );
  auto MDIMC = atf::tp( "MDIMC", atf::interval<int>(0,DIM_M, atf::pow_2) );
  auto NDIMC = atf::tp( "NDIMC", atf::interval<int>(0,DIM_N, atf::pow_2) );
  auto MDIMA = atf::tp( "MDIMA", atf::interval<int>(0,DIM_M, atf::pow_2) );
  auto NDIMB = atf::tp( "NDIMB", atf::interval<int>(0,DIM_N, atf::pow_2) );
  auto KWI   = atf::tp( "KWI"  , atf::interval<int>(0,DIM_K, atf::pow_2) );
  auto VWM   = atf::tp( "VWM"  , {1,2,4,8}                               );
  auto VWN   = atf::tp( "VWN"  , {1,2,4,8}                               );
  auto STRM  = atf::tp( "STRM" , {0, 1}                                  );
  auto STRN  = atf::tp( "STRN" , {0, 1}                                  );
  auto SA    = atf::tp( "SA"   , {0, 1}                                  );
  auto SB    = atf::tp( "SB"   , {0, 1}                                  );
#endif
 
 
#if 1 // -> ATF
  auto MWG   = atf::tp( "MWG"  , atf::interval<int>(0,DIM_M, atf::pow_2)  );
  auto NWG   = atf::tp( "NWG"  , atf::interval<int>(0,DIM_N, atf::pow_2)  );
  auto KWG   = atf::tp( "KWG"  , atf::interval<int>(0,DIM_K, atf::pow_2)  );
  auto MDIMC = atf::tp( "MDIMC", atf::interval<int>(0,DIM_M, atf::pow_2), atf::divides(MWG)  );
  auto NDIMC = atf::tp( "NDIMC", atf::interval<int>(0,DIM_N, atf::pow_2), atf::divides(NWG)  );
  auto MDIMA = atf::tp( "MDIMA", atf::interval<int>(0,DIM_M, atf::pow_2), atf::divides(MWG) && [&](auto MDIMA) { return (MDIMC*NDIMC) % MDIMA == 0; } && [&](auto MDIMA) { return KWG % ( (MDIMC*NDIMC) / MDIMA ) == 0; }  );
  auto NDIMB = atf::tp( "NDIMB", atf::interval<int>(0,DIM_N, atf::pow_2), atf::divides(NWG) && [&](auto NDIMB) { return (MDIMC*NDIMC) % NDIMB == 0; } && [&](auto NDIMB) { return KWG % ( (MDIMC*NDIMC) / NDIMB ) == 0; }  );
  auto KWI   = atf::tp( "KWI"  , atf::interval<int>(0,DIM_K, atf::pow_2), atf::divides( KWG ) );
  auto VWM   = atf::tp( "VWM"  , {1,2,4,8},         atf::divides(MWG / MDIMC) && atf::divides(MWG / MDIMA) );
  auto VWN   = atf::tp( "VWN"  , {1,2,4,8},         atf::divides(NWG / NDIMC) && atf::divides(NWG / NDIMB) /*&& [&](auto){ return ( (kSizeM * MDIMC) / MWG ) + ( (kSizeN * NDIMC) / NWG ) <= 1024; } <- Falsch: local size ist beschränkt und "+" durch "*" ersetzen */ );
  auto STRM  = atf::tp( "STRM" , {0, 1}             );
  auto STRN  = atf::tp( "STRN" , {0, 1}             );
  auto SA    = atf::tp( "SA"   , {0, 1}             );
  auto SB    = atf::tp( "SB"   , {0, 1}             );
#endif
  
  auto PRECISION = atf::tp( "PRECISION", {32} );

  atf::cf::thread_configurations_t thread_configurations;
  
  auto ocl_kernel = atf::cf::ocl( {"NVIDIA",
                                          atf::cf::device_info::GPU,
                                          0},
                                          gemm_ocl_source,
                                          inputs( atf::scalar<int>(kSizeM),
                                                  atf::scalar<int>(kSizeN),
                                                  atf::scalar<int>(kSizeK),
                                                  atf::buffer<float>(kSizeM * kSizeK),
                                                  atf::buffer<float>(kSizeK * kSizeN),
                                                  atf::buffer<float>(kSizeM * kSizeN)
                                                ),
                                          atf::cf::GS( (kSizeM * MDIMC) / MWG, (kSizeN * NDIMC) / NWG ),
                                          atf::cf::LS(  MDIMC                ,  NDIMC                 )
                                        ).save_thread_configuration( thread_configurations );


  auto best_config =  
//atf::exhaustive()
atf::open_tuner
//( atf::cond::valid_evaluations(117) )
( atf::cond::speedup(1, 100) || atf::cond::evaluations(1000) )
                                    ( G(MWG,
                                        NWG,
                                        KWG,
                                        MDIMC,
                                        NDIMC,
                                        MDIMA,
                                        NDIMB,
                                        KWI,
                                        VWM,
                                        VWN
                                       ),
                                      G( STRM ),
                                      G( STRN ),
                                      G( SA ),
                                      G( SB ),
                                      G( PRECISION )
                                    )
                                    ( ocl_kernel );
  
 for( auto& tp : best_config )
   std::cout << tp.first << " = " << tp.second << std::endl;

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  auto dummy_config = atf::configuration{};
  std::cout << "saxpy runtime with default parameter values: " << ocl_kernel( dummy_config ) << std::endl;

//  auto thread_configuration = thread_configurations[ best_config ];
//  for( size_t i = 0 ; i < 2 ; ++i )
//  {
//    for( size_t j = 0 ; j < 3 ; ++j )
//      std::cout << thread_configuration[ i ][ j ] << " ";
//    std::cout << std::endl;
//  }

  return 0;
}
#endif



#ifdef CLTUNE_CONV
// Helper function to perform an integer division + ceiling (round-up)
size_t CeilDiv(size_t a, size_t b)
{
  return (a + b - 1)/b;
}

// Helper function to determine whether or not 'a' is a multiple of 'b'
bool IsMultiple(size_t a, size_t b)
{
  return ((a/b)*b == a) ? true : false;
};

#define HFS (3)         // Half filter size
#define FS  (HFS+HFS+1) // Filter size

int main()
{
  auto start = std::chrono::system_clock::now();
  
  std::string conv_ocl_source =
                                #include "conv.cl"
                              ;

  const auto kExtraSize = size_t{FS*8};
//  const int N = 1024;
  
  const int kSizeX = 8192;
  const int kSizeY = 4096;
  
  auto mat_a = std::vector<float>( (kSizeX+kExtraSize) * (kSizeY+kExtraSize), 1 );
  auto mat_b = std::vector<float>(  kSizeX             *  kSizeY            , 0 );
  auto coeff = std::vector<float>(  FS                 *  FS                    );

  // Creates the filter coefficients (gaussian blur)
  auto sigma = 1.0f;
  auto mean = FS/2.0f;
  auto sum = 0.0f;
  for (auto x=size_t{0}; x<FS; ++x) {
    for (auto y=size_t{0}; y<FS; ++y) {
      auto exponent = -0.5f * (pow((x-mean)/sigma, 2.0f) + pow((y-mean)/sigma, 2.0f));
      coeff[y*FS + x] = static_cast<float>(exp(exponent) / (2.0f * 3.14159265f * sigma * sigma));
      sum += coeff[y*FS + x];
    }
  }
  for (auto &item: coeff) { item = item / sum; }

#if 1
  auto TBX   = atf::tp( "TBX"  , {8, 16, 32, 64} );
  auto TBY   = atf::tp( "TBY"  , {8, 16, 32, 64} );
  auto LOCAL = atf::tp( "LOCAL", {0, 1, 2}       );
  auto WPTX  = atf::tp( "WPTX" , {1, 2, 4, 8}    );
  auto WPTY  = atf::tp( "WPTY" , {1, 2, 4, 8}    );

  auto VECTOR_constraint = [&]( auto VECTOR )
                           {
                             if( LOCAL == 2 )
                               return IsMultiple( WPTX, VECTOR ) && IsMultiple( 2*HFS , VECTOR );
                             else
                               return IsMultiple( WPTX, VECTOR );
                            };
  auto VECTOR        = atf::tp( "VECTOR"       , {1, 2, 4}, VECTOR_constraint );
  auto UNROLL_FACTOR = atf::tp( "UNROLL_FACTOR", {1, FS}                      );

  auto PADDING_constraint = [&]( auto PADDING )
                            {
                              return ( PADDING == 0 || LOCAL != 0 );
                            };
  
// auto LocalMemorySize = [] (std::vector<size_t> v) {
//    if (LOCAL != 0) { return ((TBY*WPTY + 2*HFS) * (TBX*WPTX + 2*HFS + self))*sizeof(float); }
//    else           { return size_t{0}; }
//  };
//  tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"LOCAL", "TBX", "WPTX", "TBY", "WPTY", "PADDING"});
  
  
  
  auto PADDING = atf::tp( "PADDING", {0, 1}, PADDING_constraint );
  
  auto integers = std::initializer_list<size_t>{
    8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,
    32,33,34,35,36,37,38,39,40,41,42,
    64,65,66,67,68,69,70,71,72,73,74
  };
 
  auto TBX_XL_constraint = [&]( auto TBX_XL ) -> bool
                           {
                             if( LOCAL == 2 )
                               return ( TBX_XL == TBX + CeilDiv( 2*HFS, WPTX ) );
                             else
                               return ( TBX_XL == TBX );
                           };
  auto TBX_XL = atf::tp( "TBX_XL", integers, TBX_XL_constraint );
  
  auto TBY_XL_constraint = [&]( auto TBY_XL ) -> bool
                           {
                             if( LOCAL == 2 )
                               return ( TBY_XL == TBY + CeilDiv( 2*HFS, WPTY ) );
                             else
                               return ( TBY_XL == TBY );
                           };
  auto TBY_XL = atf::tp( "TBY_XL", integers, TBY_XL_constraint );


  auto GS_0 = atf::tp( "GS_0", atf::interval<size_t>(1,1, [&]( auto ) { return ( (kSizeX * TBX_XL) / TBX ) / WPTX; } ) );
  auto GS_1 = atf::tp( "GS_1", atf::interval<size_t>(1,1, [&]( auto ) { return ( (kSizeY * TBY_XL) / TBY ) / WPTY; } ) );
  
  auto LS_0 = atf::tp( "LS_0", atf::interval<size_t>(1,1, [&]( auto ) { return TBX_XL; } ) );
  auto LS_1 = atf::tp( "LS_1", atf::interval<size_t>(1,1, [&]( auto ) { return TBY_XL; } ), [&](auto i){ return LS_0*i<=1024;} );

#endif


#if 0
  auto TBX   = atf::tp( "TBX"  , atf::interval(3, 12, atf::pow_2) );
  auto TBY   = atf::tp( "TBY"  , atf::interval(3, 13, atf::pow_2) );
  auto LOCAL = atf::tp( "LOCAL", {0, 1, 2}       );
  auto WPTX  = atf::tp( "WPTX" , {1, 2, 4, 8}    );
  auto WPTY  = atf::tp( "WPTY" , {1, 2, 4, 8}    );

  auto VECTOR_constraint = [&]( auto VECTOR )
                           {
                             if( LOCAL == 2 )
                               return IsMultiple( WPTX, VECTOR ) && IsMultiple( 2*HFS , VECTOR );
                             else
                               return IsMultiple( WPTX, VECTOR );
                            };
  auto VECTOR        = atf::tp( "VECTOR"       , {1, 2, 4}, VECTOR_constraint );
  auto UNROLL_FACTOR = atf::tp( "UNROLL_FACTOR", {1, FS}                      );

  auto PADDING_constraint = [&]( auto PADDING )
                            {
                              return ( PADDING == 0 || LOCAL != 0 );
                            };
  
  auto PADDING = atf::tp( "PADDING", {0, 1}, PADDING_constraint );
  
  auto integers = std::initializer_list<size_t>{
    8,9,10,11,12,13,14,15,                16, 17, 18,
    16,17,18,19,20,21,22,23,24,25,26,
    32,33,34,35,36,37,38,39,40,41,42,
    64,65,66,67,68,69,70,71,72,73,74,

    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266,
    512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522,
    1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034,
    2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058
  };
  
  auto TBX_XL_constraint = [&]( auto TBX_XL ) -> bool
                           {
                             if( LOCAL == 2 )
                               return ( TBX_XL == TBX + CeilDiv( 2*HFS, WPTX ) && ( (kSizeX * TBX_XL) >= TBX) && ( ((kSizeX * TBX_XL) / TBX ) % WPTX == 0  ) );
                             else
                               return ( (TBX_XL == TBX) && ( (kSizeX * TBX_XL) >= TBX) && ( ((kSizeX * TBX_XL) / TBX ) % WPTX == 0  ) );
                           };
  auto TBX_XL = atf::tp( "TBX_XL", integers, TBX_XL_constraint );
  
  auto TBY_XL_constraint = [&]( auto TBY_XL ) -> bool
                           {
                             if( LOCAL == 2 )
                               return ( TBY_XL == TBY + CeilDiv( 2*HFS, WPTY ) && ( (kSizeY * TBY_XL) >= TBY) && ( ((kSizeY * TBY_XL) / TBY ) % WPTY == 0  ) );
                             else
                               return ( (TBY_XL == TBY) && ( (kSizeY * TBY_XL) >= TBY) && ( ((kSizeY * TBY_XL) / TBY ) % WPTY == 0  ) );
                           };
  auto TBY_XL = atf::tp( "TBY_XL", integers, TBY_XL_constraint );


  auto GS_0 = atf::tp( "GS_0", atf::interval<size_t>(1,1, [&]( auto ) { return ( (kSizeX * TBX_XL) / TBX ) / WPTX; } ) );
  auto GS_1 = atf::tp( "GS_1", atf::interval<size_t>(1,1, [&]( auto ) { return ( (kSizeY * TBY_XL) / TBY ) / WPTY; } ) );
  
  auto LS_0 = atf::tp( "LS_0", atf::interval<size_t>(1,1, [&]( auto ) { return TBX_XL; } ) );
  auto LS_1 = atf::tp( "LS_1", atf::interval<size_t>(1,1, [&]( auto ) { return TBY_XL; } ), [&](auto i){ return LS_0*i<=1024;} );

#endif

  auto ocl_kernel = atf::cf::ocl( "Apple",
                                  CL_DEVICE_TYPE_GPU,
                                  0,
                                  conv_ocl_source,
                                  inputs( atf::scalar<int>( kSizeX ),
                                          atf::scalar<int>( kSizeY ),
                                          atf::buffer<float>( mat_a ),
                                          atf::buffer<float>( coeff ),
                                          atf::buffer<float>( mat_b )
                                        )
                                );



  atf::open_tuner( atf::cond::valid_test_count(10) ).set_TPs(
                                                      TBX,
                                                      TBY,
                                                      LOCAL,
                                                      WPTX,
                                                      WPTY,
                                                      VECTOR,
                                                      UNROLL_FACTOR, // in own group G(...)
                                                      PADDING,
                                                      TBX_XL,
                                                      TBY_XL,
                                                      GS_0,
                                                      GS_1,
                                                      LS_0,
                                                      LS_1
                                                     ) ( ocl_kernel );

  auto end = std::chrono::system_clock::now();
  auto runtime_in_sec = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
  std::cout << std::endl << "total runtime for tuning and search space generation = " << runtime_in_sec << "sec\n" << std::endl;

  return 0;
}
#endif


#ifdef SEARCH_SPACE_GENERATION_TEST_USING_GEMM
size_t CeilDiv(const size_t x, const size_t y) {
  return 1 + ((x - 1) / y);
}
size_t Ceil(const size_t x, const size_t y) {
  return CeilDiv(x,y)*y;
}


void fill( std::vector<float>& mat )
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(1.0, 10.0);

  for( size_t i = 0 ; i < mat.size() ; ++i )
    mat[ i ] = dist(mt);
}


int main( int argc, char* argv[] )
{
  auto start = std::chrono::system_clock::now();

  
  size_t N = 8;//static_cast<size_t>(std::stoi(std::string{argv[1]}));
  const int kSizeM = pow(2,N);
  const int kSizeN = pow(2,N);
  const int kSizeK = pow(2,N);
  
  std::vector<float> A( kSizeM * kSizeK ); fill( A );
  std::vector<float> B( kSizeK * kSizeN ); fill( B );
  std::vector<float> C( kSizeM * kSizeN ); fill( C );
  
//#if 0 // CLTune
//  auto WGD    = atf::tp( "WGD",    {8, 16, 32, 64, 128} );
//  auto MDIMCD = atf::tp( "MDIMCD", {8, 16, 32},     atf::divides( WGD ) /*&& atf::divides(kSizeM)*/ );
//  auto NDIMCD = atf::tp( "NDIMCD", {8, 16, 32},     atf::divides( WGD ) /*&& atf::divides(kSizeN)*/ );
//  auto MDIMAD = atf::tp( "MDIMAD", {8, 16, 32},     atf::divides( WGD ) && [&](auto MDIMAD) { return (MDIMCD*NDIMCD) % MDIMAD == 0; } && [&](auto MDIMAD) { return WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0; } );
//  auto NDIMBD = atf::tp( "NDIMBD", {8, 16, 32},     atf::divides( WGD ) && [&](auto NDIMBD) { return (MDIMCD*NDIMCD) % NDIMBD == 0; } && [&](auto NDIMBD) { return WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0; } );
//  auto KWID   = atf::tp( "KWID",   {2, 8, 16},      atf::divides( WGD ) );
//  auto VWMD   = atf::tp( "VWMD",   {1, 2, 4, 8},    atf::divides(WGD / MDIMCD) && atf::divides(WGD / MDIMAD) );
//  auto VWND   = atf::tp( "VWND",   {1, 2, 4, 8},    atf::divides(WGD / NDIMCD) && atf::divides(WGD / NDIMBD) );
//  auto PADA   = atf::tp( "PADA",   {0, 1}  ); 
//  auto PADB   = atf::tp( "PADB",   {0, 1}  );
//#endif

 
#if 1 // -> ATF
  const int kSizeMax = std::max( kSizeM, kSizeN );

  auto WGD    = atf::tp( "WGD",    atf::interval<int>(1,kSizeMax) );
  auto MDIMCD = atf::tp( "MDIMCD", atf::interval<int>(1,kSizeM),   atf::divides( WGD ) /*&& atf::divides(kSizeM)*/ );
  auto NDIMCD = atf::tp( "NDIMCD", atf::interval<int>(1,kSizeN),   atf::divides( WGD ) /*&& atf::divides(kSizeN)*/ );
  auto MDIMAD = atf::tp( "MDIMAD", atf::interval<int>(1,kSizeM),   atf::divides( WGD ) && [&](auto MDIMAD) { return (MDIMCD*NDIMCD) % MDIMAD == 0; } && [&](auto MDIMAD) { return WGD % ( (MDIMCD*NDIMCD)/MDIMAD ) == 0; } );
  auto NDIMBD = atf::tp( "NDIMBD", atf::interval<int>(1,kSizeN),   atf::divides( WGD ) && [&](auto NDIMBD) { return (MDIMCD*NDIMCD) % NDIMBD == 0; } && [&](auto NDIMBD) { return WGD % ( (MDIMCD*NDIMCD)/NDIMBD ) == 0; } );
  auto KWID   = atf::tp( "KWID",   atf::interval<int>(1,kSizeK),   atf::divides( WGD ) );
  auto VWMD   = atf::tp( "VWMD",   {1, 2, 4, 8},                   atf::divides(WGD / MDIMCD) && atf::divides(WGD / MDIMAD) );
  auto VWND   = atf::tp( "VWND",   {1, 2, 4, 8},                   atf::divides(WGD / NDIMCD) && atf::divides(WGD / NDIMBD) && [&](auto){ return (((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD)+(((1 + ((kSizeM - 1) / WGD))*WGD * MDIMCD) / WGD) <= 1024; } );
  auto PADA   = atf::tp( "PADA",   {0, 1}  ); 
  auto PADB   = atf::tp( "PADB",   {0, 1}  );
#endif
  
  auto PRECISION = atf::tp( "PRECISION", {32} );

  auto dummy_kernel = [](auto){ return 0; };
  auto best_config =
atf::exhaustive( atf::cond::evaluations(0) )
  ( G(WGD,
      MDIMCD,
      NDIMCD,
      MDIMAD,
      NDIMBD,
      KWID,
      VWMD,
      VWND
     ),
    G( PADA ),
    G( PADB ),
    G( PRECISION )
  )
  ( dummy_kernel );
 
  auto end = std::chrono::system_clock::now();
  auto runtime_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
  std::cout << std::endl << "total runtime for search space generation = " << runtime_in_ms << "ms\n" << std::endl;

  return 0;
}
#endif


#ifdef OT_EXAMPLE
int main()
{

 auto O = atf::tp("O", atf::interval<int>(0, 3));

 auto align_loop_iterations = atf::tp("align_loop_iterations", atf::interval<int>(0, 200));

 auto align_threshold = atf::tp("align_threshold", atf::interval<int>(1, 200));

 auto allow_store_data_races = atf::tp("allow_store_data_races", atf::interval<int>(0, 1));

 auto asan_globals = atf::tp("asan_globals", atf::interval<int>(0, 1));

 auto asan_instrument_reads = atf::tp("asan_instrument_reads", atf::interval<int>(0, 1));

 auto asan_instrument_writes = atf::tp("asan_instrument_writes", atf::interval<int>(0, 1));

 auto asan_memintrin = atf::tp("asan_memintrin", atf::interval<int>(0, 1));

 auto asan_stack = atf::tp("asan_stack", atf::interval<int>(0, 1));

 auto asan_use_after_return = atf::tp("asan_use_after_return", atf::interval<int>(0, 1));

 auto builtin_expect_probability = atf::tp("builtin_expect_probability", atf::interval<int>(0, 100));

 auto case_values_threshold = atf::tp("case_values_threshold", atf::interval<int>(0, 200));

 auto comdat_sharing_probability = atf::tp("comdat_sharing_probability", atf::interval<int>(0, 200));

 auto cxx_max_namespaces_for_diagnostic_help = atf::tp("cxx_max_namespaces_for_diagnostic_help", atf::interval<int>(0, 200));

 auto early_inlining_insns = atf::tp("early_inlining_insns", atf::interval<int>(0, 200));

 auto gcse_after_reload_critical_fraction = atf::tp("gcse_after_reload_critical_fraction", atf::interval<int>(0, 200));

 auto gcse_after_reload_partial_fraction = atf::tp("gcse_after_reload_partial_fraction", atf::interval<int>(0, 200));

 auto gcse_cost_distance_ratio = atf::tp("gcse_cost_distance_ratio", atf::interval<int>(0, 200));

 auto gcse_unrestricted_cost = atf::tp("gcse_unrestricted_cost", atf::interval<int>(0, 200));

 auto ggc_min_expand = atf::tp("ggc_min_expand", atf::interval<int>(0, 200));

 auto ggc_min_heapsize = atf::tp("ggc_min_heapsize", atf::interval<int>(0, 200));

 auto graphite_max_bbs_per_function = atf::tp("graphite_max_bbs_per_function", atf::interval<int>(0, 200));

 auto graphite_max_nb_scop_params = atf::tp("graphite_max_nb_scop_params", atf::interval<int>(0, 200));

 auto hot_bb_count_ws_permille = atf::tp("hot_bb_count_ws_permille", atf::interval<int>(0, 1000));

 auto hot_bb_frequency_fraction = atf::tp("hot_bb_frequency_fraction", atf::interval<int>(0, 200));

 auto inline_min_speedup = atf::tp("inline_min_speedup", atf::interval<int>(0, 200));

 auto inline_unit_growth = atf::tp("inline_unit_growth", atf::interval<int>(0, 200));

 auto integer_share_limit = atf::tp("integer_share_limit", atf::interval<int>(2, 200));

 auto ipa_cp_array_index_hint_bonus = atf::tp("ipa_cp_array_index_hint_bonus", atf::interval<int>(0, 200));

 auto ipa_cp_eval_threshold = atf::tp("ipa_cp_eval_threshold", atf::interval<int>(0, 200));

 auto ipa_cp_loop_hint_bonus = atf::tp("ipa_cp_loop_hint_bonus", atf::interval<int>(0, 200));

 auto ipa_cp_value_list_size = atf::tp("ipa_cp_value_list_size", atf::interval<int>(0, 200));

 auto ipa_max_agg_items = atf::tp("ipa_max_agg_items", atf::interval<int>(0, 200));

 auto ipa_sra_ptr_growth_factor = atf::tp("ipa_sra_ptr_growth_factor", atf::interval<int>(0, 200));

 auto ipcp_unit_growth = atf::tp("ipcp_unit_growth", atf::interval<int>(0, 200));

 auto ira_loop_reserved_regs = atf::tp("ira_loop_reserved_regs", atf::interval<int>(0, 200));

 auto ira_max_conflict_table_size = atf::tp("ira_max_conflict_table_size", atf::interval<int>(0, 200));

 auto ira_max_loops_num = atf::tp("ira_max_loops_num", atf::interval<int>(0, 200));

 auto iv_always_prune_cand_set_bound = atf::tp("iv_always_prune_cand_set_bound", atf::interval<int>(0, 200));

 auto iv_consider_all_candidates_bound = atf::tp("iv_consider_all_candidates_bound", atf::interval<int>(0, 200));

 auto iv_max_considered_uses = atf::tp("iv_max_considered_uses", atf::interval<int>(0, 200));

 auto l1_cache_size = atf::tp("l1_cache_size", atf::interval<int>(0, 200));

 auto l2_cache_size = atf::tp("l2_cache_size", atf::interval<int>(0, 200));

 auto large_function_growth = atf::tp("large_function_growth", atf::interval<int>(0, 200));

 auto large_function_insns = atf::tp("large_function_insns", atf::interval<int>(0, 200));

 auto large_stack_frame = atf::tp("large_stack_frame", atf::interval<int>(0, 200));

 auto large_stack_frame_growth = atf::tp("large_stack_frame_growth", atf::interval<int>(0, 200));

 auto large_unit_insns = atf::tp("large_unit_insns", atf::interval<int>(0, 200));

 auto lim_expensive = atf::tp("lim_expensive", atf::interval<int>(0, 200));

 auto loop_block_tile_size = atf::tp("loop_block_tile_size", atf::interval<int>(0, 200));

 auto loop_invariant_max_bbs_in_loop = atf::tp("loop_invariant_max_bbs_in_loop", atf::interval<int>(0, 200));

 auto loop_max_datarefs_for_datadeps = atf::tp("loop_max_datarefs_for_datadeps", atf::interval<int>(0, 200));

 auto lra_max_considered_reload_pseudos = atf::tp("lra_max_considered_reload_pseudos", atf::interval<int>(0, 200));

 auto lto_min_partition = atf::tp("lto_min_partition", atf::interval<int>(0, 200));

 auto lto_partitions = atf::tp("lto_partitions", atf::interval<int>(1, 200));

 auto max_average_unrolled_insns = atf::tp("max_average_unrolled_insns", atf::interval<int>(0, 200));

 auto max_completely_peel_loop_nest_depth = atf::tp("max_completely_peel_loop_nest_depth", atf::interval<int>(0, 200));

 auto max_completely_peel_times = atf::tp("max_completely_peel_times", atf::interval<int>(0, 200));

 auto max_completely_peeled_insns = atf::tp("max_completely_peeled_insns", atf::interval<int>(0, 200));

 auto max_crossjump_edges = atf::tp("max_crossjump_edges", atf::interval<int>(0, 200));

 auto max_cse_insns = atf::tp("max_cse_insns", atf::interval<int>(0, 200));

 auto max_cse_path_length = atf::tp("max_cse_path_length", atf::interval<int>(1, 200));

 auto max_cselib_memory_locations = atf::tp("max_cselib_memory_locations", atf::interval<int>(0, 200));

 auto max_delay_slot_insn_search = atf::tp("max_delay_slot_insn_search", atf::interval<int>(0, 200));

 auto max_delay_slot_live_search = atf::tp("max_delay_slot_live_search", atf::interval<int>(0, 200));

 auto max_dse_active_local_stores = atf::tp("max_dse_active_local_stores", atf::interval<int>(0, 200));

 auto max_early_inliner_iterations = atf::tp("max_early_inliner_iterations", atf::interval<int>(0, 200));

 auto max_fields_for_field_sensitive = atf::tp("max_fields_for_field_sensitive", atf::interval<int>(0, 200));

 auto max_gcse_insertion_ratio = atf::tp("max_gcse_insertion_ratio", atf::interval<int>(0, 200));

 auto max_gcse_memory = atf::tp("max_gcse_memory", atf::interval<int>(0, 200));

 auto max_goto_duplication_insns = atf::tp("max_goto_duplication_insns", atf::interval<int>(0, 200));

 auto max_grow_copy_bb_insns = atf::tp("max_grow_copy_bb_insns", atf::interval<int>(0, 200));

 auto max_hoist_depth = atf::tp("max_hoist_depth", atf::interval<int>(0, 200));

 auto max_inline_insns_auto = atf::tp("max_inline_insns_auto", atf::interval<int>(0, 200));

 auto max_inline_insns_recursive = atf::tp("max_inline_insns_recursive", atf::interval<int>(0, 200));

 auto max_inline_insns_recursive_auto = atf::tp("max_inline_insns_recursive_auto", atf::interval<int>(0, 200));

 auto max_inline_insns_single = atf::tp("max_inline_insns_single", atf::interval<int>(0, 200));

 auto max_inline_recursive_depth = atf::tp("max_inline_recursive_depth", atf::interval<int>(0, 200));

 auto max_inline_recursive_depth_auto = atf::tp("max_inline_recursive_depth_auto", atf::interval<int>(0, 200));

 auto max_iterations_computation_cost = atf::tp("max_iterations_computation_cost", atf::interval<int>(0, 200));

 auto max_iterations_to_track = atf::tp("max_iterations_to_track", atf::interval<int>(0, 200));

 auto max_jump_thread_duplication_stmts = atf::tp("max_jump_thread_duplication_stmts", atf::interval<int>(0, 200));

 auto max_last_value_rtl = atf::tp("max_last_value_rtl", atf::interval<int>(0, 200));

 auto max_modulo_backtrack_attempts = atf::tp("max_modulo_backtrack_attempts", atf::interval<int>(0, 200));

 auto max_once_peeled_insns = atf::tp("max_once_peeled_insns", atf::interval<int>(0, 200));

 auto max_partial_antic_length = atf::tp("max_partial_antic_length", atf::interval<int>(0, 200));

 auto max_peel_branches = atf::tp("max_peel_branches", atf::interval<int>(0, 200));

 auto max_peel_times = atf::tp("max_peel_times", atf::interval<int>(0, 200));

 auto max_peeled_insns = atf::tp("max_peeled_insns", atf::interval<int>(0, 200));

 auto max_pending_list_length = atf::tp("max_pending_list_length", atf::interval<int>(0, 200));

 auto max_pipeline_region_blocks = atf::tp("max_pipeline_region_blocks", atf::interval<int>(0, 200));

 auto max_pipeline_region_insns = atf::tp("max_pipeline_region_insns", atf::interval<int>(0, 200));

 auto max_predicted_iterations = atf::tp("max_predicted_iterations", atf::interval<int>(0, 200));

 auto max_reload_search_insns = atf::tp("max_reload_search_insns", atf::interval<int>(0, 200));

 auto max_sched_extend_regions_iters = atf::tp("max_sched_extend_regions_iters", atf::interval<int>(0, 200));

 auto max_sched_insn_conflict_delay = atf::tp("max_sched_insn_conflict_delay", atf::interval<int>(1, 10));

 auto max_sched_ready_insns = atf::tp("max_sched_ready_insns", atf::interval<int>(0, 200));

 auto max_sched_region_blocks = atf::tp("max_sched_region_blocks", atf::interval<int>(0, 200));

 auto max_sched_region_insns = atf::tp("max_sched_region_insns", atf::interval<int>(0, 200));

 auto max_slsr_cand_scan = atf::tp("max_slsr_cand_scan", atf::interval<int>(1, 999999));

 auto max_stores_to_sink = atf::tp("max_stores_to_sink", atf::interval<int>(0, 200));

 auto max_tail_merge_comparisons = atf::tp("max_tail_merge_comparisons", atf::interval<int>(0, 200));

 auto max_tail_merge_iterations = atf::tp("max_tail_merge_iterations", atf::interval<int>(0, 200));

 auto max_tracked_strlens = atf::tp("max_tracked_strlens", atf::interval<int>(0, 200));

 auto max_unroll_times = atf::tp("max_unroll_times", atf::interval<int>(0, 200));

 auto max_unrolled_insns = atf::tp("max_unrolled_insns", atf::interval<int>(0, 200));

 auto max_unswitch_insns = atf::tp("max_unswitch_insns", atf::interval<int>(0, 200));

 auto max_unswitch_level = atf::tp("max_unswitch_level", atf::interval<int>(0, 200));

 auto max_variable_expansions_in_unroller = atf::tp("max_variable_expansions_in_unroller", atf::interval<int>(0, 200));

 auto max_vartrack_expr_depth = atf::tp("max_vartrack_expr_depth", atf::interval<int>(0, 200));

 auto max_vartrack_reverse_op_size = atf::tp("max_vartrack_reverse_op_size", atf::interval<int>(0, 200));

 auto max_vartrack_size = atf::tp("max_vartrack_size", atf::interval<int>(0, 200));

 auto min_crossjump_insns = atf::tp("min_crossjump_insns", atf::interval<int>(1, 200));

 auto min_inline_recursive_probability = atf::tp("min_inline_recursive_probability", atf::interval<int>(0, 200));

 auto min_insn_to_prefetch_ratio = atf::tp("min_insn_to_prefetch_ratio", atf::interval<int>(0, 200));

 auto min_size_for_stack_sharing = atf::tp("min_size_for_stack_sharing", atf::interval<int>(0, 200));

 auto min_spec_prob = atf::tp("min_spec_prob", atf::interval<int>(0, 200));

 auto min_vect_loop_bound = atf::tp("min_vect_loop_bound", atf::interval<int>(1, 200));

 auto partial_inlining_entry_probability = atf::tp("partial_inlining_entry_probability", atf::interval<int>(0, 200));

 auto predictable_branch_outcome = atf::tp("predictable_branch_outcome", atf::interval<int>(0, 50));

 auto prefetch_latency = atf::tp("prefetch_latency", atf::interval<int>(0, 200));

 auto prefetch_min_insn_to_mem_ratio = atf::tp("prefetch_min_insn_to_mem_ratio", atf::interval<int>(0, 200));

 auto sccvn_max_alias_queries_per_access = atf::tp("sccvn_max_alias_queries_per_access", atf::interval<int>(0, 200));

 auto sccvn_max_scc_size = atf::tp("sccvn_max_scc_size", atf::interval<int>(10, 200));

 auto scev_max_expr_complexity = atf::tp("scev_max_expr_complexity", atf::interval<int>(0, 200));

 auto scev_max_expr_size = atf::tp("scev_max_expr_size", atf::interval<int>(0, 200));

 auto sched_mem_true_dep_cost = atf::tp("sched_mem_true_dep_cost", atf::interval<int>(0, 200));

 auto sched_pressure_algorithm = atf::tp("sched_pressure_algorithm", atf::interval<int>(1, 2));

 auto sched_spec_prob_cutoff = atf::tp("sched_spec_prob_cutoff", atf::interval<int>(0, 100));

 auto sched_state_edge_prob_cutoff = atf::tp("sched_state_edge_prob_cutoff", atf::interval<int>(0, 100));

 auto selsched_insns_to_rename = atf::tp("selsched_insns_to_rename", atf::interval<int>(0, 200));

 auto selsched_max_lookahead = atf::tp("selsched_max_lookahead", atf::interval<int>(0, 200));

 auto selsched_max_sched_times = atf::tp("selsched_max_sched_times", atf::interval<int>(0, 200));

 auto simultaneous_prefetches = atf::tp("simultaneous_prefetches", atf::interval<int>(0, 200));

 auto sink_frequency_threshold = atf::tp("sink_frequency_threshold", atf::interval<int>(0, 100));

 auto slp_max_insns_in_bb = atf::tp("slp_max_insns_in_bb", atf::interval<int>(0, 200));

 auto sms_dfa_history = atf::tp("sms_dfa_history", atf::interval<int>(0, 200));

 auto sms_loop_average_count_threshold = atf::tp("sms_loop_average_count_threshold", atf::interval<int>(0, 200));

 auto sms_max_ii_factor = atf::tp("sms_max_ii_factor", atf::interval<int>(0, 200));

 auto sms_min_sc = atf::tp("sms_min_sc", atf::interval<int>(1, 200));

 auto ssp_buffer_size = atf::tp("ssp_buffer_size", atf::interval<int>(1, 200));

 auto switch_conversion_max_branch_ratio = atf::tp("switch_conversion_max_branch_ratio", atf::interval<int>(1, 200));

 auto tm_max_aggregate_size = atf::tp("tm_max_aggregate_size", atf::interval<int>(0, 200));

 auto tracer_dynamic_coverage = atf::tp("tracer_dynamic_coverage", atf::interval<int>(0, 100));

 auto tracer_dynamic_coverage_feedback = atf::tp("tracer_dynamic_coverage_feedback", atf::interval<int>(0, 100));

 auto tracer_max_code_growth = atf::tp("tracer_max_code_growth", atf::interval<int>(0, 200));

 auto tracer_min_branch_probability = atf::tp("tracer_min_branch_probability", atf::interval<int>(0, 100));

 auto tracer_min_branch_probability_feedback = atf::tp("tracer_min_branch_probability_feedback", atf::interval<int>(0, 100));

 auto tracer_min_branch_ratio = atf::tp("tracer_min_branch_ratio", atf::interval<int>(0, 100));

 auto tree_reassoc_width = atf::tp("tree_reassoc_width", atf::interval<int>(0, 200));

 auto uninit_control_dep_attempts = atf::tp("uninit_control_dep_attempts", atf::interval<int>(1, 200));

 auto unlikely_bb_count_fraction = atf::tp("unlikely_bb_count_fraction", atf::interval<int>(1, 10000));

 auto use_canonical_types = atf::tp("use_canonical_types", atf::interval<int>(0, 1));

 auto vect_max_version_for_alias_checks = atf::tp("vect_max_version_for_alias_checks", atf::interval<int>(0, 200));

 auto vect_max_version_for_alignment_checks = atf::tp("vect_max_version_for_alignment_checks", atf::interval<int>(0, 200));

 auto faggressive_loop_optimizations = atf::tp("faggressive_loop_optimizations", atf::interval<int>(0,1));

 auto falign_functions = atf::tp("falign_functions", atf::interval<int>(0,1));

 auto falign_jumps = atf::tp("falign_jumps", atf::interval<int>(0,1));

 auto falign_labels = atf::tp("falign_labels", atf::interval<int>(0,1));

 auto falign_loops = atf::tp("falign_loops", atf::interval<int>(0,1));

 auto fassociative_math = atf::tp("fassociative_math", atf::interval<int>(0,1));

 auto fasynchronous_unwind_tables = atf::tp("fasynchronous_unwind_tables", atf::interval<int>(0,1));

 auto fauto_inc_dec = atf::tp("fauto_inc_dec", atf::interval<int>(0,1));

 auto fbranch_count_reg = atf::tp("fbranch_count_reg", atf::interval<int>(0,1));

 auto fbranch_probabilities = atf::tp("fbranch_probabilities", atf::interval<int>(0,1));

 auto fbranch_target_load_optimize = atf::tp("fbranch_target_load_optimize", atf::interval<int>(0,1));

 auto fbranch_target_load_optimize2 = atf::tp("fbranch_target_load_optimize2", atf::interval<int>(0,1));

 auto fbtr_bb_exclusive = atf::tp("fbtr_bb_exclusive", atf::interval<int>(0,1));

 auto fcaller_saves = atf::tp("fcaller_saves", atf::interval<int>(0,1));

 auto fcombine_stack_adjustments = atf::tp("fcombine_stack_adjustments", atf::interval<int>(0,1));

 auto fcompare_elim = atf::tp("fcompare_elim", atf::interval<int>(0,1));

 auto fconserve_stack = atf::tp("fconserve_stack", atf::interval<int>(0,1));

 auto fcprop_registers = atf::tp("fcprop_registers", atf::interval<int>(0,1));

 auto fcrossjumping = atf::tp("fcrossjumping", atf::interval<int>(0,1));

 auto fcse_follow_jumps = atf::tp("fcse_follow_jumps", atf::interval<int>(0,1));

 auto fcx_fortran_rules = atf::tp("fcx_fortran_rules", atf::interval<int>(0,1));

 auto fcx_limited_range = atf::tp("fcx_limited_range", atf::interval<int>(0,1));

 auto fdce = atf::tp("fdce", atf::interval<int>(0,1));

 auto fdefer_pop = atf::tp("fdefer_pop", atf::interval<int>(0,1));

 auto fdelete_dead_exceptions = atf::tp("fdelete_dead_exceptions", atf::interval<int>(0,1));

 auto fdelete_null_pointer_checks = atf::tp("fdelete_null_pointer_checks", atf::interval<int>(0,1));

 auto fdevirtualize = atf::tp("fdevirtualize", atf::interval<int>(0,1));

 auto fdevirtualize_speculatively = atf::tp("fdevirtualize_speculatively", atf::interval<int>(0,1));

 auto fdse = atf::tp("fdse", atf::interval<int>(0,1));

 auto fearly_inlining = atf::tp("fearly_inlining", atf::interval<int>(0,1));

 auto fexceptions = atf::tp("fexceptions", atf::interval<int>(0,1));

 auto fexpensive_optimizations = atf::tp("fexpensive_optimizations", atf::interval<int>(0,1));

 auto ffinite_math_only = atf::tp("ffinite_math_only", atf::interval<int>(0,1));

 auto ffloat_store = atf::tp("ffloat_store", atf::interval<int>(0,1));

 auto fforward_propagate = atf::tp("fforward_propagate", atf::interval<int>(0,1));

 auto ffunction_cse = atf::tp("ffunction_cse", atf::interval<int>(0,1));

 auto fgcse = atf::tp("fgcse", atf::interval<int>(0,1));

 auto fgcse_after_reload = atf::tp("fgcse_after_reload", atf::interval<int>(0,1));

 auto fgcse_las = atf::tp("fgcse_las", atf::interval<int>(0,1));

 auto fgcse_lm = atf::tp("fgcse_lm", atf::interval<int>(0,1));

 auto fgcse_sm = atf::tp("fgcse_sm", atf::interval<int>(0,1));

 auto fgraphite = atf::tp("fgraphite", atf::interval<int>(0,1));

 auto fgraphite_identity = atf::tp("fgraphite_identity", atf::interval<int>(0,1));

 auto fguess_branch_probability = atf::tp("fguess_branch_probability", atf::interval<int>(0,1));

 auto fhoist_adjacent_loads = atf::tp("fhoist_adjacent_loads", atf::interval<int>(0,1));

 auto fif_conversion = atf::tp("fif_conversion", atf::interval<int>(0,1));

 auto fif_conversion2 = atf::tp("fif_conversion2", atf::interval<int>(0,1));

 auto findirect_inlining = atf::tp("findirect_inlining", atf::interval<int>(0,1));

 auto finline = atf::tp("finline", atf::interval<int>(0,1));

 auto finline_atomics = atf::tp("finline_atomics", atf::interval<int>(0,1));

 auto finline_functions = atf::tp("finline_functions", atf::interval<int>(0,1));

 auto finline_functions_called_once = atf::tp("finline_functions_called_once", atf::interval<int>(0,1));

 auto finline_small_functions = atf::tp("finline_small_functions", atf::interval<int>(0,1));

 auto fipa_cp = atf::tp("fipa_cp", atf::interval<int>(0,1));

 auto fipa_cp_alignment = atf::tp("fipa_cp_alignment", atf::interval<int>(0,1));

 auto fipa_cp_clone = atf::tp("fipa_cp_clone", atf::interval<int>(0,1));

 auto fipa_icf = atf::tp("fipa_icf", atf::interval<int>(0,1));

 auto fipa_icf_functions = atf::tp("fipa_icf_functions", atf::interval<int>(0,1));

 auto fipa_profile = atf::tp("fipa_profile", atf::interval<int>(0,1));

 auto fipa_pta = atf::tp("fipa_pta", atf::interval<int>(0,1));

 auto fipa_pure_const = atf::tp("fipa_pure_const", atf::interval<int>(0,1));

 auto fipa_ra = atf::tp("fipa_ra", atf::interval<int>(0,1));

 auto fipa_reference = atf::tp("fipa_reference", atf::interval<int>(0,1));

 auto fipa_sra = atf::tp("fipa_sra", atf::interval<int>(0,1));

 auto fira_hoist_pressure = atf::tp("fira_hoist_pressure", atf::interval<int>(0,1));

 auto fira_loop_pressure = atf::tp("fira_loop_pressure", atf::interval<int>(0,1));

 auto fira_share_save_slots = atf::tp("fira_share_save_slots", atf::interval<int>(0,1));

 auto fira_share_spill_slots = atf::tp("fira_share_spill_slots", atf::interval<int>(0,1));

 auto fisolate_erroneous_paths_attribute = atf::tp("fisolate_erroneous_paths_attribute", atf::interval<int>(0,1));

 auto fisolate_erroneous_paths_dereference = atf::tp("fisolate_erroneous_paths_dereference", atf::interval<int>(0,1));

 auto fivopts = atf::tp("fivopts", atf::interval<int>(0,1));

 auto fjump_tables = atf::tp("fjump_tables", atf::interval<int>(0,1));

 auto fkeep_gc_roots_live = atf::tp("fkeep_gc_roots_live", atf::interval<int>(0,1));

 auto flifetime_dse = atf::tp("flifetime_dse", atf::interval<int>(0,1));

 auto flive_range_shrinkage = atf::tp("flive_range_shrinkage", atf::interval<int>(0,1));

 auto floop_nest_optimize = atf::tp("floop_nest_optimize", atf::interval<int>(0,1));

 auto floop_parallelize_all = atf::tp("floop_parallelize_all", atf::interval<int>(0,1));

 auto flra_remat = atf::tp("flra_remat", atf::interval<int>(0,1));

 auto fmath_errno = atf::tp("fmath_errno", atf::interval<int>(0,1));

 auto fmodulo_sched = atf::tp("fmodulo_sched", atf::interval<int>(0,1));

 auto fmodulo_sched_allow_regmoves = atf::tp("fmodulo_sched_allow_regmoves", atf::interval<int>(0,1));

 auto fmove_loop_invariants = atf::tp("fmove_loop_invariants", atf::interval<int>(0,1));

 auto fnon_call_exceptions = atf::tp("fnon_call_exceptions", atf::interval<int>(0,1));

 auto fnothrow_opt = atf::tp("fnothrow_opt", atf::interval<int>(0,1));

 auto fomit_frame_pointer = atf::tp("fomit_frame_pointer", atf::interval<int>(0,1));

 auto fopt_info = atf::tp("fopt_info", atf::interval<int>(0,1));

 auto foptimize_sibling_calls = atf::tp("foptimize_sibling_calls", atf::interval<int>(0,1));

 auto foptimize_strlen = atf::tp("foptimize_strlen", atf::interval<int>(0,1));

 auto fpartial_inlining = atf::tp("fpartial_inlining", atf::interval<int>(0,1));

 auto fpeel_loops = atf::tp("fpeel_loops", atf::interval<int>(0,1));

 auto fpeephole = atf::tp("fpeephole", atf::interval<int>(0,1));

 auto fpeephole2 = atf::tp("fpeephole2", atf::interval<int>(0,1));

 auto fplt = atf::tp("fplt", atf::interval<int>(0,1));

 auto fpredictive_commoning = atf::tp("fpredictive_commoning", atf::interval<int>(0,1));

 auto fprefetch_loop_arrays = atf::tp("fprefetch_loop_arrays", atf::interval<int>(0,1));

 auto freciprocal_math = atf::tp("freciprocal_math", atf::interval<int>(0,1));

 auto freg_struct_return = atf::tp("freg_struct_return", atf::interval<int>(0,1));

 auto frename_registers = atf::tp("frename_registers", atf::interval<int>(0,1));

 auto freorder_blocks = atf::tp("freorder_blocks", atf::interval<int>(0,1));

 auto freorder_blocks_and_partition = atf::tp("freorder_blocks_and_partition", atf::interval<int>(0,1));

 auto freorder_functions = atf::tp("freorder_functions", atf::interval<int>(0,1));

 auto frerun_cse_after_loop = atf::tp("frerun_cse_after_loop", atf::interval<int>(0,1));

 auto freschedule_modulo_scheduled_loops = atf::tp("freschedule_modulo_scheduled_loops", atf::interval<int>(0,1));

 auto frounding_math = atf::tp("frounding_math", atf::interval<int>(0,1));

 auto frtti = atf::tp("frtti", atf::interval<int>(0,1));

 auto fsched_critical_path_heuristic = atf::tp("fsched_critical_path_heuristic", atf::interval<int>(0,1));

 auto fsched_dep_count_heuristic = atf::tp("fsched_dep_count_heuristic", atf::interval<int>(0,1));

 auto fsched_group_heuristic = atf::tp("fsched_group_heuristic", atf::interval<int>(0,1));

 auto fsched_interblock = atf::tp("fsched_interblock", atf::interval<int>(0,1));

 auto fsched_last_insn_heuristic = atf::tp("fsched_last_insn_heuristic", atf::interval<int>(0,1));

 auto fsched_pressure = atf::tp("fsched_pressure", atf::interval<int>(0,1));

 auto fsched_rank_heuristic = atf::tp("fsched_rank_heuristic", atf::interval<int>(0,1));

 auto fsched_spec = atf::tp("fsched_spec", atf::interval<int>(0,1));

 auto fsched_spec_insn_heuristic = atf::tp("fsched_spec_insn_heuristic", atf::interval<int>(0,1));

 auto fsched_spec_load = atf::tp("fsched_spec_load", atf::interval<int>(0,1));

 auto fsched_spec_load_dangerous = atf::tp("fsched_spec_load_dangerous", atf::interval<int>(0,1));

 auto fsched_stalled_insns = atf::tp("fsched_stalled_insns", atf::interval<int>(0,1));

 auto fsched_stalled_insns_dep = atf::tp("fsched_stalled_insns_dep", atf::interval<int>(0,1));

 auto fsched2_use_superblocks = atf::tp("fsched2_use_superblocks", atf::interval<int>(0,1));

 auto fschedule_fusion = atf::tp("fschedule_fusion", atf::interval<int>(0,1));

 auto fschedule_insns = atf::tp("fschedule_insns", atf::interval<int>(0,1));

 auto fschedule_insns2 = atf::tp("fschedule_insns2", atf::interval<int>(0,1));

 auto fsel_sched_pipelining = atf::tp("fsel_sched_pipelining", atf::interval<int>(0,1));

 auto fsel_sched_pipelining_outer_loops = atf::tp("fsel_sched_pipelining_outer_loops", atf::interval<int>(0,1));

 auto fsel_sched_reschedule_pipelined = atf::tp("fsel_sched_reschedule_pipelined", atf::interval<int>(0,1));

 auto fselective_scheduling = atf::tp("fselective_scheduling", atf::interval<int>(0,1));

 auto fselective_scheduling2 = atf::tp("fselective_scheduling2", atf::interval<int>(0,1));

 auto fshort_enums = atf::tp("fshort_enums", atf::interval<int>(0,1));

 auto fshort_wchar = atf::tp("fshort_wchar", atf::interval<int>(0,1));

 auto fshrink_wrap = atf::tp("fshrink_wrap", atf::interval<int>(0,1));

 auto fsignaling_nans = atf::tp("fsignaling_nans", atf::interval<int>(0,1));

 auto fsigned_zeros = atf::tp("fsigned_zeros", atf::interval<int>(0,1));

 auto fsingle_precision_constant = atf::tp("fsingle_precision_constant", atf::interval<int>(0,1));

 auto fsplit_ivs_in_unroller = atf::tp("fsplit_ivs_in_unroller", atf::interval<int>(0,1));

 auto fsplit_paths = atf::tp("fsplit_paths", atf::interval<int>(0,1));

 auto fsplit_wide_types = atf::tp("fsplit_wide_types", atf::interval<int>(0,1));

 auto fssa_backprop = atf::tp("fssa_backprop", atf::interval<int>(0,1));

 auto fssa_phiopt = atf::tp("fssa_phiopt", atf::interval<int>(0,1));

 auto fstdarg_opt = atf::tp("fstdarg_opt", atf::interval<int>(0,1));

 auto fstrict_aliasing = atf::tp("fstrict_aliasing", atf::interval<int>(0,1));

 auto fstrict_enums = atf::tp("fstrict_enums", atf::interval<int>(0,1));

 auto fstrict_overflow = atf::tp("fstrict_overflow", atf::interval<int>(0,1));

 auto fstrict_volatile_bitfields = atf::tp("fstrict_volatile_bitfields", atf::interval<int>(0,1));

 auto fthread_jumps = atf::tp("fthread_jumps", atf::interval<int>(0,1));

 auto fno_threadsafe_statics = atf::tp("fno_threadsafe_statics", atf::interval<int>(0,1));

 auto ftracer = atf::tp("ftracer", atf::interval<int>(0,1));

 auto ftrapping_math = atf::tp("ftrapping_math", atf::interval<int>(0,1));

 auto ftrapv = atf::tp("ftrapv", atf::interval<int>(0,1));

 auto ftree_bit_ccp = atf::tp("ftree_bit_ccp", atf::interval<int>(0,1));

 auto ftree_builtin_call_dce = atf::tp("ftree_builtin_call_dce", atf::interval<int>(0,1));

 auto ftree_ccp = atf::tp("ftree_ccp", atf::interval<int>(0,1));

 auto ftree_ch = atf::tp("ftree_ch", atf::interval<int>(0,1));

 auto ftree_coalesce_vars = atf::tp("ftree_coalesce_vars", atf::interval<int>(0,1));

 auto ftree_copy_prop = atf::tp("ftree_copy_prop", atf::interval<int>(0,1));

 auto ftree_cselim = atf::tp("ftree_cselim", atf::interval<int>(0,1));

 auto ftree_dce = atf::tp("ftree_dce", atf::interval<int>(0,1));

 auto ftree_dominator_opts = atf::tp("ftree_dominator_opts", atf::interval<int>(0,1));

 auto ftree_dse = atf::tp("ftree_dse", atf::interval<int>(0,1));

 auto ftree_forwprop = atf::tp("ftree_forwprop", atf::interval<int>(0,1));

 auto ftree_fre = atf::tp("ftree_fre", atf::interval<int>(0,1));

 auto ftree_loop_distribute_patterns = atf::tp("ftree_loop_distribute_patterns", atf::interval<int>(0,1));

 auto ftree_loop_distribution = atf::tp("ftree_loop_distribution", atf::interval<int>(0,1));

 auto ftree_loop_if_convert = atf::tp("ftree_loop_if_convert", atf::interval<int>(0,1));

 auto ftree_loop_if_convert_stores = atf::tp("ftree_loop_if_convert_stores", atf::interval<int>(0,1));

 auto ftree_loop_im = atf::tp("ftree_loop_im", atf::interval<int>(0,1));

 auto ftree_loop_ivcanon = atf::tp("ftree_loop_ivcanon", atf::interval<int>(0,1));

 auto ftree_loop_optimize = atf::tp("ftree_loop_optimize", atf::interval<int>(0,1));

 auto ftree_loop_vectorize = atf::tp("ftree_loop_vectorize", atf::interval<int>(0,1));

 auto ftree_lrs = atf::tp("ftree_lrs", atf::interval<int>(0,1));

 auto ftree_partial_pre = atf::tp("ftree_partial_pre", atf::interval<int>(0,1));

 auto ftree_phiprop = atf::tp("ftree_phiprop", atf::interval<int>(0,1));

 auto ftree_pre = atf::tp("ftree_pre", atf::interval<int>(0,1));

 auto ftree_pta = atf::tp("ftree_pta", atf::interval<int>(0,1));

 auto ftree_reassoc = atf::tp("ftree_reassoc", atf::interval<int>(0,1));

 auto ftree_scev_cprop = atf::tp("ftree_scev_cprop", atf::interval<int>(0,1));

 auto ftree_sink = atf::tp("ftree_sink", atf::interval<int>(0,1));

 auto ftree_slp_vectorize = atf::tp("ftree_slp_vectorize", atf::interval<int>(0,1));

 auto ftree_slsr = atf::tp("ftree_slsr", atf::interval<int>(0,1));

 auto ftree_sra = atf::tp("ftree_sra", atf::interval<int>(0,1));

 auto ftree_switch_conversion = atf::tp("ftree_switch_conversion", atf::interval<int>(0,1));

 auto ftree_tail_merge = atf::tp("ftree_tail_merge", atf::interval<int>(0,1));

 auto ftree_ter = atf::tp("ftree_ter", atf::interval<int>(0,1));

 auto ftree_vectorize = atf::tp("ftree_vectorize", atf::interval<int>(0,1));

 auto ftree_vrp = atf::tp("ftree_vrp", atf::interval<int>(0,1));

 auto funconstrained_commons = atf::tp("funconstrained_commons", atf::interval<int>(0,1));

 auto funroll_all_loops = atf::tp("funroll_all_loops", atf::interval<int>(0,1));

 auto funroll_loops = atf::tp("funroll_loops", atf::interval<int>(0,1));

 auto funsafe_loop_optimizations = atf::tp("funsafe_loop_optimizations", atf::interval<int>(0,1));

 auto funsafe_math_optimizations = atf::tp("funsafe_math_optimizations", atf::interval<int>(0,1));

 auto funswitch_loops = atf::tp("funswitch_loops", atf::interval<int>(0,1));

 auto funwind_tables = atf::tp("funwind_tables", atf::interval<int>(0,1));

 auto fvar_tracking = atf::tp("fvar_tracking", atf::interval<int>(0,1));

 auto fvar_tracking_assignments = atf::tp("fvar_tracking_assignments", atf::interval<int>(0,1));

 auto fvar_tracking_assignments_toggle = atf::tp("fvar_tracking_assignments_toggle", atf::interval<int>(0,1));

 auto fvar_tracking_uninit = atf::tp("fvar_tracking_uninit", atf::interval<int>(0,1));

 auto fvariable_expansion_in_unroller = atf::tp("fvariable_expansion_in_unroller", atf::interval<int>(0,1));

 auto fvpt = atf::tp("fvpt", atf::interval<int>(0,1));

 auto fweb = atf::tp("fweb", atf::interval<int>(0,1));

 auto fwrapv = atf::tp("fwrapv", atf::interval<int>(0,1));

 auto cf = [](auto){ return 1; };

 auto best = atf::open_tuner( atf::cond::duration<std::chrono::seconds>(30) )(O)(align_loop_iterations)(align_threshold)(allow_store_data_races)(asan_globals)(asan_instrument_reads)(asan_instrument_writes)(asan_memintrin)(asan_stack)(asan_use_after_return)(builtin_expect_probability)(case_values_threshold)(comdat_sharing_probability)(cxx_max_namespaces_for_diagnostic_help)(early_inlining_insns)(gcse_after_reload_critical_fraction)(gcse_after_reload_partial_fraction)(gcse_cost_distance_ratio)(gcse_unrestricted_cost)(ggc_min_expand)(ggc_min_heapsize)(graphite_max_bbs_per_function)(graphite_max_nb_scop_params)(hot_bb_count_ws_permille)(hot_bb_frequency_fraction)(inline_min_speedup)(inline_unit_growth)(integer_share_limit)(ipa_cp_array_index_hint_bonus)(ipa_cp_eval_threshold)(ipa_cp_loop_hint_bonus)(ipa_cp_value_list_size)(ipa_max_agg_items)(ipa_sra_ptr_growth_factor)(ipcp_unit_growth)(ira_loop_reserved_regs)(ira_max_conflict_table_size)(ira_max_loops_num)(iv_always_prune_cand_set_bound)(iv_consider_all_candidates_bound)(iv_max_considered_uses)(l2_cache_size)(large_function_growth)(large_function_insns)(large_stack_frame)(large_stack_frame_growth)(large_unit_insns)(lim_expensive)(loop_block_tile_size)(loop_invariant_max_bbs_in_loop)(loop_max_datarefs_for_datadeps)(lra_max_considered_reload_pseudos)(lto_min_partition)(lto_partitions)(max_average_unrolled_insns)(max_completely_peel_loop_nest_depth)(max_completely_peel_times)(max_completely_peeled_insns)(max_crossjump_edges)(max_cse_insns)(max_cse_path_length)(max_cselib_memory_locations)(max_delay_slot_insn_search)(max_delay_slot_live_search)(max_dse_active_local_stores)(max_early_inliner_iterations)(max_fields_for_field_sensitive)(max_gcse_insertion_ratio)(max_gcse_memory)(max_goto_duplication_insns)(max_grow_copy_bb_insns)(max_hoist_depth)(max_inline_insns_auto)(max_inline_insns_recursive)(max_inline_insns_recursive_auto)(max_inline_insns_single)(max_inline_recursive_depth)(max_inline_recursive_depth_auto)(max_iterations_computation_cost)(max_iterations_to_track)(max_jump_thread_duplication_stmts)(max_last_value_rtl)(max_modulo_backtrack_attempts)(max_once_peeled_insns)(max_partial_antic_length)(max_peel_branches)(max_peel_times)(max_peeled_insns)(max_pending_list_length)(max_pipeline_region_blocks)(max_pipeline_region_insns)(max_predicted_iterations)(max_reload_search_insns)(max_sched_extend_regions_iters)(max_sched_insn_conflict_delay)(max_sched_ready_insns)(max_sched_region_blocks)(max_sched_region_insns)(max_slsr_cand_scan)(max_stores_to_sink)(max_tail_merge_comparisons)(max_tail_merge_iterations)(max_tracked_strlens)(max_unroll_times)(max_unrolled_insns)(max_unswitch_insns)(max_unswitch_level)(max_variable_expansions_in_unroller)(max_vartrack_expr_depth)(max_vartrack_reverse_op_size)(max_vartrack_size)(min_crossjump_insns)(min_inline_recursive_probability)(min_insn_to_prefetch_ratio)(min_size_for_stack_sharing)(min_spec_prob)(min_vect_loop_bound)(partial_inlining_entry_probability)(predictable_branch_outcome)(prefetch_latency)(prefetch_min_insn_to_mem_ratio)(sccvn_max_alias_queries_per_access)(sccvn_max_scc_size)(scev_max_expr_complexity)(scev_max_expr_size)(sched_mem_true_dep_cost)(sched_pressure_algorithm)(sched_spec_prob_cutoff)(sched_state_edge_prob_cutoff)(selsched_insns_to_rename)(selsched_max_lookahead)(selsched_max_sched_times)(simultaneous_prefetches)(sink_frequency_threshold)(slp_max_insns_in_bb)(sms_dfa_history)(sms_loop_average_count_threshold)(sms_max_ii_factor)(sms_min_sc)(ssp_buffer_size)(switch_conversion_max_branch_ratio)(tm_max_aggregate_size)(tracer_dynamic_coverage)(tracer_dynamic_coverage_feedback)(tracer_max_code_growth)(tracer_min_branch_probability)(tracer_min_branch_probability_feedback)(tracer_min_branch_ratio)(tree_reassoc_width)(uninit_control_dep_attempts)(unlikely_bb_count_fraction)(use_canonical_types)(vect_max_version_for_alias_checks)(vect_max_version_for_alignment_checks)(faggressive_loop_optimizations)(falign_functions)(falign_jumps)(falign_labels)(falign_loops)(fassociative_math)(fasynchronous_unwind_tables)(fauto_inc_dec)(fbranch_count_reg)(fbranch_probabilities)(fbranch_target_load_optimize)(fbranch_target_load_optimize2)(fbtr_bb_exclusive)(fcaller_saves)(fcombine_stack_adjustments)(fcompare_elim)(fconserve_stack)(fcprop_registers)(fcrossjumping)(fcse_follow_jumps)(fcx_fortran_rules)(fcx_limited_range)(fdce)(fdefer_pop)(fdelete_dead_exceptions)(fdelete_null_pointer_checks)(fdevirtualize)(fdevirtualize_speculatively)(fdse)(fearly_inlining)(fexceptions)(fexpensive_optimizations)(ffinite_math_only)(ffloat_store)(fforward_propagate)(ffunction_cse)(fgcse)(fgcse_after_reload)(fgcse_las)(fgcse_lm)(fgcse_sm)(fgraphite)(fgraphite_identity)(fguess_branch_probability)(fhoist_adjacent_loads)(fif_conversion)(fif_conversion2)(findirect_inlining)(finline)(finline_atomics)(finline_functions)(finline_functions_called_once)(finline_small_functions)(fipa_cp)(fipa_cp_alignment)(fipa_cp_clone)(fipa_icf)(fipa_icf_functions)(fipa_profile)(fipa_pta)(fipa_pure_const)(fipa_ra)(fipa_reference)(fipa_sra)(fira_hoist_pressure)(fira_loop_pressure)(fira_share_save_slots)(fira_share_spill_slots)(fisolate_erroneous_paths_attribute)(fisolate_erroneous_paths_dereference)(fivopts)(fjump_tables)(fkeep_gc_roots_live)(flifetime_dse)(flive_range_shrinkage)(floop_nest_optimize)(floop_parallelize_all)(flra_remat)(fmath_errno)(fmodulo_sched)(fmodulo_sched_allow_regmoves)(fmove_loop_invariants)(fnon_call_exceptions)(fnothrow_opt)(fomit_frame_pointer)(fopt_info)(foptimize_sibling_calls)(foptimize_strlen)(fpartial_inlining)(fpeel_loops)(fpeephole)(fpeephole2)(fplt)(fpredictive_commoning)(fprefetch_loop_arrays)(freciprocal_math)(freg_struct_return)(frename_registers)(freorder_blocks)(freorder_blocks_and_partition)(freorder_functions)(frerun_cse_after_loop)(freschedule_modulo_scheduled_loops)(frounding_math)(frtti)(fsched_critical_path_heuristic)(fsched_dep_count_heuristic)(fsched_group_heuristic)(fsched_interblock)(fsched_last_insn_heuristic)(fsched_pressure)(fsched_rank_heuristic)(fsched_spec)(fsched_spec_insn_heuristic)(fsched_spec_load)(fsched_spec_load_dangerous)(fsched_stalled_insns)(fsched_stalled_insns_dep)(fsched2_use_superblocks)(fschedule_fusion)(fschedule_insns)(fschedule_insns2)(fsel_sched_pipelining)(fsel_sched_pipelining_outer_loops)(fsel_sched_reschedule_pipelined)(fselective_scheduling)(fselective_scheduling2)(fshort_enums)(fshort_wchar)(fshrink_wrap)(fsignaling_nans)(fsigned_zeros)(fsingle_precision_constant)(fsplit_ivs_in_unroller)(fsplit_paths)(fsplit_wide_types)(fssa_backprop)(fssa_phiopt)(fstdarg_opt)(fstrict_aliasing)(fstrict_enums)(fstrict_overflow)(fstrict_volatile_bitfields)(fthread_jumps)(fno_threadsafe_statics)(ftracer)(ftrapping_math)(ftrapv)(ftree_bit_ccp)(ftree_builtin_call_dce)(ftree_ccp)(ftree_ch)(ftree_coalesce_vars)(ftree_copy_prop)(ftree_cselim)(ftree_dce)(ftree_dominator_opts)(ftree_dse)(ftree_forwprop)(ftree_fre)(ftree_loop_distribute_patterns)(ftree_loop_distribution)(ftree_loop_if_convert)(ftree_loop_if_convert_stores)(ftree_loop_im)(ftree_loop_ivcanon)(ftree_loop_optimize)(ftree_loop_vectorize)(ftree_lrs)(ftree_partial_pre)(ftree_phiprop)(ftree_pre)(ftree_pta)(ftree_reassoc)(ftree_scev_cprop)(ftree_sink)(ftree_slp_vectorize)(ftree_slsr)(ftree_sra)(ftree_switch_conversion)(ftree_tail_merge)(ftree_ter)(ftree_vectorize)(ftree_vrp)(funconstrained_commons)(funroll_all_loops)(funroll_loops)(funsafe_loop_optimizations)(funsafe_math_optimizations)(funswitch_loops)(funwind_tables)(fvar_tracking)(fvar_tracking_assignments)(fvar_tracking_assignments_toggle)(fvar_tracking_uninit)(fvariable_expansion_in_unroller)(fvpt)(fweb)(fwrapv)(cf);

}
#endif
