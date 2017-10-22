
R"(

// for single device testing
#define NUM_CHUNKS_1 1
#define NUM_CHUNKS_2 1



// WG ids
#define WG_ID_1 ( get_group_id(1) ) // TODO: should start from 1
#define WG_ID_2 ( get_group_id(0) )

// WI ids
#define WI_ID_1 ( get_local_id(1) )
#define WI_ID_2 ( get_local_id(0) )

//// P_wg partitioning
//#define NUM_WG_1 ( get_num_groups(1) )
//#define NUM_WG_2 ( get_num_groups(0) )
//
//// P_wi partitioning
//#define NUM_WI_1 ( get_local_size(1) )
//#define NUM_WI_2 ( get_local_size(0) )

// P_sq partitioning
#define NUM_SQ_1 ( CHUNK_PART_SIZE_1 / ( NUM_WG_1 * NUM_WI_1 ) ) // TODO: (WG_PART_SIZE_1 / NUM_WI_1)
#define NUM_SQ_2 ( CHUNK_PART_SIZE_2 / ( NUM_WG_2 * NUM_WI_2 ) )

// dev chunk size
#define CHUNK_PART_SIZE_1 ( N_1 / NUM_CHUNKS_1 )
#define CHUNK_PART_SIZE_2 ( N_2 / NUM_CHUNKS_2 )

// P_wg parition sizes
#define WG_PART_SIZE_1 ( CHUNK_PART_SIZE_1 / NUM_WG_1 )
#define WG_PART_SIZE_2 ( CHUNK_PART_SIZE_2 / NUM_WG_2 )

// P_sq parition sizes
#define SQ_PART_SIZE_1 ( WG_PART_SIZE_1 / NUM_SQ_1 )
#define SQ_PART_SIZE_2 ( WG_PART_SIZE_2 / NUM_SQ_2 )

// P_wi parition sizes
#define WI_PART_SIZE_1 ( SQ_PART_SIZE_1 / NUM_WI_1 ) //  == 1
#define WI_PART_SIZE_2 ( SQ_PART_SIZE_2 / NUM_WI_2 )

// P_wg offsets
#define WG_OFFSET_1 ( WG_PART_SIZE_1 * WG_ID_1 )
#define WG_OFFSET_2 ( WG_PART_SIZE_2 * WG_ID_2 )

// P_sq offsets
#define SQ_OFFSET_1 ( SQ_PART_SIZE_1 * ((i_sq)-1) )
#define SQ_OFFSET_2 ( SQ_PART_SIZE_2 * ((j_sq)-1) ) // == 0

// WI offsets
#define WI_OFFSET_1 ( WI_PART_SIZE_1 * WI_ID_1 )
#define WI_OFFSET_2 ( WI_PART_SIZE_2 * WI_ID_2 )


// reordering
//#define reorder(j) (  ( ((j)-1) / WI_PART_SIZE_2 ) + ( ((j)-1) % WI_PART_SIZE_2 ) * NUM_WI_2 + 1  )
//#define reorder(j) (j)//(  (WI_ID_2+1) + NUM_WI_2 * ((j)-1))

// matrix abstraction
#define matrix( i, j ) ( in_matrix[ (chunk_id_1 * CHUNK_PART_SIZE_1 + (i-1)) * N_2 + (chunk_id_2 * CHUNK_PART_SIZE_2 + (j-1)) ] )

// vector abstraction
#define vector( j ) ( in_vector[ (chunk_id_2 * CHUNK_PART_SIZE_2 + (j-1)) ] )

// view macro
#define view( i, j, c ) ( ( (c) == 0 ) ? matrix( i,j ) : vector( j ) )

// P_wg partitioning
#define my_p_wg( i, j, c ) view( WG_OFFSET_1 + (i) , WG_OFFSET_2 + (j) , (c) )

// P_sq partitioning
#define my_p_sq( i, j, c ) my_p_wg( SQ_OFFSET_1 + (i) , SQ_OFFSET_2 + (j) , (c) )

// P_wi partitioning
#define my_p_wi( i, j, c ) my_p_sq( WI_OFFSET_1 + (i) , WI_OFFSET_2 + (j) , (c) )

// results
#define my_res(i) out_vector[ ( WG_OFFSET_1 + SQ_OFFSET_1 + WI_OFFSET_1 + (i-1) ) * NUM_WG_2 + WG_ID_2 ]


// ######################################################
// kernel section

__kernel void func( __global float* in_matrix,
                      __global float* in_vector,
                      __global float* out_vector
//                               int    chunk_id_1,
//                               int    chunk_id_2
                    )
{

  int chunk_id_1 = 0;
  int chunk_id_2 = 0;

  // private memory for a WI's computation
  __private float res_prv = 0.0f;

  // local memory for a WG's computation
  __local   float res_lcl[ NUM_WI_1 ][ NUM_WI_2 ];

  // iteration over P_sq blocks
  for( size_t i_sq = 1 ; i_sq <= NUM_SQ_1 ; ++i_sq ) {
    res_prv = 0.0f;
    for( size_t j_sq = 1 ; j_sq <= NUM_SQ_2 ; ++j_sq ) {

      // sequential computation on a P_wi partition
      for( size_t i = 1 ; i <= WI_PART_SIZE_1 ; ++i )
        for( size_t j = 1 ; j <= WI_PART_SIZE_2 ; ++j )
          res_prv += my_p_wi( i, j, 0 ) * my_p_wi( i, j, 1 );
    } // end of for-loop j_sq

      // store result in local memory
      res_lcl[ WI_ID_1 ][ WI_ID_2 ] = res_prv;

      barrier( CLK_LOCAL_MEM_FENCE );

      // combine the WIs' results in dimension x
      for( size_t stride = NUM_WI_2 / 2 ; stride > 0 ; stride /= 2)
      {
        if( WI_ID_2 < stride)
          res_lcl[ WI_ID_1 ][ WI_ID_2 ] += res_lcl[ WI_ID_1 ][ WI_ID_2 + stride ];

        barrier( CLK_LOCAL_MEM_FENCE );
      }

      // store WGs' results in global memory
      if( WI_ID_2 == 0 )
        for( size_t i = 1 ; i <= WI_PART_SIZE_1 ; ++i ) { // it holds: WI_PART_SIZE_1 == 1
          my_res(i) = res_lcl[ WI_ID_1 ][0];
        }

      barrier( CLK_LOCAL_MEM_FENCE );

  } // end of for-loop i_sq
} // end of kernel




// input dimensions for second kernel
__kernel
void gemv_2( __global float* in_vector,
             __global float* out_vector,
                      int    chunk_id_1,
                      int    chunk_id_2
           )
{


size_t NUM_WG_2_KERNEL_1 = NUM_WG_2;

#undef  NUM_WG_2
#define NUM_WG_2 1

#undef  WI_PART_SIZE_2
#define WI_PART_SIZE_2 NUM_WG_2_KERNEL_1

size_t NUM_WI_2_hlpr = (WI_PART_SIZE_2 - NUM_WI_2 <= 0) ? WI_PART_SIZE_2 : NUM_WI_2; // TODO: min( WI_PART_SIZE_2, NUM_WI_2 );
#undef  NUM_WI_2
#define NUM_WI_2 NUM_WI_2_hlpr

#undef  my_res

// input
#define my_input( i, j ) in_vector[ ( WG_OFFSET_1 + SQ_OFFSET_1 + WI_OFFSET_1 + (i)-1 ) * WI_PART_SIZE_2 + (j)-1 ]

// results
#define my_res out_vector[ WG_OFFSET_1 + WI_OFFSET_1 + SQ_OFFSET_1 ]




//  for( size_t i = 0 ; i < N_1 * WI_PART_SIZE_2 ; ++i )
//   printf("in_vector[%i]=%f, N_1=%i, WI_PART_SIZE_2=%i, NUM_WG_2_KERNEL_1=%i\n", i, in_vector[i], N_1, WI_PART_SIZE_2, NUM_WG_2_KERNEL_1 );

  // private memory for a WI's computation
  __private float res_prv = 0.0f;

  // iteration over P_sq blocks
  for( size_t i_sq = 1 ; i_sq <= NUM_SQ_1 ; ++i_sq )
  {
    res_prv = 0.0f;

    // sequential computation on a P_wi partition
    for( size_t i = 1 ; i <= WI_PART_SIZE_1 ; ++i )
      for( size_t j = 1 ; j <= WI_PART_SIZE_2 ; ++j )
      {
        res_prv += my_input( i,j );
//        printf("my_input(%i,%i)=%f, i_sq=%i, id_1=%i, id_2=%i\n", i, j, my_input( i,j ), i_sq, get_global_id(0), get_global_id(1));
      }

    // store WGs' results in global memory
    if( WI_ID_2 == 0 )
    {
      my_res = res_prv;
//           printf("res_prv=%f, i_sq=%i\n", res_prv, i_sq );
    }

  } // end of for-loop i_sq
} // end of kernel

)"; // end of raw string
