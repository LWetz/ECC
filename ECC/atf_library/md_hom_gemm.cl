#if 1

R"(
#define NUM_WG_ITERATIONS_OVER_LM_BLOCKS_3 ( GM_SIZE_3 / (NUM_WG_3 * LM_SIZE_3) )
#define NUM_WG_ITERATIONS_OVER_LM_BLOCKS_2 ( GM_SIZE_2 / (NUM_WG_2 * LM_SIZE_2) )
#define NUM_WG_ITERATIONS_OVER_LM_BLOCKS_1 ( GM_SIZE_1 / (NUM_WG_1 * LM_SIZE_1) )

#define NUM_WI_ITERATIONS_OVER_PM_BLOCKS_3 ( LM_SIZE_3 / (NUM_WI_3 * PM_SIZE_3) )
#define NUM_WI_ITERATIONS_OVER_PM_BLOCKS_2 ( LM_SIZE_2 / (NUM_WI_2 * PM_SIZE_2) )
#define NUM_WI_ITERATIONS_OVER_PM_BLOCKS_1 ( LM_SIZE_1 / (NUM_WI_1 * PM_SIZE_1) )


// given by view
#define LM_SIZE_input_1_dim_1 LM_SIZE_3
#define LM_SIZE_input_1_dim_2 LM_SIZE_1

#define LM_SIZE_input_2_dim_1 LM_SIZE_2
#define LM_SIZE_input_2_dim_2 LM_SIZE_1


#define PM_SIZE_input_1_dim_1 PM_SIZE_3
#define PM_SIZE_input_1_dim_2 PM_SIZE_1

#define PM_SIZE_input_2_dim_1 PM_SIZE_2
#define PM_SIZE_input_2_dim_2 PM_SIZE_1



// helper
#define LM_BLOCK_SIZE_3 (LM_SIZE_3 * NUM_WG_3)
#define LM_BLOCK_SIZE_2 (LM_SIZE_2 * NUM_WG_2)
#define LM_BLOCK_SIZE_1 (LM_SIZE_1 * NUM_WG_1)

#define PM_BLOCK_SIZE_3 (PM_SIZE_3 * NUM_WI_LCL_3)
#define PM_BLOCK_SIZE_2 (PM_SIZE_2 * NUM_WI_LCL_2)
#define PM_BLOCK_SIZE_1 (PM_SIZE_1 * NUM_WI_LCL_1)


#define NUM_WI_GLB_3 (NUM_WG_3 * NUM_WI_LCL_3)
#define NUM_WI_GLB_2 (NUM_WG_2 * NUM_WI_LCL_2)
#define NUM_WI_GLB_1 (NUM_WG_1 * NUM_WI_LCL_1)

#define NUM_WI_LCL_3 NUM_WI_3
#define NUM_WI_LCL_2 NUM_WI_2
#define NUM_WI_LCL_1 NUM_WI_1

#define NUM_WI_PRV_3 1
#define NUM_WI_PRV_2 1
#define NUM_WI_PRV_1 1



inline void update_LM_cache( __global const float* restrict glb_input_1,       __global const float* restrict glb_input_2,
                             __local        float* restrict lcl_cache_input_1, __local        float* restrict lcl_cache_input_2,
                             const int WI_ID_LCL_3,   const int WI_ID_LCL_2,   const int WI_ID_LCL_1,
                             const int WI_ID_GLB_3,   const int WI_ID_GLB_2,   const int WI_ID_GLB_1,
                             const int id_lm_block_3, const int id_lm_block_2, const int id_lm_block_1
                           )
{
#if 1
 //  barrier( CLK_LOCAL_MEM_FENCE );

 // offsets to LM block
 const int OFFSET_3 = id_lm_block_3 * LM_BLOCK_SIZE_3;
 const int OFFSET_2 = id_lm_block_2 * LM_BLOCK_SIZE_2;
 const int OFFSET_1 = id_lm_block_1 * LM_BLOCK_SIZE_1;
 
// if( get_group_id(1) == 0 && get_local_id(2) == 1 )
// printf("\n\nOFFSET_3 = %i, OFFSET_2 = %i, OFFSET_1 = %i \n\n", OFFSET_3, OFFSET_2, OFFSET_1 );

 // flat ids
 const int WI_ID_LCL_FLAT = WI_ID_LCL_3 * (NUM_WI_LCL_2 * NUM_WI_LCL_1) + WI_ID_LCL_2 * NUM_WI_LCL_1 + WI_ID_LCL_1;
 const int WI_ID_GLB_FLAT = WI_ID_GLB_3 * (NUM_WI_GLB_2 * NUM_WI_GLB_1) + WI_ID_GLB_2 * NUM_WI_GLB_1 + WI_ID_GLB_1;
//if( get_global_id(2) == 1 )
//printf("\nWI_ID_LCL_3 = %i, NUM_WI_LCL_2 = %i, NUM_WI_LCL_1 = %i, WI_ID_LCL_2 = %i, NUM_WI_LCL_1 = %i, WI_ID_LCL_1 = %i\n\n",
//        WI_ID_LCL_3,      NUM_WI_LCL_2,      NUM_WI_LCL_1,      WI_ID_LCL_2,      NUM_WI_LCL_1,      WI_ID_LCL_1 );
  
 // cache first input
 { // new scope
   const int WI_ID_LCL_REORDER_1 =   WI_ID_LCL_FLAT %  NUM_WI_LCL_1;
   const int WI_ID_LCL_REORDER_3 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_1 * NUM_WI_LCL_3)                ) /  NUM_WI_LCL_1;
   const int WI_ID_LCL_REORDER_2 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_1 * NUM_WI_LCL_3 * NUM_WI_LCL_2) ) / (NUM_WI_LCL_1 * NUM_WI_LCL_3);

//if( get_global_id(2) == 1 )
//printf("WI_ID_LCL_FLAT = %i, WI_ID_LCL_REORDER_1 = %i, WI_ID_LCL_REORDER_2 = %i, WI_ID_LCL_REORDER_3 = %i, gs = %i\n", WI_ID_LCL_FLAT, WI_ID_LCL_REORDER_1, WI_ID_LCL_REORDER_2, WI_ID_LCL_REORDER_3, get_global_size(2) );

   const int WI_ID_GLB_REORDER_1 =   WI_ID_GLB_FLAT %  NUM_WI_GLB_1;
   const int WI_ID_GLB_REORDER_3 = ( WI_ID_GLB_FLAT % (NUM_WI_GLB_1 * NUM_WI_GLB_3)                ) /  NUM_WI_GLB_1;
   const int WI_ID_GLB_REORDER_2 = ( WI_ID_GLB_FLAT % (NUM_WI_GLB_1 * NUM_WI_GLB_3 * NUM_WI_GLB_2) ) / (NUM_WI_GLB_1 * NUM_WI_GLB_3);

//if( get_global_id(0) == 1 )
//{
//  printf("WI_ID_LCL_REORDER_1 = %i, WI_ID_LCL_REORDER_2 = %i, WI_ID_LCL_REORDER_3 = %i\n", WI_ID_LCL_REORDER_1, WI_ID_LCL_REORDER_2, WI_ID_LCL_REORDER_3);
//  printf("WI_ID_GLB_REORDER_1 = %i, WI_ID_GLB_REORDER_2 = %i, WI_ID_GLB_REORDER_3 = %i\n", WI_ID_GLB_REORDER_1, WI_ID_GLB_REORDER_2, WI_ID_GLB_REORDER_3);
//}

#if 0
   if( WI_ID_LCL_REORDER_2 == 0 )
     for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
       for( int i_1 = 0 ; i_1 < LM_SIZE_1 / NUM_WI_LCL_1 ; ++i_1 )
       {
         const int lcl_index_3 = WI_ID_LCL_REORDER_3 + i_3 * NUM_WI_LCL_3;
         const int lcl_index_1 = WI_ID_LCL_REORDER_1 + i_1 * NUM_WI_LCL_1;

         const int glb_index_3 = WI_ID_GLB_REORDER_3 + i_3 * NUM_WI_GLB_3;
         const int glb_index_1 = WI_ID_GLB_REORDER_1 + i_1 * NUM_WI_GLB_1;
#else
   if( WI_ID_LCL_2 == 0 )
     for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
       for( int i_1 = 0 ; i_1 < LM_SIZE_1 / NUM_WI_LCL_1 ; ++i_1 )
       {
         const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
         const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;

         const int glb_index_3 = WI_ID_GLB_3 + i_3 * NUM_WI_GLB_3;
         const int glb_index_1 = WI_ID_GLB_1 + i_1 * NUM_WI_GLB_1;
#endif
//if( get_group_id(1) == 0 && get_local_id(2) == 1 )
//printf("glb_index_3 = %i, glb_index_1 = %i\n", (OFFSET_3 + glb_index_3), (OFFSET_1 + glb_index_1) );

         lcl_cache_input_1[ lcl_index_3 * LM_SIZE_input_1_dim_2 + lcl_index_1 ] =
         glb_input_1[ (OFFSET_3 + glb_index_3) * GM_SIZE_1 + (OFFSET_1 + glb_index_1) ];
// if( WI_ID_LCL_3 == 0 )
//{
//  printf("glb_input_1[ %i ] = %f, glb_index_3 = %i, glb_index_1 = %i\n", (OFFSET_3 + glb_index_3) * GM_SIZE_1 + (OFFSET_1 + glb_index_1), glb_input_1[ (OFFSET_3 + glb_index_3) * GM_SIZE_1 + (OFFSET_1 + glb_index_1) ], WI_ID_GLB_REORDER_3 + i_3 * NUM_WI_GLB_3, WI_ID_GLB_REORDER_1 + i_1 * NUM_WI_GLB_1 );
//  printf("  lcl_cache_input_1[ %i ] = %f, lcl_index_3 = %i, lcl_index_1 = %i\n", lcl_index_3 * LM_SIZE_input_1_dim_2 + lcl_index_1, lcl_cache_input_1[ lcl_index_3 * LM_SIZE_input_1_dim_2 + lcl_index_1 ], lcl_index_3, lcl_index_1 );
//}
       }
 } // end of "new scope"
 
// if( get_group_id(1) == 0 && get_local_id(2) == 1 )
//printf("\nEND\n");
 // cache second input
 { // new scope
   // leading dims = 2->1->3
   const int WI_ID_LCL_REORDER_1 =   WI_ID_LCL_FLAT %  NUM_WI_LCL_1;
   const int WI_ID_LCL_REORDER_2 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_1 * NUM_WI_LCL_2)                ) /  NUM_WI_LCL_1;
   const int WI_ID_LCL_REORDER_3 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_1 * NUM_WI_LCL_2 * NUM_WI_LCL_3) ) / (NUM_WI_LCL_1 * NUM_WI_LCL_2);

   const int WI_ID_GLB_REORDER_1 =   WI_ID_GLB_FLAT %  NUM_WI_GLB_1;
   const int WI_ID_GLB_REORDER_2 = ( WI_ID_GLB_FLAT % (NUM_WI_GLB_1 * NUM_WI_GLB_2)                ) /  NUM_WI_GLB_1;
   const int WI_ID_GLB_REORDER_3 = ( WI_ID_GLB_FLAT % (NUM_WI_GLB_1 * NUM_WI_GLB_2 * NUM_WI_GLB_3) ) / (NUM_WI_GLB_1 * NUM_WI_GLB_2);

//if( get_global_id(0) == 1 )
//{
//  printf("WI_ID_LCL_REORDER_1 = %i, WI_ID_LCL_REORDER_2 = %i, WI_ID_LCL_REORDER_3 = %i\n", WI_ID_LCL_REORDER_1, WI_ID_LCL_REORDER_2, WI_ID_LCL_REORDER_3);
//  printf("WI_ID_GLB_REORDER_1 = %i, WI_ID_GLB_REORDER_2 = %i, WI_ID_GLB_REORDER_3 = %i\n", WI_ID_GLB_REORDER_1, WI_ID_GLB_REORDER_2, WI_ID_GLB_REORDER_3);
//}


#if 0
   if( WI_ID_LCL_3 == 0 )
     //#pragma unroll
     for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
       //#pragma unroll
       for( int i_1 = 0 ; i_1 < LM_SIZE_1 / NUM_WI_LCL_1 ; ++i_1 )
       {
         const int lcl_index_2 = WI_ID_LCL_REORDER_2 + i_2 * NUM_WI_LCL_2;
         const int lcl_index_1 = WI_ID_LCL_REORDER_1 + i_1 * NUM_WI_LCL_1;

         const int glb_index_2 = WI_ID_GLB_REORDER_2 + i_2 * NUM_WI_GLB_2;
         const int glb_index_1 = WI_ID_GLB_REORDER_1 + i_1 * NUM_WI_GLB_1;
#else
   if( WI_ID_LCL_3 == 0 )
     //#pragma unroll
     for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
       //#pragma unroll
       for( int i_1 = 0 ; i_1 < LM_SIZE_1 / NUM_WI_LCL_1 ; ++i_1 )
       {
         const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;
         const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;

         const int glb_index_2 = WI_ID_GLB_2 + i_2 * NUM_WI_GLB_2;
         const int glb_index_1 = WI_ID_GLB_1 + i_1 * NUM_WI_GLB_1;
#endif
         lcl_cache_input_2[ lcl_index_2 * LM_SIZE_input_2_dim_2 + lcl_index_1 ] =
         glb_input_2[ (OFFSET_2 + glb_index_2) * GM_SIZE_1 + (OFFSET_1 + glb_index_1) ];
//printf("glb_input_2[ %i ] = %f, glb_index_2 = %i, glb_index_1 = %i\n", (OFFSET_2 + glb_index_2) * GM_SIZE_1 + (OFFSET_1 + glb_index_1), glb_input_2[ (OFFSET_2 + glb_index_2) * GM_SIZE_1 + (OFFSET_1 + glb_index_1) ], WI_ID_GLB_REORDER_2 + i_2 * NUM_WI_GLB_2, WI_ID_GLB_REORDER_1 + i_1 * NUM_WI_GLB_1 );
       }
 } // end of "new scope"
 
// barrier( CLK_LOCAL_MEM_FENCE );
// if( get_global_id(0) == 0 )
// for( int i_3 = 0 ; i_3 < 4 * 4 ; ++i_3 )
//   printf("lcl_cache_input_2[ %i ] = %f\n", i_3, lcl_cache_input_2[ i_3 ] );
 
 
#else
 for( int i_3 = 0 ; i_3 < (LM_SIZE_input_1_dim_1 * LM_SIZE_input_1_dim_2) / (NUM_WI_LCL_3 * NUM_WI_LCL_2 * NUM_WI_LCL_1) ; ++i_3 )
   lcl_cache_input_1[get_local_id(0)] = glb_input_1[ get_global_id(0) ];

 for( int i_3 = 0 ; i_3 < (LM_SIZE_input_2_dim_1 * LM_SIZE_input_2_dim_2) / (NUM_WI_LCL_3*NUM_WI_LCL_2*NUM_WI_LCL_1) ; ++i_3 )
   lcl_cache_input_2[get_local_id(0)] = glb_input_2[ get_global_id(0) ];

// for( int i_3 = 0 ; i_3 < 4*4 ; ++i_3 )
//   lcl_cache_input_1[ i_3] = i_3;
//
// for( int i_3 = 0 ; i_3 < 4*4 ; ++i_3 )
//   lcl_cache_input_2[ i_3 ] = i_3;
#endif

#if 0
 for( int i = 0 ; i < LM_SIZE_input_1_dim_1 * LM_SIZE_input_1_dim_2 ; ++i )
   lcl_cache_input_1[ i ] = 1;

 for( int i = 0 ; i < LM_SIZE_input_2_dim_1 * LM_SIZE_input_2_dim_2 ; ++i )
   lcl_cache_input_2[ i ] = 1;
#endif

//if( get_group_id(1) == 0 && get_local_id(2) == 1 )
// for( int i = 0 ; i < LM_SIZE_input_1_dim_1 * LM_SIZE_input_1_dim_2 ; ++i )
//   printf("lcl_cache_input_1[ %i ] = %f\n", i, lcl_cache_input_1[ i ] );
//
//if( get_group_id(1) == 0 && get_local_id(2) == 1  )
// for( int i = 0 ; i < LM_SIZE_input_2_dim_1 * LM_SIZE_input_2_dim_2 ; ++i )
//   printf("lcl_cache_input_2[ %i ] = %f\n", i ,lcl_cache_input_2[ i ] );
//
//if( get_group_id(1) == 0 && get_local_id(2) == 1  )
//   printf("\n\n");


}


inline void update_PM_cache( __local   const float* restrict lcl_cache_input_1, __local   const float* restrict lcl_cache_input_2,
                           __private       float* restrict prv_cache_input_1, __private       float* restrict prv_cache_input_2,
//                           const int WI_ID_PRV_3,   const int WI_ID_PRV_2,   const int WI_ID_PRV_1,
                           const int WI_ID_LCL_3,   const int WI_ID_LCL_2,   const int WI_ID_LCL_1,
//                           const int WI_ID_GLB_3,   const int WI_ID_GLB_2,   const int WI_ID_GLB_1,
                           const int id_pm_block_3, const int id_pm_block_2, const int id_pm_block_1
                           )

{

#if 1
 // offsets to LM block
 const int OFFSET_3 = id_pm_block_3 * PM_BLOCK_SIZE_3;
 const int OFFSET_2 = id_pm_block_2 * PM_BLOCK_SIZE_2;
 const int OFFSET_1 = id_pm_block_1 * PM_BLOCK_SIZE_1;
 
 
 const int WI_ID_PRV_3 = 0;
 const int WI_ID_PRV_2 = 0;
 const int WI_ID_PRV_1 = 0;

//  const int WI_ID_GLB_FLAT = WI_ID_GLB_3 * (NUM_WI_GLB_2 * NUM_WI_GLB_1) + WI_ID_GLB_2 * NUM_WI_GLB_1 + WI_ID_GLB_1;
//  const int WI_ID_LCL_FLAT = WI_ID_GLB_FLAT % NUM_WI_LCL;
//  const int WI_ID_PRV_FLAT = WI_ID_LCL_FLAT % NUM_WI_PRV;

 // cache first input
 { // new scope

   // leading dimension order: 1->3->2
//    const int WI_ID_PRV_REORDER_1 =   WI_ID_PRV_FLAT %  NUM_WI_PRV_1;
//    const int WI_ID_PRV_REORDER_3 = ( WI_ID_PRV_FLAT % (NUM_WI_PRV_1 * NUM_WI_PRV_3)              ) /  NUM_WI_PRV_1;
//    const int WI_ID_PRV_REORDER_2 = ( WI_ID_PRV_FLAT % (NUM_WI_PRV_1 * NUM_WI_PRV_3 * NUM_WI_PRV_2) ) / (NUM_WI_PRV_1 * NUM_WI_PRV_3);
//    
//    const int WI_ID_LCL_REORDER_1 =   WI_ID_LCL_FLAT %  NUM_WI_LCL_1;
//    const int WI_ID_LCL_REORDER_3 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_1 * NUM_WI_LCL_3)              ) /  NUM_WI_LCL_1;
//    const int WI_ID_LCL_REORDER_2 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_1 * NUM_WI_LCL_3 * NUM_WI_LCL_2) ) / (NUM_WI_LCL_1 * NUM_WI_LCL_3);


   //if( WI_ID_PRV_2 == 0 )
     //#pragma unroll
     for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
       //#pragma unroll
       for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
       {
         const int prv_index_3 = WI_ID_PRV_3 + i_3 * NUM_WI_PRV_3;
         const int prv_index_1 = WI_ID_PRV_1 + i_1 * NUM_WI_PRV_1;

         const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
         const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;

         prv_cache_input_1[ prv_index_3 * PM_SIZE_input_1_dim_2 + prv_index_1 ] =
         lcl_cache_input_1[ (OFFSET_3 + lcl_index_3) * LM_SIZE_input_1_dim_2 + (OFFSET_1 + lcl_index_1) ];
//printf("%i, lcl_cache_input_1 = %f\n", prv_index_3 * PM_SIZE_input_1_dim_2 + prv_index_1, lcl_cache_input_1[ (OFFSET_3 + lcl_index_3) * LM_SIZE_input_1_dim_2 + (OFFSET_1 + lcl_index_1) ] );
       }
 }

 // cache second input
 { // new scope
   //  const int WI_ID_PRV_FLAT = WI_ID_PRV_3 * (NUM_WI_PRV_2 + NUM_WI_PRV_1) +  + WI_ID_PRV_2 * NUM_WI_PRV_1 + WI_ID_PRV_1;
   //  const int WI_ID_LCL_FLAT = WI_ID_LCL_3 * (NUM_WI_LCL_2 + NUM_WI_LCL_1) +  + WI_ID_LCL_2 * NUM_WI_LCL_1 + WI_ID_LCL_1;
   //  //leading dim = 2->1->3
   //  const int WI_ID_PRV_REORDER_2 =   WI_ID_PRV_FLAT %  NUM_WI_PRV_2;
   //  const int WI_ID_PRV_REORDER_1 = ( WI_ID_PRV_FLAT % (NUM_WI_PRV_2 * NUM_WI_PRV_1)             ) /  NUM_WI_PRV_2;
   //  const int WI_ID_PRV_REORDER_3 = ( WI_ID_PRV_FLAT % (NUM_WI_PRV_2 * NUM_WI_PRV_1 * NUM_WI_PRV_3) ) / (NUM_WI_PRV_2 * NUM_WI_PRV_1);
   //
   //  const int WI_ID_LCL_REORDER_2 =   WI_ID_LCL_FLAT %  NUM_WI_LCL_2;
   //  const int WI_ID_LCL_REORDER_1 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_2 * NUM_WI_LCL_1)             ) /  NUM_WI_LCL_2;
   //  const int WI_ID_LCL_REORDER_3 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_2 * NUM_WI_LCL_1 * NUM_WI_LCL_3) ) / (NUM_WI_LCL_2 * NUM_WI_LCL_1);

   // leading dimension order: 2->1->3
//    const int WI_ID_PRV_REORDER_2 =   WI_ID_PRV_FLAT %  NUM_WI_PRV_2;
//    const int WI_ID_PRV_REORDER_1 = ( WI_ID_PRV_FLAT % (NUM_WI_PRV_2 * NUM_WI_PRV_1)              ) /  NUM_WI_PRV_2;
//    const int WI_ID_PRV_REORDER_3 = ( WI_ID_PRV_FLAT % (NUM_WI_PRV_2 * NUM_WI_PRV_1 * NUM_WI_PRV_3) ) / (NUM_WI_PRV_2 * NUM_WI_PRV_1);
//
//    const int WI_ID_LCL_REORDER_2 =   WI_ID_LCL_FLAT %  NUM_WI_LCL_2;
//    const int WI_ID_LCL_REORDER_1 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_2 * NUM_WI_LCL_1)              ) /  NUM_WI_LCL_2;
//    const int WI_ID_LCL_REORDER_3 = ( WI_ID_LCL_FLAT % (NUM_WI_LCL_2 * NUM_WI_LCL_1 * NUM_WI_LCL_3) ) / (NUM_WI_LCL_2 * NUM_WI_LCL_1);

   //if( WI_ID_PRV_3 == 0 )
     //#pragma unroll
     for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
       //#pragma unroll
       for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
       {
//printf(" i_2 = %i, i_1 = %i\n", i_2, i_1 );
         const int prv_index_2 = WI_ID_PRV_2 + i_2 * NUM_WI_PRV_2;
         const int prv_index_1 = WI_ID_PRV_1 + i_1 * NUM_WI_PRV_1;
//printf(" prv_index_2 = %i, prv_index_1 = %i\n", prv_index_2, prv_index_1 );

         const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;
         const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;

         prv_cache_input_2[ prv_index_2 * PM_SIZE_input_2_dim_2 + prv_index_1 ] =
         lcl_cache_input_2[ (OFFSET_2 + lcl_index_2) * LM_SIZE_input_2_dim_2 + (OFFSET_1 + lcl_index_1) ];
//printf("%i, lcl_cache_input_2 = %f\n", prv_index_2 * PM_SIZE_input_1_dim_1 + prv_index_1, lcl_cache_input_2[ (OFFSET_2 + lcl_index_2) * LM_SIZE_input_1_dim_1 + (OFFSET_1 + lcl_index_1) ] );
       }
 }
#else
 for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
   for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
     for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
       prv_cache_input_1[0] = lcl_cache_input_1[ get_local_id(0) ];

 for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
   for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
     for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
       prv_cache_input_2[0] = lcl_cache_input_2[ get_local_id(0) ];

#endif
#if 0
 //#pragma unroll
 for( int i = 0 ; i < PM_SIZE_3 * PM_SIZE_1 ; ++i )
   prv_cache_input_1[ i ] = 1;
 //#pragma unroll
 for( int i = 0 ; i < PM_SIZE_1 * PM_SIZE_2 ; ++i )
   prv_cache_input_2[ i ] = 1;
#endif

// for( int i = 0 ; i < PM_SIZE_input_1_dim_1 * PM_SIZE_input_1_dim_2 ; ++i )
//   printf("prv_cache_input_1[ %i ] = %f\n", i, prv_cache_input_1[ i ] );
//
// for( int i = 0 ; i < PM_SIZE_input_2_dim_1 * PM_SIZE_input_2_dim_2 ; ++i )
//   printf("prv_cache_input_2[ %i ] = %f\n", i, prv_cache_input_2[ i ] );
//  
//   printf("\n\n");

}


// view
struct pair
{
 float const* lhs;
 float const* rhs;
};


inline const struct pair view( __private const float* restrict in_1,
                             __private const float* restrict in_2,
                             const int i_3,
                             const int i_2,
                             const int i_1
                             )
{
 struct pair p;

 p.lhs = &in_1[ i_3 * PM_SIZE_input_1_dim_2 + i_1 ];
 p.rhs = &in_2[ i_2 * PM_SIZE_input_2_dim_2 + i_1 ];

 return p;
}

// user function (TODO pre-user function: add, ... , stencils, etc. )
inline float f( const struct pair p )
{
 return *(p.lhs) * *(p.rhs);
}


__kernel void gemm_1( __global const float* restrict glb_input_1,
                    __global const float* restrict glb_input_2,
                    __global       float* restrict glb_output
                    )
{
#if 1
 for( size_t i = 0 ; i  < GM_SIZE_3 * GM_SIZE_2 ; ++i )
   glb_output[ i ]  =  0;
#endif

 // cache memory
 __local float lcl_cache_input_1[ LM_SIZE_input_1_dim_1 ][ LM_SIZE_input_1_dim_2 ];
 __local float lcl_cache_input_2[ LM_SIZE_input_2_dim_1 ][ LM_SIZE_input_2_dim_2 ];

 __private float prv_cache_input_1[ PM_SIZE_input_1_dim_1 ][ PM_SIZE_input_1_dim_2 ];
 __private float prv_cache_input_2[ PM_SIZE_input_2_dim_1 ][ PM_SIZE_input_2_dim_2 ];

 // memory for intermediate results
 __private float tmp_res_prv[ PM_SIZE_3 ][ PM_SIZE_2 ]; // for schematical implementation -> [ 1 * PM_SIZE_3 ][ 1 * PM_SIZE_2 ][ 1 ]
 __local   float tmp_res_lcl[ LM_SIZE_3 ][ LM_SIZE_2 ][ NUM_WI_LCL_1 ];

 // WG/WI ids
 const int WG_ID_3 = get_group_id( 2 );
 const int WG_ID_2 = get_group_id( 1 );
 const int WG_ID_1 = get_group_id( 0 );

 const int WI_ID_GLB_3 = get_global_id( 2 );
 const int WI_ID_GLB_2 = get_global_id( 1 );
 const int WI_ID_GLB_1 = get_global_id( 0 );

 const int WI_ID_LCL_3 = get_local_id( 2 );
 const int WI_ID_LCL_2 = get_local_id( 1 );
 const int WI_ID_LCL_1 = get_local_id( 0 );

 const int WI_ID_PRV_3 = 0;
 const int WI_ID_PRV_2 = 0;
 const int WI_ID_PRV_1 = 0;

 // clean tmp_res_prv
 //#pragma unroll
 for( int i_3 = 0 ; i_3 < PM_SIZE_3 / NUM_WI_PRV_3 ; ++i_3 )
   //#pragma unroll
   for( int i_2 = 0 ; i_2 < PM_SIZE_2 / NUM_WI_PRV_2 ; ++i_2 )
   {
     const int prv_index_3 = i_3;
     const int prv_index_2 = i_2;

     tmp_res_prv[ prv_index_3 ][ prv_index_2 ] = 0;
   }

 // clean tmp_res_lcl
 //#pragma unroll
 for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
   //#pragma unroll
   for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
   {
     const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
     const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

     tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 ] = 0;
   }
 barrier( CLK_LOCAL_MEM_FENCE );

 // iteration over local memory blocks
 //#pragma unroll
 for( int id_lm_block_3 = 0 ; id_lm_block_3 < NUM_WG_ITERATIONS_OVER_LM_BLOCKS_3 ; ++id_lm_block_3 )
   //#pragma unroll
   for( int id_lm_block_2 = 0 ; id_lm_block_2 < NUM_WG_ITERATIONS_OVER_LM_BLOCKS_2 ; ++id_lm_block_2 )
   {
     //#pragma unroll
     for( int id_lm_block_1 = 0 ; id_lm_block_1 < NUM_WG_ITERATIONS_OVER_LM_BLOCKS_1 ; ++id_lm_block_1 )
     {
       update_LM_cache( glb_input_1, glb_input_2,
                       lcl_cache_input_1, lcl_cache_input_2,
                       WI_ID_LCL_3, WI_ID_LCL_2, WI_ID_LCL_1,
                       WI_ID_GLB_3, WI_ID_GLB_2, WI_ID_GLB_1,
                       id_lm_block_3, id_lm_block_2, id_lm_block_1
                       );
       barrier( CLK_LOCAL_MEM_FENCE );

       // iteration over private memory blocks
       //#pragma unroll
       for( int id_pm_block_3 = 0 ; id_pm_block_3 < NUM_WI_ITERATIONS_OVER_PM_BLOCKS_3 ; ++id_pm_block_3 )
         //#pragma unroll
         for( int id_pm_block_2 = 0 ; id_pm_block_2 < NUM_WI_ITERATIONS_OVER_PM_BLOCKS_2 ; ++id_pm_block_2 )
         {
           //#pragma unroll
           for( int id_pm_block_1 = 0 ; id_pm_block_1 < NUM_WI_ITERATIONS_OVER_PM_BLOCKS_1 ; ++id_pm_block_1 )
           {
             update_PM_cache( lcl_cache_input_1, lcl_cache_input_2,
                             prv_cache_input_1, prv_cache_input_2,
                             //WI_ID_PRV_3, WI_ID_PRV_2, WI_ID_PRV_1,
                             WI_ID_LCL_3, WI_ID_LCL_2, WI_ID_LCL_1,
                             id_pm_block_3, id_pm_block_2, id_pm_block_1
                             );

             // computation of a WI
             //#pragma unroll
             for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
               //#pragma unroll
               for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
                 //#pragma unroll
                 for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
{
                   //                    tmp_res_prv[ i_3 ][ i_2 ] += f( view( prv_cache_input_1, prv_cache_input_2, i_3, i_2, i_1 ) );
                   tmp_res_prv[ i_3 ][ i_2 ] += prv_cache_input_1[ i_3 ][ i_1 ] * prv_cache_input_2[ i_2 ][ i_1 ];
//printf("prv_cache_input_1[ i_3 ][ i_1 ] = %f,  prv_cache_input_2[ i_1 ][ i_2 ]  = %f \n", prv_cache_input_1[ i_3 ][ i_1 ], prv_cache_input_2[ i_1 ][ i_2 ] );
}
           } // end for-loop "i_pm_1"

           // i) copy intermediate results from private to local memory, and ii) clean "tmp_res_prv"
           //#pragma unroll
           for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
             //#pragma unroll
             for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
             {
               const int OFFSET_3 = id_pm_block_3 * PM_BLOCK_SIZE_3;
               const int OFFSET_2 = id_pm_block_2 * PM_BLOCK_SIZE_2;

               const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
               const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

               const int prv_index_3 = i_3;
               const int prv_index_2 = i_2;

               tmp_res_lcl[ OFFSET_3 + lcl_index_3 ][ OFFSET_2 + lcl_index_2 ][ WI_ID_LCL_1 ] += tmp_res_prv[ prv_index_3 ][ prv_index_2 ];
               tmp_res_prv[ prv_index_3 ][ prv_index_2 ] = 0;
             }

           //barrier( CLK_LOCAL_MEM_FENCE );

         } // end for-loop "i_pm_2"
     } // end for-loop "id_lm_block_1"

     // parallel reduction of intermediate results
     //#pragma unroll
     for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 ) //+= NUM_WI_LCL_3 )
       //#pragma unroll
       for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 ) //+= NUM_WI_LCL_2 )
         //#pragma unroll
         for( int stride = NUM_WI_LCL_1 / 2 ; stride > 0 ; stride /= 2 )
         {
           const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
           const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

           if( WI_ID_LCL_1 < stride )
             tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 ] += tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 + stride ];

           barrier( CLK_LOCAL_MEM_FENCE );
         }

     // i) copy intermediate results from local memory to global memory, and ii) clean "tmp_res_lcl"
     //barrier( CLK_GLOBAL_MEM_FENCE );
     //#pragma unroll
     for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
       //#pragma unroll
       for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
       {
         const int OFFSET_3 = id_lm_block_3 * LM_BLOCK_SIZE_3;
         const int OFFSET_2 = id_lm_block_2 * LM_BLOCK_SIZE_2;

         const int glb_index_3 = WI_ID_GLB_3 + i_3 * NUM_WI_GLB_3;
         const int glb_index_2 = WI_ID_GLB_2 + i_2 * NUM_WI_GLB_2;

         const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
         const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;


         if( WI_ID_LCL_1 == 0 )
           glb_output[ (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
                      (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
                      WG_ID_1                                             // dim 3
                      ] += tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ 0 ];

         //  if( WI_ID_LCL_1 == 0 && WI_ID_LCL_2 == 0 )
         //  {
         //  printf("y = %i, x = %i, k = %i, WI_ID = %i\n\n", (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1), (OFFSET_2 + glb_index_2) * NUM_WG_1, WG_ID_1, WI_ID_LCL_1 );
         //printf("index = %i, OFFSET_3 = %i, glb_index_3 = %i, GM_SIZE_2 = %i, NUM_WG_1 = %i, OFFSET_2 = %i, id_lm_block_2 = %i, LM_BLOCK_SIZE_2 = %i,  glb_index_2 = %i, NUM_WG_1 = %i, WG_ID_1 = %i, res = %f\n",
         //(OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
         //                        (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
         //                        WG_ID_1,
         //OFFSET_3, glb_index_3, GM_SIZE_2, NUM_WG_1, OFFSET_2, id_lm_block_2, LM_BLOCK_SIZE_2, glb_index_2, NUM_WG_1, WG_ID_1,
         //glb_output[ (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
         //                        (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
         //                        WG_ID_1                                             // dim 3
         //                      ]
         //);
         //}


         //printf("res = %f\n",             glb_output[ (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
         //                      (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
         //                      WG_ID_1                                             // dim 3
         //                    ] );

         // clean "tmp_res_lcl"
         tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 ] = 0;
       } // end for-loop "i_2"

     barrier( CLK_LOCAL_MEM_FENCE );

   } // end for-loop "i_lm_2"
}


__kernel void gemm_2( __global float* output_1,
                    __global float* output_2
                    )
{
}

)";


#else

R"(
#define NUM_WG_ITERATIONS_OVER_LM_BLOCKS_3 ( GM_SIZE_3 / (NUM_WG_3 * LM_SIZE_3) )
#define NUM_WG_ITERATIONS_OVER_LM_BLOCKS_2 ( GM_SIZE_2 / (NUM_WG_2 * LM_SIZE_2) )
#define NUM_WG_ITERATIONS_OVER_LM_BLOCKS_1 ( GM_SIZE_1 / (NUM_WG_1 * LM_SIZE_1) )

#define NUM_WI_ITERATIONS_OVER_PM_BLOCKS_3 ( LM_SIZE_3 / (NUM_WI_3 * PM_SIZE_3) )
#define NUM_WI_ITERATIONS_OVER_PM_BLOCKS_2 ( LM_SIZE_2 / (NUM_WI_2 * PM_SIZE_2) )
#define NUM_WI_ITERATIONS_OVER_PM_BLOCKS_1 ( LM_SIZE_1 / (NUM_WI_1 * PM_SIZE_1) )


// given by view
#define LM_SIZE_input_1_dim_1 LM_SIZE_3
#define LM_SIZE_input_1_dim_2 LM_SIZE_1

#define LM_SIZE_input_2_dim_1 LM_SIZE_1
#define LM_SIZE_input_2_dim_2 LM_SIZE_2


#define PM_SIZE_input_1_dim_1 PM_SIZE_3
#define PM_SIZE_input_1_dim_2 PM_SIZE_1

#define PM_SIZE_input_2_dim_1 PM_SIZE_1
#define PM_SIZE_input_2_dim_2 PM_SIZE_2



// helper
#define LM_BLOCK_SIZE_3 (LM_SIZE_3 * NUM_WG_3)
#define LM_BLOCK_SIZE_2 (LM_SIZE_2 * NUM_WG_2)
#define LM_BLOCK_SIZE_1 (LM_SIZE_1 * NUM_WG_1)

#define PM_BLOCK_SIZE_3 (PM_SIZE_3 * NUM_WI_LCL_3)
#define PM_BLOCK_SIZE_2 (PM_SIZE_2 * NUM_WI_LCL_2)
#define PM_BLOCK_SIZE_1 (PM_SIZE_1 * NUM_WI_LCL_1)


#define NUM_WI_GLB_3 (NUM_WG_3 * NUM_WI_LCL_3)
#define NUM_WI_GLB_2 (NUM_WG_2 * NUM_WI_LCL_2)
#define NUM_WI_GLB_1 (NUM_WG_1 * NUM_WI_LCL_1)

#define NUM_WI_LCL_3 NUM_WI_3
#define NUM_WI_LCL_2 NUM_WI_2
#define NUM_WI_LCL_1 NUM_WI_1

#define NUM_WI_PRV_3 1
#define NUM_WI_PRV_2 1
#define NUM_WI_PRV_1 1



inline void update_LM_cache( __global const float* restrict glb_input_1,       __global const float* restrict glb_input_2,
                           __local        float* restrict lcl_cache_input_1, __local        float* restrict lcl_cache_input_2,
                           const int WI_ID_LCL_3, const int WI_ID_LCL_2, const int WI_ID_LCL_1,
                           const int WI_ID_GLB_3, const int WI_ID_GLB_2, const int WI_ID_GLB_1,
                           const int id_lm_block_3,     const int id_lm_block_2,     const int id_lm_block_1
                           )
{
#if 1
 barrier( CLK_LOCAL_MEM_FENCE );

 // offsets to LM block
 const int OFFSET_3 = id_lm_block_3 * LM_BLOCK_SIZE_3;
 const int OFFSET_2 = id_lm_block_2 * LM_BLOCK_SIZE_2;
 const int OFFSET_1 = id_lm_block_1 * LM_BLOCK_SIZE_1;


 // cache first input
 for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
   for( int i_1 = 0 ; i_1 < LM_SIZE_1 / NUM_WI_LCL_1 ; ++i_1 )
   {
     const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
     const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;

     const int glb_index_3 = WI_ID_GLB_3 + i_3 * NUM_WI_GLB_3;
     const int glb_index_1 = WI_ID_GLB_1 + i_1 * NUM_WI_GLB_1;

     lcl_cache_input_1[ lcl_index_3 * LM_SIZE_input_1_dim_2 + lcl_index_1 ] =
     glb_input_1[ (OFFSET_3 + glb_index_3) * GM_SIZE_1 +(OFFSET_1 + glb_index_1) ];
   }

 // cache second input
 //#pragma unroll
 for( int i_1 = 0 ; i_1 < LM_SIZE_1 / NUM_WI_LCL_1 ; ++i_1 )
   //#pragma unroll
   for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
   {
     const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;
     const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

     const int glb_index_1 = WI_ID_GLB_1 + i_1 * NUM_WI_GLB_1;
     const int glb_index_2 = WI_ID_GLB_2 + i_2 * NUM_WI_GLB_2;

     lcl_cache_input_2[ lcl_index_1 * LM_SIZE_input_2_dim_2 + lcl_index_2 ] =
     glb_input_2[ (OFFSET_1 + glb_index_1) * GM_SIZE_2 + (OFFSET_2 + glb_index_2) ];
   }

 barrier( CLK_LOCAL_MEM_FENCE );
#endif
}


inline void update_PM_cache( __local   const float* restrict lcl_cache_input_1, __local   const float* restrict lcl_cache_input_2,
                           __private       float* restrict prv_cache_input_1, __private       float* restrict prv_cache_input_2,
                           const int WI_ID_PRV_3, const int WI_ID_PRV_2, const int WI_ID_PRV_1,
                           const int WI_ID_LCL_3, const int WI_ID_LCL_2, const int WI_ID_LCL_1,
                           const int id_pm_block_3,     const int id_pm_block_2,     const int id_pm_block_1
                           )

{

#if 1
 // offsets to LM block
 const int OFFSET_3 = id_pm_block_3 * PM_BLOCK_SIZE_3;
 const int OFFSET_2 = id_pm_block_2 * PM_BLOCK_SIZE_2;
 const int OFFSET_1 = id_pm_block_1 * PM_BLOCK_SIZE_1;


 //printf("PM_SIZE_3 = %i, PM_SIZE_1 = %i\n", PM_SIZE_3, PM_SIZE_1 );
 // cache first input
 //#pragma unroll
 for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
   //#pragma unroll
   for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
   {
     const int prv_index_3 = WI_ID_PRV_3 + i_3 * NUM_WI_PRV_3;
     const int prv_index_1 = WI_ID_PRV_1 + i_1 * NUM_WI_PRV_1;

     const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
     const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;

     prv_cache_input_1[ prv_index_3 * PM_SIZE_input_1_dim_2 + prv_index_1 ] =
     lcl_cache_input_1[ (OFFSET_3 + lcl_index_3) * LM_SIZE_input_1_dim_2 + (OFFSET_1 + lcl_index_1) ];
   }

 // cache second input
 //#pragma unroll
 for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
   //#pragma unroll
   for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
   {
     const int prv_index_1 = WI_ID_PRV_1 + i_1 * NUM_WI_PRV_1;
     const int prv_index_2 = WI_ID_PRV_2 + i_2 * NUM_WI_PRV_2;

     const int lcl_index_1 = WI_ID_LCL_1 + i_1 * NUM_WI_LCL_1;
     const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

     prv_cache_input_2[ prv_index_1 * PM_SIZE_input_2_dim_2 + prv_index_2 ] =
     lcl_cache_input_2[ (OFFSET_1 + lcl_index_1) * LM_SIZE_input_2_dim_2 + (OFFSET_2 + lcl_index_2) ];
   }
#else
 //#pragma unroll
 for( int i = 0 ; i < PM_SIZE_3 * PM_SIZE_1 ; ++i )
   prv_cache_input_1[ i ] = 1;
 //#pragma unroll
 for( int i = 0 ; i < PM_SIZE_1 * PM_SIZE_2 ; ++i )
   prv_cache_input_2[ i ] = 1;
#endif
}


// view
struct pair
{
 float const* lhs;
 float const* rhs;
};


inline const struct pair view( __private const float* restrict in_1,
                             __private const float* restrict in_2,
                             const int i_3,
                             const int i_2,
                             const int i_1
                             )
{
 struct pair p;

 p.lhs = &in_1[ i_3 * PM_SIZE_input_1_dim_2 + i_1 ];
 p.rhs = &in_2[ i_1 * PM_SIZE_input_2_dim_2 + i_2 ];

 return p;
}

// user function (TODO pre-user function: add, ... , stencils, etc. )
inline float f( const struct pair p )
{
 return *(p.lhs) * *(p.rhs);
}


__kernel void gemm_1( __global const float* restrict glb_input_1,
                    __global const float* restrict glb_input_2,
                    __global       float* restrict glb_output
                    )
{
#if 0
 for( size_t i = 0 ; i  < GM_SIZE_3 * GM_SIZE_2 ; ++i )
   glb_output[ i ]  =  0;
#endif 

 // cache memory
 __local float lcl_cache_input_1[ LM_SIZE_input_1_dim_1 ][ LM_SIZE_input_1_dim_2 ];
 __local float lcl_cache_input_2[ LM_SIZE_input_2_dim_1 ][ LM_SIZE_input_2_dim_2 ];

 __private float prv_cache_input_1[ PM_SIZE_input_1_dim_1 ][ PM_SIZE_input_1_dim_2 ];
 __private float prv_cache_input_2[ PM_SIZE_input_2_dim_1 ][ PM_SIZE_input_2_dim_2 ];

 // memory for intermediate results
 __private float tmp_res_prv[ PM_SIZE_3 ][ PM_SIZE_2 ]; // for schematical implementation -> [ 1 * PM_SIZE_3 ][ 1 * PM_SIZE_2 ][ 1 ]
 __local   float tmp_res_lcl[ LM_SIZE_3 ][ LM_SIZE_2 ][ NUM_WI_LCL_1 ];

 // WG/WI ids
 const int WG_ID_3 = get_group_id( 2 );
 const int WG_ID_2 = get_group_id( 1 );
 const int WG_ID_1 = get_group_id( 0 );

 const int WI_ID_GLB_3 = get_global_id( 2 );
 const int WI_ID_GLB_2 = get_global_id( 1 );
 const int WI_ID_GLB_1 = get_global_id( 0 );

 const int WI_ID_LCL_3 = get_local_id( 2 );
 const int WI_ID_LCL_2 = get_local_id( 1 );
 const int WI_ID_LCL_1 = get_local_id( 0 );

 const int WI_ID_PRV_3 = 0;
 const int WI_ID_PRV_2 = 0;
 const int WI_ID_PRV_1 = 0;

 // clean tmp_res_prv
 //#pragma unroll
 for( int i_3 = 0 ; i_3 < PM_SIZE_3 / NUM_WI_PRV_3 ; ++i_3 )
   //#pragma unroll
   for( int i_2 = 0 ; i_2 < PM_SIZE_2 / NUM_WI_PRV_2 ; ++i_2 )
   {
     const int prv_index_3 = i_3;
     const int prv_index_2 = i_2;

     tmp_res_prv[ prv_index_3 ][ prv_index_2 ] = 0;
   }

 // clean tmp_res_lcl
 //#pragma unroll
 for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
   //#pragma unroll
   for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
   {
     const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
     const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

     tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 ] = 0;
   }
 barrier( CLK_LOCAL_MEM_FENCE );

 // iteration over local memory blocks
 //#pragma unroll
 for( int id_lm_block_3 = 0 ; id_lm_block_3 < NUM_WG_ITERATIONS_OVER_LM_BLOCKS_3 ; ++id_lm_block_3 )
   //#pragma unroll
   for( int id_lm_block_2 = 0 ; id_lm_block_2 < NUM_WG_ITERATIONS_OVER_LM_BLOCKS_2 ; ++id_lm_block_2 )
   {
     //#pragma unroll
     for( int id_lm_block_1 = 0 ; id_lm_block_1 < NUM_WG_ITERATIONS_OVER_LM_BLOCKS_1 ; ++id_lm_block_1 )
     {
       update_LM_cache( glb_input_1, glb_input_2,
                       lcl_cache_input_1, lcl_cache_input_2,
                       WI_ID_LCL_3, WI_ID_LCL_2, WI_ID_LCL_1,
                       WI_ID_GLB_3, WI_ID_GLB_2, WI_ID_GLB_1,
                       id_lm_block_3, id_lm_block_2, id_lm_block_1
                       );

       // iteration over private memory blocks
       //#pragma unroll
       for( int id_pm_block_3 = 0 ; id_pm_block_3 < NUM_WI_ITERATIONS_OVER_PM_BLOCKS_3 ; ++id_pm_block_3 )
         //#pragma unroll
         for( int id_pm_block_2 = 0 ; id_pm_block_2 < NUM_WI_ITERATIONS_OVER_PM_BLOCKS_2 ; ++id_pm_block_2 )
         {
           //#pragma unroll
           for( int id_pm_block_1 = 0 ; id_pm_block_1 < NUM_WI_ITERATIONS_OVER_PM_BLOCKS_1 ; ++id_pm_block_1 )
           {
             update_PM_cache( lcl_cache_input_1, lcl_cache_input_2,
                             prv_cache_input_1, prv_cache_input_2,
                             WI_ID_PRV_3, WI_ID_PRV_2, WI_ID_PRV_1,
                             WI_ID_LCL_3, WI_ID_LCL_2, WI_ID_LCL_1,
                             id_pm_block_3, id_pm_block_2, id_pm_block_1
                             );

             // computation of a WI
             //#pragma unroll
             for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
               //#pragma unroll
               for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
                 //#pragma unroll
                 for( int i_1 = 0 ; i_1 < PM_SIZE_1 ; ++i_1 )
                 {
                   tmp_res_prv[ i_3 ][ i_2 ] += f( view( prv_cache_input_1, prv_cache_input_2, i_3, i_2, i_1 ) );
                   //printf("res = %f\n", f( view( prv_cache_input_1, prv_cache_input_2, i_3, i_2, i_1 ) ) );
                 }
           } // end for-loop "i_pm_1"

           // i) copy intermediate results from private to local memory, and ii) clean "tmp_res_prv"
           //#pragma unroll
           for( int i_3 = 0 ; i_3 < PM_SIZE_3 ; ++i_3 )
             //#pragma unroll
             for( int i_2 = 0 ; i_2 < PM_SIZE_2 ; ++i_2 )
             {
               const int OFFSET_3 = id_pm_block_3 * PM_BLOCK_SIZE_3;
               const int OFFSET_2 = id_pm_block_2 * PM_BLOCK_SIZE_2;

               const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
               const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

               const int prv_index_3 = i_3;
               const int prv_index_2 = i_2;

               //if( get_global_id(0) == 0 )
               //{
               tmp_res_lcl[ OFFSET_3 + lcl_index_3 ][ OFFSET_2 + lcl_index_2 ][ WI_ID_LCL_1 ] += tmp_res_prv[ prv_index_3 ][ prv_index_2 ];
               //if( get_global_id(0) == 1 )
               //printf("prv = %f\n", tmp_res_prv[ prv_index_3 ][ prv_index_2 ] );
               //printf("tmp_res_lcl[ %i ][ %i ][ %i ] = %f\n", OFFSET_3 + lcl_index_3, OFFSET_2 + lcl_index_2, WI_ID_LCL_1, tmp_res_lcl[ OFFSET_3 + lcl_index_3 ][ OFFSET_2 + lcl_index_2 ][ WI_ID_LCL_1 ] );
               tmp_res_prv[ prv_index_3 ][ prv_index_2 ] = 0;
               //}
               //barrier( CLK_LOCAL_MEM_FENCE );
               //if( get_global_id(0) == 0 )
               //printf("lcl = %f\n", tmp_res_lcl[ OFFSET_3 + lcl_index_3 ][ OFFSET_2 + lcl_index_2 ][ WI_ID_LCL_1 ] );
             }
           barrier( CLK_LOCAL_MEM_FENCE );

         } // end for-loop "i_pm_2"
     } // end for-loop "id_lm_block_1"

     // parallel reduction of intermediate results
     //#pragma unroll
     for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 ) //+= NUM_WI_LCL_3 )
       //#pragma unroll
       for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 ) //+= NUM_WI_LCL_2 )
         //#pragma unroll
         for( int stride = NUM_WI_LCL_1 / 2 ; stride > 0 ; stride /= 2 )
         {
           const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
           const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

           if( WI_ID_LCL_1 < stride )
             tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 ] += tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 + stride ];

           barrier( CLK_LOCAL_MEM_FENCE );
         }

     // i) copy intermediate results from local memory to global memory, and ii) clean "tmp_res_lcl"
     //#pragma unroll
     for( int i_3 = 0 ; i_3 < LM_SIZE_3 / NUM_WI_LCL_3 ; ++i_3 )
       //#pragma unroll
       for( int i_2 = 0 ; i_2 < LM_SIZE_2 / NUM_WI_LCL_2 ; ++i_2 )
       {
         const int OFFSET_3 = id_lm_block_3 * LM_BLOCK_SIZE_3;
         const int OFFSET_2 = id_lm_block_2 * LM_BLOCK_SIZE_2;

         const int glb_index_3 = WI_ID_GLB_3 + i_3 * NUM_WI_GLB_3;
         const int glb_index_2 = WI_ID_GLB_2 + i_2 * NUM_WI_GLB_2;

         const int lcl_index_3 = WI_ID_LCL_3 + i_3 * NUM_WI_LCL_3;
         const int lcl_index_2 = WI_ID_LCL_2 + i_2 * NUM_WI_LCL_2;

         if( WI_ID_LCL_1 == 0 )
           glb_output[ (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
                      (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
                      WG_ID_1                                             // dim 3
                      ] += tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ 0 ];

         //  if( WI_ID_LCL_1 == 0 && WI_ID_LCL_2 == 0 )
         //  {
         //  printf("y = %i, x = %i, k = %i, WI_ID = %i\n\n", (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1), (OFFSET_2 + glb_index_2) * NUM_WG_1, WG_ID_1, WI_ID_LCL_1 );
         //printf("index = %i, OFFSET_3 = %i, glb_index_3 = %i, GM_SIZE_2 = %i, NUM_WG_1 = %i, OFFSET_2 = %i, id_lm_block_2 = %i, LM_BLOCK_SIZE_2 = %i,  glb_index_2 = %i, NUM_WG_1 = %i, WG_ID_1 = %i, res = %f\n",
         //(OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
         //                        (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
         //                        WG_ID_1,
         //OFFSET_3, glb_index_3, GM_SIZE_2, NUM_WG_1, OFFSET_2, id_lm_block_2, LM_BLOCK_SIZE_2, glb_index_2, NUM_WG_1, WG_ID_1,
         //glb_output[ (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
         //                        (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
         //                        WG_ID_1                                             // dim 3
         //                      ]
         //);
         //}
         barrier( CLK_GLOBAL_MEM_FENCE );

         //printf("res = %f\n",             glb_output[ (OFFSET_3 + glb_index_3) * (GM_SIZE_2 * NUM_WG_1) + // dim 1
         //                      (OFFSET_2 + glb_index_2) *              NUM_WG_1  + // dim 2
         //                      WG_ID_1                                             // dim 3
         //                    ] );

         // clean "tmp_res_lcl"
         tmp_res_lcl[ lcl_index_3 ][ lcl_index_2 ][ WI_ID_LCL_1 ] = 0;
         barrier( CLK_LOCAL_MEM_FENCE );
       } // end for-loop "i_2"

   } // end for-loop "i_lm_2"
}


__kernel void gemm_2( __global float* output_1,
                    __global float* output_2
                    )
{
}

)";

#endif
