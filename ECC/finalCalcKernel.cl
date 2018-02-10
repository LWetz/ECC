#define INSTANCES_PER_ITEM (NUM_INSTANCES/(NUM_WI_INSTANCES_FC*NUM_WG_INSTANCES_FC))
#define LABELS_PER_ITEM (NUM_LABELS/(NUM_WI_LABELS_FC*NUM_WG_LABELS_FC))
#define CHAINS_PER_ITEM (NUM_CHAINS/(NUM_WI_CHAINS_FC*NUM_WG_CHAINS_FC))

#define LB_IDX(I, L, C) ((NUM_LABELS * I + L) * NUM_CHAINS + C)
#define LO_IDX(I, L, C) ((NUM_WI_LABELS_FC * I + L) * NUM_WI_CHAINS_FC + C)
#define IN_IDX(I, L, WG) ((NUM_LABELS * I + L) * NUM_WG_CHAINS_FC + WG)

kernel void finalCalc(	global double* labelBuffer,
						local double* localBuffer,
						global double* intermediateBuffer)
{
	int i_wg_instance = get_group_id(0);
	int i_wg_label = get_group_id(1);
	int i_wg_chain = get_group_id(2);

	int i_wi_instance = get_local_id(0);
	int i_wi_label = get_local_id(1);
	int i_wi_chain = get_local_id(2);

	for (int i = 0; i < INSTANCES_PER_ITEM; ++i)
	{
		int instance = i_wi_instance + i_wg_instance * NUM_WI_INSTANCES_FC + i * NUM_WG_INSTANCES_FC * NUM_WI_INSTANCES_FC;
		for (int l = 0; l < LABELS_PER_ITEM; ++l)
		{
			int label = i_wi_label + i_wg_label * NUM_WI_LABELS_FC + l * NUM_WG_LABELS_FC * NUM_WI_LABELS_FC;
			double res_prv = 0.0;

			for (int c = 0; c < CHAINS_PER_ITEM; ++c)
			{
				int chain = i_wi_chain + i_wg_chain * NUM_WI_CHAINS_FC + c * NUM_WG_CHAINS_FC * NUM_WI_CHAINS_FC;
				
				int labelIndex = LB_IDX(instance, label, chain);
				res_prv += labelBuffer[labelIndex];
			}

			int localIndex = LO_IDX(i_wi_instance, i_wi_label, i_wi_chain);
			localBuffer[localIndex] = res_prv;

			barrier(CLK_LOCAL_MEM_FENCE);

#if ( NUM_WI_CHAINS_FC & ( NUM_WI_CHAINS_FC - 1)) == 0
			int c = NUM_WI_CHAINS_FC / 2;
#else
			int c = pow(2, floor(log2((float)NUM_WI_CHAINS_FC)));
			if (i_wi_chain < c && i_wi_chain + c <  NUM_WI_CHAINS_FC) {
				res_lcl[i_wi_chain] += res_lcl[i_wi_chain + c];
			}
			c /= 2;
			barrier(CLK_LOCAL_MEM_FENCE);
#endif
			for (; c > 0; c /= 2)
			{
				if (i_wi_chain < c)
				{
					localBuffer[localIndex] += localBuffer[localIndex + c];
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			} 

			if (i_wi_chain == 0)
			{
				int intermediateIndex = IN_IDX(instance, label, i_wg_chain);

				intermediateBuffer[intermediateIndex] = localBuffer[localIndex];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}
