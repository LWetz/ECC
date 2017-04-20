//#define NUM_CHAINS 
//#define NUM_INSTANCES
//#define NUM_TREES
//#define NUM_LABELS
//
//#define CHAINS_PER_WG_RF
//#define INSTANCES_PER_WG_RF
//#define LABELS_PER_WG_RF
//
//#define INSTANCES_PER_WI_RF
//#define LABELS_PER_WI_RF

#define INSTANCES_PER_ITEM (NUM_INSTANCES/(NUM_WI_INSTANCES_FC*NUM_WG_INSTANCES_FC))
#define LABELS_PER_ITEM (NUM_LABELS/(NUM_WI_LABELS_FC*NUM_WG_LABELS_FC))
#define CHAINS_PER_ITEM (NUM_CHAINS/(NUM_WI_CHAINS_FC*NUM_WG_CHAINS_FC))

#define LB_IDX(I, L, C) ((NUM_LABELS * I + L) * NUM_CHAINS + C)
#define LO_IDX(I, L, C) ((NUM_WI_LABELS_FC * I + L) * NUM_WI_CHAINS_FC + C)
#define IN_IDX(I, L, WG) ((NUM_LABELS * I + L) * NUM_WG_CHAINS_FC + WG)

typedef struct OutputAtom
{
	double result;
	int vote;
}OutputAtom;

inline void addAssignOutputAtomsLoc(local OutputAtom* a, local OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

inline void addAssignOutputAtomsPrvGlo(OutputAtom* a, global OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

kernel void finalCalc(	global OutputAtom* labelBuffer,
						local OutputAtom* localBuffer,
						global OutputAtom* intermediateBuffer)
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
			OutputAtom res_prv = (OutputAtom) { 0, 0 };

			for (int c = 0; c < CHAINS_PER_ITEM; ++c)
			{
				int chain = i_wi_chain + i_wg_chain * NUM_WI_CHAINS_FC + c * NUM_WG_CHAINS_FC * NUM_WI_CHAINS_FC;
				global OutputAtom* res = labelBuffer + LB_IDX(instance, label, chain);
				addAssignOutputAtomsPrvGlo(&res_prv, res);
			}

			int localIndex = LO_IDX(i_wi_instance, i_wi_label, i_wi_chain);
			localBuffer[localIndex] = res_prv;

			barrier(CLK_LOCAL_MEM_FENCE);

			for (int c = NUM_WI_CHAINS_FC >> 1; c > 0; c >>= 1)
			{
				if (i_wi_chain < c)
				{
					addAssignOutputAtomsLoc(localBuffer + localIndex, localBuffer + localIndex + c);
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