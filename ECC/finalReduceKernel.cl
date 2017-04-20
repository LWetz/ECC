//#define NUM_CHAINS 
//#define NUM_INSTANCES
//#define NUM_TREES
//#define NUM_LABELS
//
//#define CHAINS_PER_WG_R
//#define INSTANCES_PER_WG_R
//#define TREES_PER_WG_R
//
//#define CHAINS_PER_WI_R
//#define INSTANCES_PER_WI_R

#define INSTANCES_PER_ITEM (NUM_INSTANCES/(NUM_WI_INSTANCES_FR*NUM_WG_INSTANCES_FR))
#define CHAINS_PER_ITEM (NUM_WG_CHAINS_FC/NUM_WI_CHAINS_FR)
#define LABELS_PER_ITEM (NUM_LABELS/(NUM_WI_LABELS_FR*NUM_WG_LABELS_FR))

#define LO_IDX(I, L, C) ((NUM_WI_LABELS_FR * I + L) * NUM_WI_CHAINS_FR + C)
#define IN_IDX(I, L, WG) ((NUM_LABELS * I + L) * NUM_WG_CHAINS_FC + WG)

typedef struct OutputAtom
{
	double result;
	int vote;
}OutputAtom;

inline void addAssignOutputAtomsPrvGlo(OutputAtom* a, global OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

inline void addAssignOutputAtomsLoc(local OutputAtom* a, local OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

inline void addAssignOutputAtomsGlo(global OutputAtom* a, global OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

kernel void finalReduce(	global OutputAtom* intermediateBuffer,
							global OutputAtom* resultBuffer,
							local OutputAtom* localBuffer)
{
	int i_wg_instance = get_group_id(0);
	int i_wg_label = get_group_id(1);

	int i_wi_instance = get_local_id(0);
	int i_wi_label = get_local_id(1);
	int i_wi_chain = get_local_id(2);

	for (int i = 0; i < INSTANCES_PER_ITEM; ++i)
	{
		int instance = i_wi_instance + i_wg_instance * NUM_WI_INSTANCES_FR + i * NUM_WG_INSTANCES_FR * NUM_WI_INSTANCES_FR;
		for (int l = 0; l < LABELS_PER_ITEM; ++l)
		{
			int label = i_wi_label + i_wg_label * NUM_WI_LABELS_FR + l * NUM_WG_LABELS_FR * NUM_WI_LABELS_FR;
			OutputAtom res_prv = (OutputAtom) { 0, 0 };

			for (int c = 0; c < CHAINS_PER_ITEM; ++c)
			{
				int chain = i_wi_chain + c * NUM_WI_CHAINS_FR;
				int intermediateIndex = IN_IDX(instance, label, chain);
				addAssignOutputAtomsPrvGlo(&res_prv, intermediateBuffer + intermediateIndex);
			}

			int localIndex = LO_IDX(i_wi_instance, i_wi_label, i_wi_chain);
			localBuffer[localIndex] = res_prv;
			
			barrier(CLK_LOCAL_MEM_FENCE);

			for (int c = NUM_WI_CHAINS_FR >> 1; c > 0; c >>= 1)
			{
				if (i_wi_chain < c)
				{
					addAssignOutputAtomsLoc(localBuffer + localIndex, localBuffer + localIndex + c);
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			if (i_wi_chain == 0)
			{
				int resultIndex = instance * NUM_LABELS + label;

				resultBuffer[resultIndex] = localBuffer[localIndex];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}