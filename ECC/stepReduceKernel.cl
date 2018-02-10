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

#define INSTANCES_PER_ITEM (NUM_INSTANCES/(NUM_WI_INSTANCES_SR*NUM_WG_INSTANCES_SR))
#define CHAINS_PER_ITEM (NUM_CHAINS/(NUM_WI_CHAINS_SR*NUM_WG_CHAINS_SR))
#define TREES_PER_ITEM (NUM_WG_TREES_SC/NUM_WI_TREES_SR)

#define LO_IDX(I, C, T) ((NUM_WI_CHAINS_SR * I + C) * NUM_WI_TREES_SR + T)
#define LB_IDX(I, L, C) ((NUM_LABELS * I + L) * NUM_CHAINS + C)
#define IN_IDX(I, C, WG) ((NUM_CHAINS * I + C) * NUM_WG_TREES_SC + WG)

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

kernel void stepReduce(	global OutputAtom* intermediateBuffer,
						global double* labelBuffer,
						global int* labelOrders,
						local OutputAtom* localBuffer,
						int forest)
{
	int i_wg_instance = get_group_id(0);
	int i_wg_chain = get_group_id(1);

	int i_wi_instance = get_local_id(0);
	int i_wi_chain = get_local_id(1);
	int i_wi_tree = get_local_id(2);

	for (int i = 0; i < INSTANCES_PER_ITEM; ++i)
	{
		int instance = i_wi_instance + i_wg_instance * NUM_WI_INSTANCES_SR + i * NUM_WG_INSTANCES_SR * NUM_WI_INSTANCES_SR;
		for (int c = 0; c < CHAINS_PER_ITEM; ++c)
		{
			int chain = i_wi_chain + i_wg_chain * NUM_WI_CHAINS_SR + c * NUM_WG_CHAINS_SR * NUM_WI_CHAINS_SR;
			OutputAtom res_prv = (OutputAtom) { 0, 0 };

			for (int t = 0; t < TREES_PER_ITEM; ++t)
			{
				int tree = i_wi_tree + t * NUM_WI_TREES_SR;
				int intermediateIndex = IN_IDX(instance, chain, tree);
				addAssignOutputAtomsPrvGlo(&res_prv, intermediateBuffer + intermediateIndex);
			}

			int localIndex = LO_IDX(i_wi_instance, i_wi_chain, i_wi_tree);
			localBuffer[localIndex] = res_prv;
			
			barrier(CLK_LOCAL_MEM_FENCE);

#if ( NUM_WI_TREES_SR & ( NUM_WI_TREES_SR - 1)) == 0
			int t = NUM_WI_TREES_SR / 2;
#else
			int t = pow(2, floor(log2((float)NUM_WI_TREES_SR)));
			if (i_wi_tree < t && i_wi_tree + t <  NUM_WI_TREES_SR) {
				res_lcl[i_wi_tree] += res_lcl[i_wi_tree + t];
			}
			t /= 2;
			barrier(CLK_LOCAL_MEM_FENCE);
#endif
			for (; t > 0; t /= 2)
			{
				if (i_wi_tree < t)
				{
					addAssignOutputAtomsLoc(localBuffer + localIndex, localBuffer + localIndex + t);
				}
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			if (i_wi_tree == 0)
			{
				int label = labelOrders[chain * NUM_LABELS + forest];
				int labelIndex = LB_IDX(instance, label, chain);

				if (localBuffer[localIndex].vote == 0)
					labelBuffer[labelIndex] = 0.5;
				else 
					labelBuffer[labelIndex] = (localBuffer[localIndex].result / localBuffer[localIndex].vote + 1.0) / 2.0;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}