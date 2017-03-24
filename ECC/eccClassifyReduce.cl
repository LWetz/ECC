#ifndef INSTANCES_PER_ITEM
#define INSTANCES_PER_ITEM 2
#endif

#ifndef CHAINS_PER_ITEM
#define CHAINS_PER_ITEM 2
#endif

kernel void eccClassifyReduce(global double* globalResultBuffer,
	global int* globalVoteBuffer,
	global int* globalBufferSize,
	global double* labelBuffer,
	global int* labelVoteBuffer,
	global int* pNumLabels,
	global int* pChainIndex)
{
	int gidOff = get_global_id(0) * INSTANCES_PER_ITEM;
	int ensembleOff = get_global_id(1) * CHAINS_PER_ITEM;
	int tree = get_global_id(2);

	int chainIndex = *pChainIndex;
	int numLabels = *pNumLabels;

	int groupSize = localBufferSize[2];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = groupSize >> 1; i > 0; i >>= 1)
	{
		if (localTree < i)
		{
			for (int gid = gidOff; gid < gidOff + INSTANCES_PER_ITEM; ++gid)
			{
				for (int ensembleIndex = ensembleOff; ensembleIndex < ensembleOff + CHAINS_PER_ITEM; ++ensembleIndex)
				{
					int localIndex = globalBufferSize[1] * globalBufferSize[2] * gid + globalBufferSize[1] * ensembleIndex + tree;
					globalResultBuffer[localIndex] += globalResultBuffer[localIndex + i];
					globalVoteBuffer[localIndex] += globalVoteBuffer[localIndex + i];
				}
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (tree == 0)
	{
		for (int gid = gidOff; gid < gidOff + INSTANCES_PER_ITEM; ++gid)
		{
			for (int ensembleIndex = ensembleOff; ensembleIndex < ensembleOff + CHAINS_PER_ITEM; ++ensembleIndex)
			{
				int label = labelOrders[numLabels * ensembleIndex + chainIndex];

				int globalIndex = globalBufferSize[1] * globalBufferSize[2] * gid+ globalBufferSize[1] * ensembleIndex + get_group_id(2);
				int labelIndex = numLabels * globalBufferSize[1] * gid + numLabels * ensembleIndex + label;

				labelVoteBuffer[labelIndex] = globalVoteBuffer[globalIndex];
				labelBuffer[labelIndex] = globalResultBuffer[globalIndex];
			}
		}
	}
}