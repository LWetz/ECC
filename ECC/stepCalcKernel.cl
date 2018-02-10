//#define NUM_CHAINS 
//#define NUM_INSTANCES
//#define NUM_TREES
//#define NUM_LABELS
//#define NUM_ATTRIBUTES
//#define MAX_LEVEL
//#define NODES_PER_TREE

#define INSTANCES_PER_ITEM (NUM_INSTANCES/(NUM_WI_INSTANCES_SC*NUM_WG_INSTANCES_SC))
#define CHAINS_PER_ITEM (NUM_CHAINS/(NUM_WI_CHAINS_SC*NUM_WG_CHAINS_SC))
#define TREES_PER_ITEM (NUM_TREES/(NUM_WI_TREES_SC*NUM_WG_TREES_SC))

#define LB_IDX(I, L, C) ((NUM_LABELS * I + L) * NUM_CHAINS + C)
#define LO_IDX(I, C, T) ((NUM_WI_CHAINS_SC * I + C) * NUM_WI_TREES_SC + T)
#define IN_IDX(I, C, WG) ((NUM_CHAINS * I + C) * NUM_WG_TREES_SC + WG)
#define TREE_IDX(C,T) ((C * NUM_TREES + T) * NODES_PER_TREE)

#define VIEW(I,C,T) (InputAtom){(Instance){data + I * (NUM_ATTRIBUTES+NUM_LABELS), labelBuffer + LB_IDX(I,0,C)},(Tree){attributeIndices + TREE_IDX(C,T), nodeValues + TREE_IDX(C,T)}}

#define IDX(I,C,F,T) (((NUM_CHAINS * I + C) * NUM_LABELS + F) * NUM_TREES + T)

typedef struct OutputAtom
{
	double result;
	int vote;
}OutputAtom;

inline void addAssignOutputAtomsPrv(OutputAtom* a, OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

inline void addAssignOutputAtomsLoc(local OutputAtom* a,local OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

inline void addAssignOutputAtomsGlo(global OutputAtom* a, global OutputAtom* b)
{
	a->vote += b->vote;
	a->result += b->result;
}

typedef struct Instance
{
	global double* attributes;
	global double* labels;
}Instance;

typedef struct Tree
{
	global int* nodeIndices;
	global double* nodeValues;
}Tree;

typedef struct InputAtom
{
	Instance inst;
	Tree tree;
}InputAtom;

OutputAtom traverse(InputAtom input)
{ 
    int right = 0;
    int nodeIndex = 0;

    for(int level = 0; level < MAX_LEVEL; ++level)
    {
        double nodeValue = input.tree.nodeValues[nodeIndex];
        int attributeIndex = input.tree.nodeIndices[nodeIndex];

		double value;
		if (attributeIndex < NUM_ATTRIBUTES)
			value = input.inst.attributes[attributeIndex];
		else
		{
			value = input.inst.labels[(attributeIndex - NUM_ATTRIBUTES)*NUM_CHAINS];
			value = value > 0 ? 1.0 : 0.0;
		}

        if(value > nodeValue) 
        {
            right = 2;
        }
        else
        {   
            right = 1;
        }

        nodeIndex = nodeIndex * 2 + right;
    }

	OutputAtom o;
	o.result = input.tree.nodeValues[nodeIndex];
	o.vote = o.result != 0 ? 1 : 0;

	return o;
}

kernel void stepCalc(
	//input (read-only)
		global double* nodeValues,
		global int* attributeIndices,
		global double* data,
		global double* labelBuffer,
		local OutputAtom* localBuffer,
		global OutputAtom* intermediateBuffer
	)
{
	int i_wg_instance = get_group_id(0);
	int i_wg_chain = get_group_id(1);
	int i_wg_tree = get_group_id(2);

	int i_wi_instance = get_local_id(0);
	int i_wi_chain = get_local_id(1);
	int i_wi_tree = get_local_id(2);

	for (int i = 0; i < INSTANCES_PER_ITEM; ++i)
	{
		int instance = i_wi_instance + i_wg_instance * NUM_WI_INSTANCES_SC + i * NUM_WG_INSTANCES_SC * NUM_WI_INSTANCES_SC;
		for (int c = 0; c < CHAINS_PER_ITEM; ++c)
		{
			int chain = i_wi_chain + i_wg_chain * NUM_WI_CHAINS_SC + c * NUM_WG_CHAINS_SC * NUM_WI_CHAINS_SC;
			OutputAtom res_prv = (OutputAtom) { 0, 0 };

			for (int t = 0; t < TREES_PER_ITEM; ++t)
			{
				int tree = i_wi_tree + i_wg_tree * NUM_WI_TREES_SC + t * NUM_WG_TREES_SC * NUM_WI_TREES_SC;
				OutputAtom res = traverse(VIEW(instance, chain, tree));
				addAssignOutputAtomsPrv(&res_prv, &res);
			}

			int localIndex = LO_IDX(i_wi_instance, i_wi_chain, i_wi_tree);
			localBuffer[localIndex] = res_prv;

			barrier(CLK_LOCAL_MEM_FENCE);

#if ( NUM_WI_TREES_SC & ( NUM_WI_TREES_SC - 1)) == 0
			int t = NUM_WI_TREES_SC / 2;
#else
			int t = pow(2, floor(log2((float)NUM_WI_TREES_SC)));
			if (i_wi_tree < t && i_wi_tree + t <  NUM_WI_TREES_SC) {
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
				int intermediateIndex = IN_IDX(instance, chain, i_wg_tree);
				intermediateBuffer[intermediateIndex] = localBuffer[localIndex];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}
