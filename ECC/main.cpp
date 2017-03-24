#include "ECCExecutor.h"

int main(int argc, char* argv[])
{
	//std::ifstream cmp1("\\\\X-THINK\\Users\\Public\\cmp.txt"), cmp2("cmp2.txt");

	//std::string str1, str2;
	//int line = 1;
	//while (std::getline(cmp1, str1) && std::getline(cmp2, str2))
	//{
	//	double d1 = std::stod(str1);
	//	double d2 = std::stod(str2);

	//	if (abs(d1 - d2) > 0.001)
	//	{
	//		__debugbreak();
	//	}
	//	++line;
	//}

	//return 0;

	if(!PlatformUtil::init("NVIDIA", "GTX"))
	{
		PlatformUtil::deinit();
		return -1;
	}

	{
		ECCData data(3, "data/NNRTI.arff");
		int trainSize = 0.67 * data.getSize();
		int evalSize = data.getSize() - trainSize;

		std::vector<MultilabelInstance> inputCopy = data.getInstances();
		std::vector<MultilabelInstance> trainInstances;
		std::vector<MultilabelInstance> evalInstances;

		trainInstances.reserve(trainSize);
		evalInstances.reserve(evalSize);

		for (int i = 0; i < trainSize; i++)
		{
			int idx = Util::randomInt(inputCopy.size());
			trainInstances.push_back(inputCopy[idx]);
			inputCopy.erase(inputCopy.begin() + idx);
		}

		for (int i = 0; i < evalSize; i++)
		{
			int idx = Util::randomInt(inputCopy.size());
			MultilabelInstance inst = inputCopy[idx];
			
			for (int i = inst.getNumAttribs(); i < inst.getValueCount(); ++i)
			{
				inst.getData()[i] = -100.0;
			}

			evalInstances.push_back(inst);
			inputCopy.erase(inputCopy.begin() + idx);
		}

		ECCData trainData(trainInstances, data.getAttribCount(), data.getLabelCount());
		ECCData evalData(evalInstances, data.getAttribCount(), data.getLabelCount());

		ECCExecutor eccex(true, true, 10, 244, 10, 1, 1, 1);
		eccex.runBuild(trainData, 10, 6, 6, 100, 50);

		std::vector<double> values;
		std::vector<int> votes;

		eccex.runClassify(evalData, values, votes);
	}

	PlatformUtil::deinit();
	system("pause");
	return 0;
}