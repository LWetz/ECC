#include "ECCExecutor.h"

int main2(int argc, char* argv[])
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

	if(!PlatformUtil::init("NVIDIA", "Tesla"))
	{
		PlatformUtil::deinit();
		return -1;
	}

	{
		ECCData data(14, "data/yeast.arff");
		int trainSize = 0.67 * data.getSize();
		int evalSize = data.getSize() - trainSize;

		double avgSpeed = 0.0;
		double avgAccOld = 0.0;
		double avgAccNew = 0.0;
		int iterations = 1;
		for (int i = 0; i < iterations; i++)
		{
			std::vector<MultilabelInstance> inputCopy = data.getInstances();
			std::vector<MultilabelInstance> trainInstances;
			std::vector<MultilabelInstance> evalInstances;

			trainInstances.reserve(trainSize);
			evalInstances.reserve(evalSize);

			for (int i = 0; i < trainSize; ++i)
			{
				int idx = Util::randomInt(inputCopy.size());
				trainInstances.push_back(inputCopy[idx]);
				inputCopy.erase(inputCopy.begin() + idx);
			}

			std::vector<MultilabelInstance> evalCopy;
			for (int i = 0; i < evalSize; ++i)
			{
				int idx = Util::randomInt(inputCopy.size());
				MultilabelInstance inst = inputCopy[idx];
				evalCopy.push_back(inst);
				for (int i = inst.getNumAttribs(); i < inst.getValueCount(); ++i)
				{
					inst.getData()[i] = 0.0;
				}

				evalInstances.push_back(inst);
				inputCopy.erase(inputCopy.begin() + idx);
			}

			ECCData trainData(trainInstances, data.getAttribCount(), data.getLabelCount());
			ECCData evalData(evalInstances, data.getAttribCount(), data.getLabelCount());

			ECCExecutor eccex(10, 244, 16);
			eccex.runBuild(trainData, 16, 8, 8, 100, 50);

			std::vector<double> valOld, valNew;
			std::vector<int> voteOld, voteNew;
			eccex.runClassifyOld(data, valOld, voteOld);
			//eccex.runClassifyNew(data, valNew, voteNew);
			//double speedup = eccex.getSpeedup();
			//std::cout << "Speedup: " << speedup << std::endl;
			//avgSpeed += speedup;
			double hitsOld = 0, hitsNew = 0;
			bool sameResult = true;
			for (int i = 0; i < evalCopy.size(); ++i)
			{
				MultilabelInstance iOrig = evalCopy[i];
				int numL = iOrig.getNumLabels();
				double maxVotes = 16.0 * 16.0;
				for (int l = 0; l < iOrig.getNumLabels(); ++l)
				{
					bool noprint = true;
					if (abs(valOld[i*numL + l] - valNew[i*numL + l]) > 0.001)
						{noprint = sameResult = false;}
					if (abs(voteOld[i*numL + l] - voteNew[i*numL + l]) > 0.001)
						{noprint = sameResult = false;}
					double predNew = valNew[i*numL + l] > 0 ? 1.0 : 0.0;
					double predOld = valOld[i*numL + l] > 0 ? 1.0 : 0.0;
					
					if(!noprint){
						std::cout << "Ref: " << iOrig.getData()[l + iOrig.getNumAttribs()]
						<< " | Old: " << predOld << " | New: " << predNew << std::endl;
					}
					if (predOld == iOrig.getData()[l + iOrig.getNumAttribs()])
						hitsOld++;
					if (predNew == iOrig.getData()[l + iOrig.getNumAttribs()])
						hitsNew++;
				}
				//std::cout << std::endl;
			}
			avgAccNew += (hitsNew / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0;
			avgAccOld += (hitsOld / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0;
			std::cout << "Same Result: " << std::boolalpha << sameResult << std::endl;
			std::cout << "Prediction Performance: Old " << (hitsOld / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "% | New " << (hitsNew / (evalCopy.size()*evalCopy[0].getNumLabels()))*100.0 << "%" << std::endl;
		}

		std::cout << "Average Speedup: " << avgSpeed / (double)iterations << std::endl;
		std::cout << "Average Accuracy: Old " << avgAccOld / (double)iterations << " | New " << avgAccNew / (double)iterations << std::endl;
	}

	PlatformUtil::deinit();
	system("pause");
	return 0;
}
