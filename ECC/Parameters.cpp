#include "ECCExecutor.h" 
#include "atf_library/atf.h" 
#include <fstream>

#define NUM_CHAINS 64 
#define NUM_TREES 32 
#define MAX_LEVEL 10

ECCExecutor *ecc;
ECCData *evalData;
std::vector<MultilabelInstance> evalCopy;
std::vector<double> valOld, valNew, valFixed;
std::vector<int> voteOld, voteNew, voteFixed;

std::ofstream outfile;

bool firstRun;
int numAttributes = 0;
int numLabels = 0;
int numInstances = 0;
std::map<std::string, int> extraParams;

enum TuneTime {TuneLoopStep, TuneLoopFinal, TuneLoopFinalNoRemain, TuneRemainStep, TuneRemainFinal};

TuneTime tuneTime;

void isSameResult()
{
	bool sameResult = true;
	for (int i = 0; i < (evalCopy.size()) && sameResult; ++i)
	{
		MultilabelInstance iOrig = evalCopy[i];
		int numL = iOrig.getNumLabels();
		for (int l = 0; (l < numLabels) && sameResult; ++l)
		{
			if (abs(valFixed[i*numL + l] - valNew[i*numL + l]) > 0.001)
			{
				sameResult = false;
			}
			if (abs(voteFixed[i*numL + l] - voteNew[i*numL + l]) > 0.001)
			{
				sameResult = false;
			}
		}
	}
	std::cout << "Same Result: " << std::boolalpha << sameResult << std::endl;
}

void loss()
{
	double hitsOld = 0, hitsNew = 0, hitsFixed = 0;
	for (int i = 0; i < evalCopy.size(); ++i)
	{
		MultilabelInstance iOrig = evalCopy[i];
		bool hitOld = 1.0;
		bool hitNew = 1.0;
		bool hitFixed = 1.0;
		for (int l = 0; l < numLabels; ++l)
		{
			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
			if (predOld != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				hitOld = 0.0; 
				break;
			}
			if (predNew != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				hitNew = 0.0;
				break;
			}
			if (predFixed != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				hitFixed = 0.0;
				break;
			}
		}
		hitsOld += hitOld;
		hitsNew += hitNew;
		hitsFixed += hitFixed;
	}
	double oldLoss = 1.0 - (hitsOld  / (double)evalCopy.size());
	double newLoss = 1.0 - (hitsNew / (double)evalCopy.size());
	double fixedLoss = 1.0 - (hitsFixed / (double)evalCopy.size());

	std::cout << "0/1 Loss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
	outfile << "0/1 Loss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
}

void hammingloss()
{
	double hitsOld = 0, hitsNew = 0, hitsFixed = 0;
	for (int i = 0; i < evalCopy.size(); ++i)
	{
		MultilabelInstance iOrig = evalCopy[i];

		for (int l = 0; l < numLabels; ++l)
		{
			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
			if (predOld != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				hitsOld += 1.0;
			}
			if (predNew != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				hitsNew += 1.0;
			}
			if (predFixed != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				hitsFixed += 1.0;
			}
		}
	}
	double oldLoss = 1.0 - (hitsOld / (double)(numLabels + evalCopy.size()));
	double newLoss = 1.0 - (hitsNew / (double)(numLabels + evalCopy.size()));
	double fixedLoss = 1.0 - (hitsFixed / (double)(numLabels + evalCopy.size()));

	std::cout << "HammingLoss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
	outfile << "HammingLoss: " << oldLoss << " (OLD) | " << newLoss << " (NEW) | " << fixedLoss << " (FIXED)" << std::endl;
}

void accuracy()
{
	double oldAccuracy = 0.0;
	double newAccuracy = 0.0;
	double fixedAccuracy = 0.0;
	for (int i = 0; i < evalCopy.size(); ++i)
	{
		MultilabelInstance iOrig = evalCopy[i];

		double unionOld = 0.0, unionNew = 0.0, unionFixed = 0.0;
		double intersectionOld = 0.0, intersectionNew = 0.0, intersectionFixed = 0.0;

		for (int l = 0; l < numLabels; ++l)
		{
			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
			if (predOld == 1.0 || iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				unionOld += 1.0;
			}
			if (predNew == 1.0 || iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				unionNew += 1.0;
			}
			if (predFixed == 1.0 || iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				unionFixed += 1.0;
			}
			if (predOld == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				intersectionOld += 1.0;
			}
			if (predNew == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				intersectionNew += 1.0;
			}
			if (predFixed == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				intersectionFixed += 1.0;
			}
		}
		oldAccuracy += intersectionOld / unionOld;
		newAccuracy += intersectionNew / unionNew;
		fixedAccuracy += intersectionFixed / unionFixed;
	}
	oldAccuracy /= ((double)evalCopy.size());
	newAccuracy /= ((double)evalCopy.size());
	fixedAccuracy /= ((double)evalCopy.size());

	std::cout << "Accuracy: " << oldAccuracy << " (OLD) | " << newAccuracy << " (NEW) | " << fixedAccuracy << " (FIXED)" << std::endl;
	outfile << "Accuracy: " << oldAccuracy << " (OLD) | " << newAccuracy << " (NEW) | " << fixedAccuracy << " (FIXED)" << std::endl;
}

void fmeasure()
{
	double oldFMeasure = 0.0;
	double newFMeasure = 0.0;
	double fixedFMeasure = 0.0;

	for (int l = 0; l < numLabels; ++l)
	{
		double tpOld = 0.0, tpNew = 0.0, tpFixed = 0.0;
		double fpfnOld = 0.0, fpfnNew = 0.0, fpfnFixed = 0.0;

		for (int i = 0; i < evalCopy.size(); ++i)
		{
			MultilabelInstance iOrig = evalCopy[i];

			double predNew = valNew[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predOld = valOld[i*numLabels + l] > 0 ? 1.0 : 0.0;
			double predFixed = valFixed[i*numLabels + l] > 0 ? 1.0 : 0.0;
			if (predOld == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				tpOld += 2.0;
			}
			if (predNew == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				tpNew += 2.0;
			}
			if (predFixed == 1.0 && iOrig.getData()[l + iOrig.getNumAttribs()] == 1.0)
			{
				tpFixed += 2.0;
			}
			if (predOld != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				fpfnOld += 1.0;
			}
			if (predNew != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				fpfnNew += 1.0;
			}
			if (predFixed != iOrig.getData()[l + iOrig.getNumAttribs()])
			{
				fpfnFixed += 1.0;
			}
		}
		oldFMeasure += tpOld / (tpOld + fpfnOld);
		newFMeasure += tpNew / (tpNew + fpfnNew);
		fixedFMeasure += tpFixed / (tpFixed + fpfnFixed);
	}
	oldFMeasure /= (double)numLabels;
	newFMeasure /= (double)numLabels;
	fixedFMeasure /= (double)numLabels;

	std::cout << "FMeasure: " << oldFMeasure << " (OLD) | " << newFMeasure << " (NEW) | " << fixedFMeasure << " (FIXED)" << std::endl;
	outfile << "FMeasure: " << oldFMeasure << " (OLD) | " << newFMeasure << " (NEW) | " << fixedFMeasure << " (FIXED)" << std::endl;
}

#define MIN(a,b) ((a) > (b)) ? (a) : (b);

void logloss()
{
	double oldLogLoss = 0.0;
	double newLogLoss = 0.0;
	double fixedLogLoss = 0.0;

	double maximum = log(evalCopy.size());

	for (int i = 0; i < evalCopy.size(); ++i)
	{
		for (int l = 0; l < numLabels; ++l)
		{
			MultilabelInstance iOrig = evalCopy[i];

			double predNew = ((double)valNew[i*numLabels + l] / (double)voteNew[i*numLabels+1]) * 0.5 + 0.5;
			double predOld = ((double)valOld[i*numLabels + l] / (double)voteOld[i*numLabels + 1]) * 0.5 + 0.5;
			double predFixed = ((double)valFixed[i*numLabels + l] / (double)voteFixed[i*numLabels + 1]) * 0.5 + 0.5;
			double real = iOrig.getData()[l + iOrig.getNumAttribs()];
			newLogLoss += MIN(maximum, -(log(predNew)*real + log(1 - predNew)*(1 - real)));
			oldLogLoss += MIN(maximum, -(log(predOld)*real + log(1 - predOld)*(1 - real)));
			fixedLogLoss += MIN(maximum, -(log(predFixed)*real + log(1 - predFixed)*(1 - real)));
		}
	}
	oldLogLoss /= (double)(numLabels + evalCopy.size());
	newLogLoss /= (double)(numLabels + evalCopy.size());
	fixedLogLoss /= (double)(numLabels + evalCopy.size());

	std::cout << "LogLoss: " << oldLogLoss << " (OLD) | " << newLogLoss << " (NEW) | " << fixedLogLoss << " (FIXED)" << std::endl;
	outfile << "LogLoss: " << oldLogLoss << " (OLD) | " << newLogLoss << " (NEW) | " << fixedLogLoss << " (FIXED)" << std::endl;
}

size_t tune(atf::configuration config)
{
	std::map<std::string, int> params = extraParams;
	valNew.clear();
	voteNew.clear();
	for (auto it = config.begin(); it != config.end(); ++it)
		params[it->first] = it->second;
	params["NUM_WG_CHAINS_SR"] = params["NUM_WG_CHAINS_SC"];
	params["NUM_WG_INSTANCES_SR"] = params["NUM_WG_INSTANCES_SC"];
	params["NUM_WI_CHAINS_SR"] = params["NUM_WI_CHAINS_SC"];
	params["NUM_WI_INSTANCES_SR"] = params["NUM_WI_INSTANCES_SC"];
	params["NUM_WG_LABELS_FR"] = params["NUM_WG_LABELS_FC"];
	params["NUM_WG_INSTANCES_FR"] = params["NUM_WG_INSTANCES_FC"];
	params["NUM_WI_LABELS_FR"] = params["NUM_WI_LABELS_FC"];
	params["NUM_WI_INSTANCES_FR"] = params["NUM_WI_INSTANCES_FC"];

	params["NUM_WG_CHAINS_SR_L"] = params["NUM_WG_CHAINS_SC_L"];
	params["NUM_WG_INSTANCES_SR_L"] = params["NUM_WG_INSTANCES_SC_L"];
	params["NUM_WI_CHAINS_SR_L"] = params["NUM_WI_CHAINS_SC_L"];
	params["NUM_WI_INSTANCES_SR_L"] = params["NUM_WI_INSTANCES_SC_L"];
	params["NUM_WG_LABELS_FR_L"] = params["NUM_WG_LABELS_FC_L"];
	params["NUM_WG_INSTANCES_FR_L"] = params["NUM_WG_INSTANCES_FC_L"];
	params["NUM_WI_LABELS_FR_L"] = params["NUM_WI_LABELS_FC_L"];
	params["NUM_WI_INSTANCES_FR_L"] = params["NUM_WI_INSTANCES_FC_L"];
	ecc->runClassifyNew(*evalData, valNew, voteNew, params);
	std::cout << "Time: " << ecc->getNewCPUTime() << std::endl;
	if (firstRun)
	{
		isSameResult();
		loss();
		hammingloss();
		accuracy();
		fmeasure();
		logloss();
		firstRun = false;
	}

	switch (tuneTime)
	{
	case TuneLoopStep:
		return ecc->getNewLoopStepTime();
		break;
	case TuneLoopFinal:
		return ecc->getNewLoopFinalTime();
		break;
	case TuneRemainStep:
		return ecc->getNewRemainStepTime();
		break;
	case TuneRemainFinal:
	case TuneLoopFinalNoRemain:
		return ecc->getNewCPUTime();
		break;
	}
}

void tuneClassify() { // ZEITEN NOCHMAL TRENNEN DANN KOPIEREN
	firstRun = true;

	size_t instCnt = ecc->instancesForMemory(1u * 64u * 1024u * 1024u, evalData->getAttribCount(), evalData->getLabelCount());
	instCnt = instCnt > numInstances ? numInstances : instCnt;

	extraParams.clear();
	extraParams["NUM_INSTANCES"] = instCnt;
	extraParams["NUM_INSTANCES_L"] = numInstances % instCnt;
	extraParams["NUM_LABELS"] = numLabels;
	extraParams["NUM_ATTRIBUTES"] = numAttributes;
	extraParams["NUM_CHAINS"] = NUM_CHAINS;
	extraParams["NUM_TREES"] = NUM_TREES;
	extraParams["MAX_LEVEL"] = MAX_LEVEL;
	extraParams["NODES_PER_TREE"] = pow(2.0f, MAX_LEVEL + 1) - 1;
	extraParams["NUM_WG_CHAINS_FC"] = extraParams["NUM_WG_INSTANCES_FC"] = extraParams["NUM_WG_LABELS_FC"] = extraParams["NUM_WI_CHAINS_FC"] = extraParams["NUM_WI_INSTANCES_FC"] =
	extraParams["NUM_WI_LABELS_FC"] = extraParams["NUM_WI_CHAINS_FR"] = 1;
	extraParams["NUM_WG_CHAINS_FC_L"] = extraParams["NUM_WG_INSTANCES_FC_L"] = extraParams["NUM_WG_LABELS_FC_L"] = extraParams["NUM_WI_CHAINS_FC_L"] = extraParams["NUM_WI_INSTANCES_FC_L"] =
	extraParams["NUM_WI_LABELS_FC_L"] = extraParams["NUM_WI_CHAINS_FR_L"] = 1;

	auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WG_CHAINS_SC) { return (NUM_CHAINS % tp_NUM_WG_CHAINS_SC) == 0; });
	auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_SC) { return (numInstances % tp_NUM_WG_INSTANCES_SC) == 0; });
	auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WG_TREES_SC) { return (NUM_TREES % tp_NUM_WG_TREES_SC) == 0; });
	auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_SC) { return ((NUM_CHAINS / tp_NUM_WG_CHAINS_SC) % tp_NUM_WI_CHAINS_SC) == 0; });
	auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_SC) { return ((numInstances / tp_NUM_WG_INSTANCES_SC) % tp_NUM_WI_INSTANCES_SC) == 0; });
	auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WI_TREES_SC) { return ((NUM_TREES / tp_NUM_WG_TREES_SC) % tp_NUM_WI_TREES_SC) == 0; });
	auto tp_NUM_WI_TREES_SR = atf::tp("NUM_WI_TREES_SR", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WI_TREES_SR) { return (tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SR) == 0; });

	tuneTime = TuneLoopStep;
	auto tunerLoopStep = atf::exhaustive();//atf::open_tuner(atf::cond::evaluations(1000));
	auto best_config = tunerLoopStep(
		G(tp_NUM_WG_CHAINS_SC, tp_NUM_WI_CHAINS_SC),
		G(tp_NUM_WG_INSTANCES_SC, tp_NUM_WI_INSTANCES_SC),
		G(tp_NUM_WG_TREES_SC, tp_NUM_WI_TREES_SC, tp_NUM_WI_TREES_SR)
	)(tune);

	for (auto it = best_config.begin(); it != best_config.end(); ++it)
		extraParams[it->first] = it->second;

	auto tp_NUM_WG_CHAINS_FC = atf::tp("NUM_WG_CHAINS_FC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WG_CHAINS_FC) { return (NUM_CHAINS % tp_NUM_WG_CHAINS_FC) == 0; });
	auto tp_NUM_WG_INSTANCES_FC = atf::tp("NUM_WG_INSTANCES_FC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_FC) { return (numInstances % tp_NUM_WG_INSTANCES_FC) == 0; });
	auto tp_NUM_WG_LABELS_FC = atf::tp("NUM_WG_LABELS_FC", atf::interval(1, numLabels),
		[&](auto tp_NUM_WG_LABELS_FC) { return (numLabels % tp_NUM_WG_LABELS_FC) == 0; });
	auto tp_NUM_WI_CHAINS_FC = atf::tp("NUM_WI_CHAINS_FC", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_FC) { return ((NUM_CHAINS / tp_NUM_WG_CHAINS_FC) % tp_NUM_WI_CHAINS_FC) == 0; });
	auto tp_NUM_WI_INSTANCES_FC = atf::tp("NUM_WI_INSTANCES_FC", atf::interval(1, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_FC) { return ((numInstances / tp_NUM_WG_INSTANCES_FC) % tp_NUM_WI_INSTANCES_FC) == 0; });
	auto tp_NUM_WI_LABELS_FC = atf::tp("NUM_WI_LABELS_FC", atf::interval(1, numLabels),
		[&](auto tp_NUM_WI_LABELS_FC) { return ((numLabels / tp_NUM_WG_LABELS_FC) % tp_NUM_WI_LABELS_FC) == 0; });
	auto tp_NUM_WI_CHAINS_FR = atf::tp("NUM_WI_CHAINS_FR", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_FR) { return (tp_NUM_WG_CHAINS_FC % tp_NUM_WI_CHAINS_FR) == 0; });

	tuneTime = extraParams["NUM_INSTANCES_L"] > 0 ? TuneLoopFinal : TuneLoopFinalNoRemain;
	auto tunerLoopFinal = atf::exhaustive();
	best_config = tunerLoopFinal(
		G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
		G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
		G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC)
	)(tune);

	for (auto it = best_config.begin(); it != best_config.end(); ++it)
		extraParams[it->first] = it->second;

	if (extraParams["NUM_INSTANCES_L"] == 0)
	{
		outfile << "Best time:" << ((double)tunerLoopFinal.best_measured_result()) * 1e-06 << "ms" << std::endl << std::endl;
		for (auto it = extraParams.begin(); it != extraParams.end(); ++it)
			outfile << it->first << " = " << it->second << std::endl;
		return;
	}

	auto tp_NUM_WG_CHAINS_SC = atf::tp("NUM_WG_CHAINS_SC_L", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WG_CHAINS_SC) { return (NUM_CHAINS % tp_NUM_WG_CHAINS_SC) == 0; });
	auto tp_NUM_WG_INSTANCES_SC = atf::tp("NUM_WG_INSTANCES_SC_L", atf::interval(1, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_SC) { return (numInstances % tp_NUM_WG_INSTANCES_SC) == 0; });
	auto tp_NUM_WG_TREES_SC = atf::tp("NUM_WG_TREES_SC_L", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WG_TREES_SC) { return (NUM_TREES % tp_NUM_WG_TREES_SC) == 0; });
	auto tp_NUM_WI_CHAINS_SC = atf::tp("NUM_WI_CHAINS_SC_L", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_SC) { return ((NUM_CHAINS / tp_NUM_WG_CHAINS_SC) % tp_NUM_WI_CHAINS_SC) == 0; });
	auto tp_NUM_WI_INSTANCES_SC = atf::tp("NUM_WI_INSTANCES_SC_L", atf::interval(1, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_SC) { return ((numInstances / tp_NUM_WG_INSTANCES_SC) % tp_NUM_WI_INSTANCES_SC) == 0; });
	auto tp_NUM_WI_TREES_SC = atf::tp("NUM_WI_TREES_SC_L", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WI_TREES_SC) { return ((NUM_TREES / tp_NUM_WG_TREES_SC) % tp_NUM_WI_TREES_SC) == 0; });
	auto tp_NUM_WI_TREES_SR = atf::tp("NUM_WI_TREES_SR_L", atf::interval(1, NUM_TREES),
		[&](auto tp_NUM_WI_TREES_SR) { return (tp_NUM_WG_TREES_SC % tp_NUM_WI_TREES_SR) == 0; });

	tuneTime = TuneRemainStep;
	auto tunerRemainLoop = atf::exhaustive();
	best_config = tunerRemainLoop(
		G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
		G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
		G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC)
	)(tune);

	for (auto it = best_config.begin(); it != best_config.end(); ++it)
		extraParams[it->first] = it->second;

	auto tp_NUM_WG_CHAINS_FC = atf::tp("NUM_WG_CHAINS_FC_L", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WG_CHAINS_FC) { return (NUM_CHAINS % tp_NUM_WG_CHAINS_FC) == 0; });
	auto tp_NUM_WG_INSTANCES_FC = atf::tp("NUM_WG_INSTANCES_FC_L", atf::interval(1, numInstances),
		[&](auto tp_NUM_WG_INSTANCES_FC) { return (numInstances % tp_NUM_WG_INSTANCES_FC) == 0; });
	auto tp_NUM_WG_LABELS_FC = atf::tp("NUM_WG_LABELS_FC_L", atf::interval(1, numLabels),
		[&](auto tp_NUM_WG_LABELS_FC) { return (numLabels % tp_NUM_WG_LABELS_FC) == 0; });
	auto tp_NUM_WI_CHAINS_FC = atf::tp("NUM_WI_CHAINS_FC_L", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_FC) { return ((NUM_CHAINS / tp_NUM_WG_CHAINS_FC) % tp_NUM_WI_CHAINS_FC) == 0; });
	auto tp_NUM_WI_INSTANCES_FC = atf::tp("NUM_WI_INSTANCES_FC_L", atf::interval(1, numInstances),
		[&](auto tp_NUM_WI_INSTANCES_FC) { return ((numInstances / tp_NUM_WG_INSTANCES_FC) % tp_NUM_WI_INSTANCES_FC) == 0; });
	auto tp_NUM_WI_LABELS_FC = atf::tp("NUM_WI_LABELS_FC_L", atf::interval(1, numLabels),
		[&](auto tp_NUM_WI_LABELS_FC) { return ((numLabels / tp_NUM_WG_LABELS_FC) % tp_NUM_WI_LABELS_FC) == 0; });
	auto tp_NUM_WI_CHAINS_FR = atf::tp("NUM_WI_CHAINS_FR_L", atf::interval(1, NUM_CHAINS),
		[&](auto tp_NUM_WI_CHAINS_FR) { return (tp_NUM_WG_CHAINS_FC % tp_NUM_WI_CHAINS_FR) == 0; });

	tuneTime = TuneRemainFinal;
	auto tunerRemainFinal = atf::exhaustive();
	best_config = tunerRemainFinal(
		G(tp_NUM_WG_CHAINS_FC, tp_NUM_WI_CHAINS_FC, tp_NUM_WI_CHAINS_FR),
		G(tp_NUM_WG_INSTANCES_FC, tp_NUM_WI_INSTANCES_FC),
		G(tp_NUM_WG_LABELS_FC, tp_NUM_WI_LABELS_FC)
	)(tune);

	for (auto it = best_config.begin(); it != best_config.end(); ++it)
		extraParams[it->first] = it->second;

	outfile << "Best time:" << ((double)tunerRemainFinal.best_measured_result()) * 1e-06 << "ms" << std::endl << std::endl;
	for (auto it = extraParams.begin(); it != extraParams.end(); ++it)
		outfile << it->first << " = " << it->second << std::endl;
}
int main(int argc, char* argv[]) {
	std::cout << "START" << std::endl;

	std::string pname = "NVIDIA";
	std::string dname = "k20";
	if (argc > 1)
		dname = argv[1];
	if (argc > 2)
		pname = argv[2];

	if (!PlatformUtil::init(pname, dname))
	{
		PlatformUtil::deinit();
		return -1;
	}
	std::cout << "Platform created!" << std::endl;

	std::map<std::string, size_t> dataSets;
	dataSets["data/bibtex.arff"] = 159;
	dataSets["data/bookmarks.arff"] = 208;
	dataSets["data/CAL500.arff"] = 174;
	dataSets["data/Corel5k.arff"] = 374;
	//dataSets["data/delicious.arff"] = 983;
	dataSets["data/emotions.arff"] = 6;
	dataSets["data/enron.arff"] = 53;
	dataSets["data/flags.arff"] = 7;
	dataSets["data/genbase.arff"] = 27;
	dataSets["data/mediamill.arff"] = 101;
	dataSets["data/medical.arff"] = 45;
	dataSets["data/NNRTI.arff"] = 3;
	dataSets["data/scene.arff"] = 6;
	dataSets["data/tmc2007.arff"] = 22;
	dataSets["data/yeast.arff"] = 14;

	//	dataSets.clear();
	//	dataSets["data/bibtex.arff"] = 159;

	outfile = std::ofstream("results_" + pname + "_" + dname + ".txt");
	outfile << "NUM_CHAINS = " << NUM_CHAINS << std::endl;
	outfile << "NUM_TREES = " << NUM_TREES << std::endl;
	outfile << "MAX_LEVEL = " << MAX_LEVEL << std::endl << std::endl;

	for (auto it = dataSets.begin(); it != dataSets.end(); ++it)
	{
		std::cout << "Dataset: " << it->first << std::endl;
		outfile << "Dataset: " << it->first << std::endl;
		//try{
		ECCData data(it->second, it->first);
		outfile << "instances: " << data.getSize() << " attribute: " << data.getAttribCount() << " labels: " << data.getLabelCount() << std::endl;
		int trainSize = 0.67 * data.getSize();
		int evalSize = data.getSize() - trainSize;
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
		evalData = new ECCData(evalInstances, data.getAttribCount(), data.getLabelCount());
		ecc = new ECCExecutor(MAX_LEVEL, evalInstances[0].getValueCount(), NUM_TREES);
		ecc->runBuild(trainData, NUM_TREES, NUM_CHAINS, NUM_CHAINS, 100, 50);
		ecc->runClassifyOld(*evalData, valOld, voteOld, false);
		outfile << "Old time: " << ((double)ecc->getOldTime()) * 1e-06 << "ms" << std::endl;
		ecc->runClassifyOld(*evalData, valFixed, voteFixed, true);
		outfile << "Old time fixed: " << ((double)ecc->getOldTime()) * 1e-06 << "ms" << std::endl;
		numLabels = evalData->getLabelCount();
		numAttributes = evalData->getAttribCount();
		numInstances = evalData->getInstances().size();
		tuneClassify();
		delete ecc;
		delete evalData;
		valOld.clear();
		voteOld.clear();
		valFixed.clear();
		voteFixed.clear();
		evalCopy.clear();
		//}catch(std::exception e)
		//{
		//	std::cout << e.what() << std::endl;
		//        continue;
		//}

	}
	PlatformUtil::deinit();
	system("Pause");
	return 0;
}
