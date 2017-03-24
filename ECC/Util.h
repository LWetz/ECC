#pragma once
#include<random>
#include<vector>
#include<fstream>
#include<iostream>
#include<string>
#include<chrono>

namespace Util
{
	class StopWatch
	{
	private:
		std::chrono::time_point<std::chrono::system_clock> startTime, endTime;

	public:
		void start();
		double stop();
	};

	int randomInt(int min, int max);
	int randomInt(int bound);
	int randomInt(int bound, const std::vector<int>& exclude);

	std::string loadFileToString(const std::string& filename);
}