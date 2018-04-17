#pragma once
#ifndef P_ARCHIVE
#define P_ARCHIVE
using namespace std;
#include "pareto.h"

struct paretoarchive{
	vector <ind> pop; // population
	int archsize;
	int optimal_size;
	paretoarchive(){archsize=0;optimal_size=0;}
	paretoarchive(int size){archsize=size;optimal_size=0;}
	~paretoarchive(){}
	void resize(int size){archsize=size;}

	void update(vector<ind>& newpop)
	{
		vector <ind> tmppop;

		pop.insert(pop.end(),newpop.begin(),newpop.end());
		//shrink pareto population by discarding outputs
		for (int i=pop.size()-newpop.size();i<pop.size();i++) {
			pop.at(i).stack_float.clear();
			pop.at(i).output.clear();
			pop[i].output_v.clear();
			pop.at(i).error.clear();
		}
		sort(pop.begin(),pop.end(),SortComplexity());
		stable_sort(pop.begin(),pop.end(),SortFit());
		vector<ind>::iterator it;
		it = std::unique (pop.begin(), pop.end(), sameFitComplexity());
		//it2 = std::unique (pop.begin(), pop.end(), sameEqn());
		pop.resize(distance(pop.begin(),it));

		int r = 0;
		while (tmppop.size()<archsize && !pop.empty())
		{
			pareto_fc(pop);
			sort(pop.begin(),pop.end(),SortRank());
			while(!pop.empty() && pop.back().rank==1)
			{
				pop.back().rank+=r;
				tmppop.push_back(pop.back());
				pop.pop_back();
			}
			if (r==0)
				optimal_size = tmppop.size();
			r++;
		}

		pop = tmppop;
		sort(pop.begin(),pop.end(),SortRank());
		stable_sort(pop.begin(),pop.end(),SortFit());
	}

};

#endif