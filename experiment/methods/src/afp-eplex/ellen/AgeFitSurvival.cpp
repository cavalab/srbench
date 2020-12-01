#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"

void AgeFitSurvival(vector<ind>& pop,params& p,vector<Randclass>& r)
{
	//boost::progress_timer timer;
	vector<float> fit(2);
	vector<int> age(2);
	vector<int> fitindex(2);
	//float minfit=p.max_fit;
	//float minage=p.g; // maximum age = total max generations
	int loser;
	int counter =0;
	bool draw=true;
	int popsize = (int)floor((float)pop.size()/2);
	while (pop.size()>popsize && counter<std::pow(float(p.popsize),2))
	{
		for (int j=0;j<2;++j)
		{
			fitindex[j]=r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			fit[j] = pop[fitindex[j]].fitness;
			age[j] = pop[fitindex[j]].age;
		}
		if(fit[0]<=fit[1] && age[0]<=age[1])
		{
			if(age[0]<age[1] || fit[0]<fit[1])
			{
				draw=false;
				loser = fitindex[1];
			}
		}
		else if(fit[1]<=fit[0] && age[1]<=age[0]) //fit0 > fit1 or age0 >age1
		{
			if(age[1]<age[0] || fit[1]<fit[0])
			{
				draw=false;
				loser = fitindex[0];
			}
		}

		// delete losers from population
		if(!draw){
			pop[loser].swap(pop.back());
			pop.pop_back();
		}

		//minfit=p.max_fit;
		//minage=p.g;
		draw=true;
		++counter;
	}
	// if too many individuals in pop, remove based on age then fitness
	if (pop.size()>popsize)	{
		// cout << "counter reached: " << counter << " \n";
		sort(pop.begin(),pop.end(),SortAge());
		stable_sort(pop.begin(),pop.end(),SortFit());

		while(pop.size()>popsize){
			pop.pop_back();
		}
	}

}
