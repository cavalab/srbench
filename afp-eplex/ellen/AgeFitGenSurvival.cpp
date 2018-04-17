#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"

void AgeFitGenSurvival(vector<ind>& pop,params& p,vector<Randclass>& r)
{
	//boost::progress_timer timer;
	vector<float> fit(2);
	vector<int> age(2);
	vector<int> fitindex(2);
	vector<int> genty(2);
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
			fitindex.at(j)=r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			fit.at(j) = pop.at(fitindex.at(j)).fitness;
			age.at(j) = pop.at(fitindex.at(j)).age;
			genty.at(j) = pop.at(fitindex.at(j)).genty;
		}
		if(fit[0]<=fit[1] && age[0]<=age[1] && genty[0]<=genty[1])
		{
			if(age[0]<age[1] || fit[0]<fit[1] || genty[0]<genty[1])
			{
				draw=false;
				loser = fitindex[1];
			}
		}
		else if(fit[1]<=fit[0] && age[1]<=age[0]) //fit0 > fit1 or age0 >age1
		{
			if(age[1]<age[0] || fit[1]<fit[0] || genty[1]<genty[0])
			{
				draw=false;
				loser = fitindex[0];
			}
		}

		// delete losers from population
		if(!draw){
			pop.at(loser).swap(pop.back());
			//std::swap(pop.at(loser),pop.back());
			pop.pop_back();
			//pop.erase(pop.begin()+loser);
		}

		//minfit=p.max_fit;
		//minage=p.g;
		draw=true;
		counter++;
	}
	if (pop.size()>popsize)
	{
		sort(pop.begin(),pop.end(),SortAge());
		stable_sort(pop.begin(),pop.end(),SortGenty());
		stable_sort(pop.begin(),pop.end(),SortFit());
	}
	while(pop.size()>popsize)
		pop.pop_back();
	
	

}