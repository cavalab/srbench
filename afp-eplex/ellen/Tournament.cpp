#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"

void Tournament(const vector<ind>& pop,vector<unsigned int>& parloc,params& p,vector<Randclass>& r)
{
	//boost::progress_timer timer;
	vector<float> fitcompare(p.tourn_size);
	vector<int> fitindex(p.tourn_size);
	float minfit=p.max_fit;
	int winner;
	bool draw=true;
	for (int i=0;i<pop.size();++i)
	{
		for (int j=0;j<p.tourn_size;++j)
		{
			fitindex.at(j)=r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			fitcompare.at(j) = pop.at(fitindex.at(j)).fitness;
			if (fitcompare.at(j)<minfit)
			{
				minfit=fitcompare.at(j);
				winner = fitindex.at(j);
				draw=false;
			}
		}
		//get lowest fitness, return pop.at(fitindex.at(lowest)).id
		if(draw)
			parloc.at(i)=fitindex.at(r[omp_get_thread_num()].rnd_int(0,p.tourn_size-1));
		else
			parloc.at(i)=winner;

		minfit=p.max_fit;
		draw=true;
	}
	

}