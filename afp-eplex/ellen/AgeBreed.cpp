#include "stdafx.h"

#include "data.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include "state.h"
#include "pareto.h"
#include "general_fns.h"
#include "FitnessEstimator.h"
#include "Generationfns.h"
#include "Fitness.h"
#include "EpiMut.h"

void AgeBreed(vector<ind>& pop,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE)
{
	//boost::progress_timer timer;
	float choice;
	
	vector <ind> tmppop;
	///if(!p.parallel)
	//{
	unsigned int numits=0;
	std::random_shuffle(pop.begin(),pop.end(),r[omp_get_thread_num()]);
	int counter=0;
	//increment age of each individual
	for (size_t i = 0; i < pop.size(); ++i)
		++pop[i].age;		
	

	while(tmppop.size()<pop.size() && counter<100000)
	{
		
		choice = r[omp_get_thread_num()].rnd_flt(0,1);

		if (choice<p.rep_wheel[1] && pop.size()>numits+1) // crossover, as long as there are at least two parents left
		{			
			// only breed if parents have different fitness 
			if(pop.at(numits).fitness!=pop.at(numits+1).fitness) 
			{				
				//cross parents to make two kids and push kids into tmppop
				Crossover(pop.at(numits),pop.at(numits+1),tmppop,p,r);
				//update ages
				tmppop.back().age=max(pop.at(numits).age,pop.at(numits+1).age);
				tmppop.at(tmppop.size()-2).age=max(pop.at(numits).age,pop.at(numits+1).age);
				// note: if these parents are picked multiple times, their age will increase by more than one per generation!
				//++pop.at(numits).age;
				//++pop.at(numits+1).age;
				
				numits+=2;
				
			}
			else
			{
				Mutate(pop.at(numits),tmppop,p,r,d);
				Mutate(pop.at(numits+1),tmppop,p,r,d);
				//update ages
				tmppop.at(tmppop.size()-2).age = pop.at(numits).age;
				tmppop.back().age = pop.at(numits+1).age;
				
				//++pop.at(numits).age;
				//++pop.at(numits+1).age;

				numits+=2;
			}
		}
		else if (choice < p.rep_wheel[2]) //mutation
		{
			Mutate(pop.at(numits),tmppop,p,r,d);
			//update ages
			tmppop.back().age=pop.at(numits).age;
			//++pop.at(numits).age;

			++numits;
		}
		counter++;
	}
	if (counter==100000)
		cout << "stuck in while loop in AgeBreed.cpp\n";

	while(tmppop.size()>pop.size())
		tmppop.erase(tmppop.end()-1);

	// epigenetic mutation
	if (p.eHC_on && p.eHC_mut){ 
		//s.out << "Applying epigenetic mutations...\n";
		for (int i = 0; i<tmppop.size(); ++i)
			EpiMut(tmppop.at(i),p,r);
	}
	//get tmppop fitness 
	Fitness(tmppop,p,d,s,FE);
	// genetic stats
	s.setCrossPct(tmppop);
	//add tmppop to pop
	pop.insert(pop.end(),tmppop.begin(),tmppop.end());
	tmppop.clear();
}
