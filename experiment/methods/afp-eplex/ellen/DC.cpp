#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "state.h"
#include "rnd.h"
#include "data.h"
#include "FitnessEstimator.h"
#include "Generationfns.h"
#include "strdist.h"
#include "Fitness.h"
#include "general_fns.h"
#include "EpiMut.h"

// steady-state deterministic crowding algorithm.
// update_locs are the indices of the update locations for passage to hill climbing. 
using std::swap;
void DC(vector<ind>& pop,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE)
{
	
	vector <ind> tmppop;
	float choice = r[omp_get_thread_num()].rnd_flt(0,1);
	int p1=0;
	int p2=0;

	if (choice<p.rep_wheel[1]) // crossover, as long as there are at least two parents left
	{			
		p1 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
		p2 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
		// only breed if parents have different fitness 
		if(pop.at(p1).fitness!=pop.at(p2).fitness) 
		{				
			//cross parents to make two kids
			Crossover(pop.at(p1),pop.at(p2),tmppop,p,r);
		}
		else
		{
			Mutate(pop.at(p1),tmppop,p,r,d);
			Mutate(pop.at(p2),tmppop,p,r,d);
		}
	}
	else if (choice < p.rep_wheel[2]) //mutation
	{
		p1 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
		Mutate(pop.at(p1),tmppop,p,r,d);
	}
	//if (p.loud ) cout << "  Fitness...\n";

	if (p.eHC_on && p.eHC_mut){ // epigenetic mutation
		for (int i = 0; i<tmppop.size(); ++i)
			EpiMut(tmppop.at(i),p,r);
	}

	Fitness(tmppop,p,d,s,FE);
	// look at parents and children equations
	
	string par1eqn = pop.at(p1).eqn;
	float par1fit = pop.at(p1).fitness;
	string kid1eqn = tmppop[0].eqn;
	float kid1fit = tmppop[0].fitness;
	string par2eqn;
	float par2fit;
	string kid2eqn; 
	float kid2fit; 
	if (tmppop.size()==2){
			par2eqn = pop.at(p2).eqn;
			par2fit = pop.at(p2).fitness;
			kid2eqn = tmppop[1].eqn;
			kid2fit = tmppop[1].fitness;
	}


	if(tmppop.size()==1)
	{
		if ( tmppop.at(0).fitness <= pop.at(p1).fitness )
		{
			if( tmppop.at(0).fitness < pop.at(p1).fitness )
				s.good_cross[omp_get_thread_num()]=s.good_cross[omp_get_thread_num()]+1;
			else
				s.neut_cross[omp_get_thread_num()]=s.neut_cross[omp_get_thread_num()]+1;
			
			//makenew(tmppop.at(0));
			//swap(tmppop.at(0),pop.at(p1));
			pop.at(p1).swap(tmppop.at(0));
		}
		else
			s.bad_cross[omp_get_thread_num()]=s.bad_cross[omp_get_thread_num()]+1;

	}
	else if (tmppop.size()==2)
	{
		if (strdist(tmppop.at(0).eqn_form,pop.at(p1).eqn_form) + strdist(tmppop.at(1).eqn_form,pop.at(p2).eqn_form) > strdist(tmppop.at(1).eqn_form,pop.at(p1).eqn_form) + strdist(tmppop.at(0).eqn_form,pop.at(p2).eqn_form))
		{	
			//swap(tmppop.at(0),tmppop.at(1));
			tmppop.at(0).swap(tmppop.at(1));
			tmppop.at(0).parentfitness = pop.at(p1).fitness;
			tmppop.at(1).parentfitness = pop.at(p2).fitness;
		}

		if ( tmppop.at(0).fitness <= pop.at(p1).fitness )
		{
			if( tmppop.at(0).fitness < pop.at(p1).fitness )
				s.good_cross[omp_get_thread_num()]=s.good_cross[omp_get_thread_num()]+1;
			else
				s.neut_cross[omp_get_thread_num()]=s.neut_cross[omp_get_thread_num()]+1;
			//makenew(tmppop.at(0));
			//swap(tmppop.at(0),pop.at(p1));
			pop.at(p1).swap(tmppop.at(0));
		}
		else
			s.bad_cross[omp_get_thread_num()]=s.bad_cross[omp_get_thread_num()]+1;

		if ( tmppop.at(1).fitness <= pop.at(p2).fitness )
		{
			if( tmppop.at(1).fitness < pop.at(p2).fitness )
				s.good_cross[omp_get_thread_num()]=s.good_cross[omp_get_thread_num()]+1;
			else
				s.neut_cross[omp_get_thread_num()]=s.neut_cross[omp_get_thread_num()]+1;
			//makenew(tmppop.at(0));
			//swap(tmppop.at(1),pop.at(p2));
			pop.at(p2).swap(tmppop.at(1));
		}
		else
			s.bad_cross[omp_get_thread_num()]=s.bad_cross[omp_get_thread_num()]+1;
	}
	else
	{
		//cout << "Problem in DC. tmppop size: " << tmppop.size() << endl;
	}
	tmppop.clear();
	
	/*for (int i = 0; i<pop.size();++i)
	{
		int totalshares = 0;
		for (int j=0;j<pop.at(i).line.size();++j)
			totalshares+=pop.at(i).line.at(j).use_count();

		if (totalshares>pop.at(i).line.size())
			cout << "shares exceeded\n";
	}*/

}