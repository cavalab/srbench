#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "state.h"
#include "rnd.h"
#include "data.h"
#include "FitnessEstimator.h"
#include "Fitness.h"
#include "Generationfns.h"
#include "InitPop.h"
#include "EpiMut.h"
#include "ParetoSurvival.hpp"
#include "general_fns.h"

void Prune(vector<ind>& pop,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE)
{ // prune dimensions from programs with a hill climber
		
		sort(pop.begin(),pop.end(),SortFit());
		ind best = pop[0];
		bool updated=false;
		vector<unsigned> roots;
		find_root_nodes(best.line, roots);

		if (roots.size()>1){
			vector<ind> pruned; 
			int pt,begin,end;
			for (unsigned j=0; j<roots.size(); ++j){
				pruned.resize(0);
				pruned.push_back(best);
				pruned[0].clrPhen();

				pt = roots[j];
				end = pt;
				if (j==roots.size()-1)
					begin = 0;
				else
					begin = roots[j+1]+1;

				pruned[0].line.erase(pruned[0].line.begin()+begin,pruned[0].line.begin()+end+1);

				Fitness(pruned,p,d,s,FE);

				if (pruned[0].fitness <= best.fitness){
					best = pruned[0];
					updated=true;
				}
			}
		}
		if (updated)
			pop[0] = best;
	
}
