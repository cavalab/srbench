#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "state.h"
#include "data.h"
#include "FitnessEstimator.h"
#include "Fitness.h"
#include "Gen2Phen.h"
#include "general_fns.h"

void EpiHC(ind& oldind,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE)
{
	////#pragma omp parallel for
	//for (int i=0; i<pop.size(); ++i) // for each individual
	//{
		
		vector<ind> tmp_ind;
		//tmp_ind[0] = oldind;
		//makenew(tmp_ind[0]);
		//tmp_ind[0].clrPhen();
		bool updated = false;
		int min_change;
		bool pass = true;
		int outstart = 0;
		int linestart = 0;
		vector<float> init_stack;
		for (int j=0;j<p.eHC_its; ++j) // for number of specified iterations
		{
		
			/* if (updated)
			{
				//tmp_ind.clear();
				tmp_ind.push_back(oldind); 
				//makenew(tmp_ind[0]);
				tmp_ind[0].clrPhen(); // clear phenotype
			} */
			// THIS NEEDS TO BE TESTED
			tmp_ind.resize(0);
			tmp_ind.push_back(oldind);
			tmp_ind[0].clrPhen();
			min_change = tmp_ind[0].line.size()+1;

			for(int h = tmp_ind[0].line.size()-1;h>=0;--h)
			{				
				if (r[omp_get_thread_num()].rnd_flt(0, 1) <= p.eHC_prob) {
					int pt1 = h;
					if (p.cross == 3) { // turn off a subtree
						int sum_arity = tmp_ind[0].line[pt1].arity_float;
						tmp_ind[0].line[pt1].on = !tmp_ind[0].line[pt1].on;
						while (sum_arity > 0 && pt1 > 0)
						{							
							--pt1;
							--sum_arity;
							sum_arity += tmp_ind[0].line[pt1].arity_float;
							tmp_ind[0].line[pt1].on = !tmp_ind[0].line[pt1].on;
						}	
					}
					else 
						tmp_ind[0].line.at(h).on = !tmp_ind[0].line.at(h).on;
					
					if (pt1 < min_change){	// determine active nodes occuring before change point		
						min_change = pt1;
						
					}
				}
				
			}	

			if (min_change==tmp_ind[0].line.size()+1){
				int tmp = r[omp_get_thread_num()].rnd_int(0,tmp_ind[0].line.size()-1);
				int pt1 = tmp;
				if (p.cross == 3) { // turn off a subtree
					int sum_arity = tmp_ind[0].line[pt1].arity_float;
					tmp_ind[0].line[pt1].on = !tmp_ind[0].line[pt1].on;
					while (sum_arity > 0 && pt1 > 0)
					{
						--pt1;
						--sum_arity;
						sum_arity += tmp_ind[0].line[pt1].arity_float;
						tmp_ind[0].line[pt1].on = !tmp_ind[0].line[pt1].on;
					}
				}
				else
					tmp_ind[0].line.at(tmp).on = !tmp_ind[0].line.at(tmp).on;

				linestart=pt1;
				for (int x = 0; x<tmp;x++)
					outstart+=(int)(tmp_ind[0].line.at(x).on);
			}
			else {
				linestart = min_change;
				for (int x = 0; x < linestart; ++x)
					outstart += (int)(tmp_ind[0].line[x].on);
			}

			//Gen2Phen(tmp_ind,p);
			//get fitness 
			if (p.eHC_slim){

				if (p.EstimateFitness) 
					tmp_ind[0].stack_float.resize(outstart*FE.FEpts.size());
				else 
					tmp_ind[0].stack_float.resize(outstart*d.vals.size());

				tmp_ind[0].stack_floatlen.resize(outstart);

				pass = SlimFitness(tmp_ind[0],p,d,s,FE,linestart,oldind.fitness);
			}
			else
				Fitness(tmp_ind,p,d,s,FE); 
			if(pass){
				if ( tmp_ind[0].fitness < oldind.fitness) // if fitness is better, replace individual
				{
					oldind = tmp_ind[0];
					updated = true;
					//tmp_ind.resize(0);
					++s.eHC_updates[omp_get_thread_num()];
				}
				else if (tmp_ind[0].fitness == oldind.fitness && tmp_ind[0].complexity < oldind.complexity) // if fitness is same but equation is smaller, replace individual
				{
					oldind = tmp_ind[0];
					updated = true;
					//tmp_ind.resize(0);
					++s.eHC_updates[omp_get_thread_num()];
					++s.eHC_ties[omp_get_thread_num()];
				}
				else
					updated = false;
			}
			else 
				pass = 1;

			//numchanges=0;
			outstart=0;
		}
		//tmp_ind.clear();
	//}

}