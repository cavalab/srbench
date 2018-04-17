#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "state.h"
#include "data.h"
#include "FitnessEstimator.h"
#include "Fitness.h"

#include "general_fns.h"
// parameter hill climber 
void HillClimb(ind& oldind,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE)
{
	//#pragma omp parallel for
	//for (int i=0; i<pop.size(); ++i) // for each individual
	//	{
			vector<ind> tmp_ind(1);
			//tmp_ind[0] = oldind; 
			//makenew(tmp_ind[0]);
			//tmp_ind[0].clrPhen(); // clear phenotype

			bool updated=false;
			int HCits; 
			//if (oldind.corr >= 0.999)
			//	HCits = 10;
			//else
				HCits = p.pHC_its;

			for (int j=0;j<HCits; ++j) // for number of specified iterations
			{
				//if (updated)
				//{
				//    tmp_ind[0] = oldind;  
				//	//makenew(tmp_ind[0]);
				//	tmp_ind[0].clrPhen(); // clear phenotype
				//}
				tmp_ind.resize(0);
				tmp_ind.push_back(oldind);
				tmp_ind[0].clrPhen();
				for (int h= 0; h<tmp_ind[0].line.size();++h) // for length of genotype
				{
					if(tmp_ind[0].line.at(h).type=='n' && tmp_ind[0].line.at(h).on) // insert function with true epiline value
					{
							/*float num = static_pointer_cast<n_num>(tmp_ind[0].line.at(h)).value;*/
							float num = tmp_ind[0].line.at(h).value;
							num = num + r[omp_get_thread_num()].gasdev()*num/10; // 10% gaussian noise perturbation
							if (p.ERCints)
								tmp_ind[0].line.at(h).value = int(num);
							else
								tmp_ind[0].line.at(h).value = num;
					}
				}
				//Gen2Phen(tmp_ind,p);
				Fitness(tmp_ind,p,d,s,FE); //get fitness 
				if ( tmp_ind[0].fitness < oldind.fitness) // if fitness is better, replace individual
				{
					oldind = tmp_ind[0];
					//swap(oldind,tmp_ind[0]);
				    //tmp_ind.clear();
					updated = true;
					s.pHC_updates[omp_get_thread_num()]++;
				}
				//else if (tmp_ind[0].fitness == oldind.fitness && tmp_ind[0].eqn.size() < oldind.eqn.size()) // if fitness is same but equation is smaller, replace individual
				//{
				//	oldind = tmp_ind[0];
				//	//tmp_ind.clear();
				//	updated = true;
				//	s.pHC_updates[omp_get_thread_num()]++;
				//}
				else
					updated = false;
			}

			tmp_ind.clear();
		//}
}
