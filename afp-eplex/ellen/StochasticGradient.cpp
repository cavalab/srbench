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
void StochasticGradient(ind& oldind,params& p,vector<Randclass>& r,Data& d,state& s,
        FitnessEstimator& FE, int gen)
{
	//#pragma omp parallel for
	//for (int i=0; i<pop.size(); ++i) // for each individual
	//	{
			vector<ind> tmp_ind(1);
			//tmp_ind[0] = oldind; 
			//makenew(tmp_ind[0]);
			//tmp_ind[0].clrPhen(); // clear phenotype

			bool updated=false;
			int SG_its; 
			//if (oldind.corr >= 0.999)
			//	SG_its = 10;
			//else
	    	
            float lr = p.learning_rate/(1.0+float(gen));
            //int batch_size = std::max(1, int(d.target.size())/gen); 
            int batch_size = 1;
//			for (int j=0;j<p.SG_its; ++j) // for number of specified iterations
//			{
				//if (updated)
				//{
				//    tmp_ind[0] = oldind;  
				//	//makenew(tmp_ind[0]);
				//	tmp_ind[0].clrPhen(); // clear phenotype
				//}
				tmp_ind.resize(0);
				tmp_ind.push_back(oldind);
				tmp_ind[0].clrPhen();
                tmp_ind.push_back(oldind);
                tmp_ind[1].clrPhen();
				for (int h= 0; h<tmp_ind[0].line.size();++h) // for length of genotype
				{
					if(tmp_ind[0].line.at(h).type=='n' && tmp_ind[0].line.at(h).on) 
                    // insert function with true epiline value
					{
							/*float num = static_pointer_cast<n_num>(tmp_ind[0].line.at(h)).value;*/
							float w = tmp_ind[0].line.at(h).value;
    						// 10% gaussian noise perturbation
                            float dw = r[omp_get_thread_num()].gasdev()*w/10; 
                            
							tmp_ind[1].line[h].value = w + dw;
                            // pick batch for learning
                            FE = FitnessEstimator(batch_size,r,d,true);
                            // temporarily turn fitness estimation on 
                            p.EstimateFitness = true;
                            Fitness(tmp_ind,p,d,s,FE);
                            p.EstimateFitness = false;
                            
                            // get change in fitness 
                            float dF = tmp_ind[0].fitness - tmp_ind[1].fitness;
                            // update parameter as w = w - lr * dF/dw
                            w -= lr * dF / dw;
                            oldind.line[h].value = w;                         
					}
				}
			//	//Gen2Phen(tmp_ind,p);
			//	Fitness(tmp_ind,p,d,s,FE); //get fitness 
			//	if ( tmp_ind[0].fitness < oldind.fitness) // if fitness is better, replace individual
			//	{
			//		oldind = tmp_ind[0];
			//		//swap(oldind,tmp_ind[0]);
			//	    //tmp_ind.clear();
			//		updated = true;
			//		s.pHC_updates[omp_get_thread_num()]++;
			//	}
			//	//else if (tmp_ind[0].fitness == oldind.fitness && tmp_ind[0].eqn.size() 
            //    //< oldind.eqn.size()) 
            //    //// if fitness is same but equation is smaller, replace individual
			//	//{
			//	//	oldind = tmp_ind[0];
			//	//	//tmp_ind.clear();
			//	//	updated = true;
			//	//	s.pHC_updates[omp_get_thread_num()]++;
			//	//}
			//	else
			//		updated = false;
			//}

			tmp_ind.clear();
		//}
}
