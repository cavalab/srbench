#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "state.h"
#include "rnd.h"
#include "data.h"
#include "FitnessEstimator.h"
#include "Generationfns.h"
#include "Fitness.h"

float std_dev(vector<float>& x) {

	// get mean of x
	float mean = std::accumulate(x.begin(), x.end(), 0.0)/x.size();
	//calculate variance
	float var = 0;
	for (vector<float>::iterator it = x.begin(); it != x.end(); ++it)
		var += pow(*it - mean,2);

	var /= (x.size()-1);
	return sqrt(var);

}
float median(vector<float> x) {
	sort(x.begin(), x.end());

	if (x.size() % 2 == 0) {
		float tmp1 = x[(x.size() / 2) - 1];
		float tmp2 = x[x.size() / 2];
		return (tmp1 + tmp2) / 2;
	}
	else
		return x[x.size() / 2];
}
float median(vector<int> x) {
	sort(x.begin(), x.end());

	if (x.size() % 2 == 0) {
		int tmp1 = x[(x.size() / 2) - 1];
		int tmp2 = x[x.size() / 2];
		return float((tmp1 + tmp2) / 2);
	}
	else
		return float(x[x.size() / 2]);
}
float mad(vector<float>& x) {
	// returns median absolute deviation (MAD)
	// get median of x
	float x_median = median(x);
  if (boost::math::isnan(x_median))
    return 0;
	//calculate absolute deviation from median
	vector<float> dev;
	for (vector<float>::iterator it = x.begin(); it != x.end(); ++it)
		dev.push_back(abs(*it - x_median));

  if (boost::math::isnan(median(dev)))
    return 0;

	return median(dev);

}
void LexicaseSelect(vector<ind>& pop,vector<unsigned int>& parloc,params& p,vector<Randclass>& r,Data& d,state& s)
{

	// add metacases if needed
	for (unsigned i = 0; i<p.lex_metacases.size(); ++i)
	{
		if (p.lex_metacases[i].compare("age")==0){
			for (unsigned j=0;j<pop.size(); ++j)
				pop[j].error.push_back(float(pop[j].age));
		}
		else if (p.lex_metacases[i].compare("complexity")==0){
			for (unsigned j=0;j<pop.size(); ++j)
				pop[j].error.push_back(float(pop[j].complexity));
		}
		else if (p.lex_metacases[i].compare("dimensionality") == 0) {
			for (unsigned j = 0; j<pop.size(); ++j)
				pop[j].error.push_back(float(pop[j].dim));
		}
	}


	//boost::progress_timer timer;
	//vector<float> fitcompare;
	vector<int> pool; // pool from which to choose parent. normally it is set to the whole population.
	int numcases;
	if (p.lex_class)
		numcases = p.number_of_classes + p.lex_metacases.size();
	else if (p.EstimateFitness)
		numcases = p.FE_ind_size*p.train_pct + p.lex_metacases.size();
	else
		numcases = d.vals.size()*p.train_pct + p.lex_metacases.size();

	if (p.lexpool==1){
		for (int i=0;i<pop.size();++i){
			if(pop[i].error.size() == numcases)
				pool.push_back(i);
		}
	}

	vector <int> starting_pool = pool;
	vector <int> case_order;
	for(int i=0;i<pop[pool[0]].error.size();++i)
		case_order.push_back(i);
	assert(case_order.size()>0);
	float minfit=p.max_fit;
	vector<int> winner;
	bool draw=true;
	bool pass;
	int h;
	// vector<int> num_passes(numcases-p.lex_metacases.size(),0);
	int tmp;
  vector<float> epsilon; // epsilon values for each case

	if (p.lex_eps_error_mad) { // errors within mad of the best error pass
									 // get minimum error on each case
		vector<float> min_error(numcases, p.max_fit);
		vector<float> mad_error(numcases);

		for (size_t i = 0; i < numcases; ++i) {
			vector<float> case_error(pool.size());
			for (size_t j = 0; j < pool.size(); ++j) {
				if (pop[pool[j]].error[i] < min_error[i])
					min_error[i] = pop[pool[j]].error[i];

				case_error[j] = pop[pool[j]].error[i];
			}
			mad_error[i] = mad(case_error);
		}
		if (p.lex_eps_global){ // if pass conditions defined relative to whole population (rather than selection pool):
				for (size_t i = 0; i < pool.size(); ++i) {
					// check if error is within epsilon
					for (size_t j = 0; j < numcases - p.lex_metacases.size(); ++j) {
						if (pop[pool[i]].error[j] <= min_error[j]+mad_error[j])
							pop[pool[i]].error[j] = 0;
						else
							pop[pool[i]].error[j] = 1;
					}
				}
		}
		else if (!p.lex_eps_dynamic){ // for non-global, non-dynamic version, save the epsilon values for each case
				for (size_t i = 0; i < numcases; ++i){
					assert(mad_error[i] >= 0);
					epsilon.push_back(mad_error[i]);
				}

		}
	}
	// measure median number of cases used
	vector<float> cases_used;
	// measure median pool size at selection
	vector<float> sel_size;

	// for each selection event:
	for (int i=0;i<parloc.size();++i)
	{
		//shuffle test cases
		std::random_shuffle(case_order.begin(),case_order.end(),r[omp_get_thread_num()]);

		// select a subset of the population if lexpool is being used
		if (p.lexpool!=1){
			for (int j=0;j<p.lexpool*pop.size();++j){
				tmp = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
				int n = 0;
				while (pop[tmp].error.empty() && n < pop.size()) {
					tmp = r[omp_get_thread_num()].rnd_int(0, pop.size() - 1);
					n += 1;
				}
				pool.push_back(tmp);
			}
		}
		else //otherwise use the whole population for selection
			pool = starting_pool;

		pass=true;
		h=0; // count the number of cases that are used
		while ( pass && h<case_order.size()) //while there are remaining cases and more than 1 unique choice
		{
			//reset winner and minfit for next case
			winner.resize(0);
			minfit=p.max_fit;

			if (p.lex_eps_global){
				// loop through the individuals and pick out the elites
				for (int j=0;j<pool.size();++j)
				{

					if (pop[pool[j]].error[case_order[h]]<minfit || j == 0)
					{
						minfit=pop[pool[j]].error[case_order[h]];
						winner.resize(0);
						winner.push_back(pool[j]);
					}
					else if (pop[pool[j]].error[case_order[h]]==minfit)
					{
						winner.push_back(pool[j]);
					}
				}
			}
			else{// epsilon lexicase with local pool pass conditions
				// pool for calculating mad on the fly if p.lex_eps_dynamic is on
				vector<float> pool_error(pool.size());
				float ep_dyn=1;
				// get best fitness in selection pool
				for (int j=0;j<pool.size();++j){
					if (pop[pool[j]].error[case_order[h]]<minfit || j == 0)
						minfit=pop[pool[j]].error[case_order[h]];
					if (p.lex_eps_dynamic) // calculate MAD dynamically
					{
						pool_error[j] = pop[pool[j]].error[case_order[h]];
						// i?? this should be getting the error for case_order[h]!
					}
				}
				if (p.lex_eps_dynamic){
          // get dynamic epsilon
					if (p.lex_eps_dynamic_rand){
						ep_dyn = pool_error[r[omp_get_thread_num()].rnd_int(0, pool_error.size() - 1)]-minfit;
						// cout << "ep_dyn: " << ep_dyn << "\n";
					}
					else{
						ep_dyn = mad(pool_error);
					}
					// lex eps madcap
					if (p.lex_eps_dynamic_madcap && r[omp_get_thread_num()].rnd_int(0,1)<0.5)
						ep_dyn = 0;
					// winners are within epsilon of the local pool's minimum fitness
					for (int j=0;j<pool.size();++j){
						if (pop[pool[j]].error[case_order[h]]<=minfit+ep_dyn)
							winner.push_back(pool[j]);
					}
				}
				else{

					// winners are within epsilon of the local pool's minimum fitness
					for (int j=0;j<pool.size();++j){
						if (pop[pool[j]].error[case_order[h]] <= minfit+epsilon[case_order[h]]){

							winner.push_back(pool[j]);
							// cout<< "winner pushed back\n";
						}
					}
				}
			}
			// if there is more than one elite individual and still more cases to consider
			if(winner.size()>1 && h<case_order.size()-1)
			{
				//check if winners are duplicates. if so, exit (pass = false).
				bool all_equal = true;
				for (auto w1 : winner){
					for (auto w2 : winner){
						all_equal = pop[w1].eqn.compare(pop[w2].eqn)==0;
						if (!all_equal)
							break;
					}
					if (!all_equal)
						break;
				}
				pass= !all_equal;
				// pass=true;
				++h;
				//reduce pool to elite individuals on case case_order[h]
				pool = winner;
			} // otherwise, a parent has been chosen or has to be chosen randomly from the remaining pool
			else
				pass=false;


		}
		//if more than one winner, pick randomly
		if(winner.size()>1)
			parloc[i]=winner[r[omp_get_thread_num()].rnd_int(0,winner.size()-1)];
		else if (winner.size()==1) // otherwise make the winner a parent
			parloc[i]=winner[0];
		else{ // otherwise throw an ??
			string print_at_once = "";
			print_at_once += "??";
			print_at_once += "winner pool is size " + to_string(winner.size()) + "\n";
			print_at_once += "pool size:" + to_string(pool.size()) + "\n";
			print_at_once += "minfit: " + to_string(minfit) + "\n";
			// if (p.lex_eps_dynamic)
			// 	print_at_once += "epsilon: " + to_string(ep_dyn) + "\n";
			// else
			print_at_once += "epsilon: " + to_string(epsilon[case_order[h]]) + "\n";
			print_at_once += "minfit + epsilon: " + to_string(minfit+epsilon[case_order[h]]) + "\n";
			print_at_once += "epsilon size:" + to_string(epsilon.size()) + "\n";
			// print_at_once += "epsilon:\n";
			// for (size_t i = 0; i< epsilon.size(); ++i)
			// 	print_at_once += to_string(epsilon[i]) + ",";

			print_at_once += "pool fitness cases:\n";
			print_at_once += "case:\t";
			for (size_t c =0; c<h; ++c){
				print_at_once += to_string(case_order[c]) + ",";
			}
			print_at_once += "\n individual fitnesses\n";
			for (size_t i=0; i<pool.size(); ++i){
				print_at_once += to_string(i) + ":\t";
				for (size_t c =0; c<h; ++c)
					print_at_once += to_string(pop[pool[i]].error[case_order[h]]) + ",";
				print_at_once += "\n";
			}

			cout << print_at_once;
		}

		assert(winner.size()>0);
		// reset minfit

		minfit=p.max_fit;


		cases_used.push_back(h+1);
		sel_size.push_back(winner.size());
		// if (h+1 == 216){
		// 	string print_at_once = "";
		// 	print_at_once += "winner size: " + to_string(winner.size()) + "\n";
		// 	print_at_once += "winners:\n";
		// 	for (auto w : winner){
		// 		print_at_once += pop[w].eqn + "\n";
		// 	}
		// 	print_at_once += "pool size: " + to_string(starting_pool.size()) + "\n";
		// 	print_at_once += "starting pool size: " + to_string(starting_pool.size()) + "\n";
		// 	print_at_once += "winner size: " + to_string(winner.size()) + "\n";
		// 	// cout << print_at_once;
		// }

	}//for (int i=0;i<parloc.size();++i)
  // cout << "cases used:";
	// for (auto cu: cases_used)
	//  	cout << cu << ",";
	// cout << "\n";
	// std::sort(cases_used.begin(), cases_used.end());
	// std::sort(sel_size.begin(), sel_size.end());

	// store median lex cases used, normalized by the number of cases
	s.median_lex_cases[omp_get_thread_num()] = median(cases_used)/case_order.size();

	// store median number of individuals that pass each case
	// s.median_passes_per_case[omp_get_thread_num()] = median(num_passes) / pop.size();

	// store median lex pool size at selection, normalized by the population size
	s.median_lex_pool[omp_get_thread_num()] = median(sel_size)/pop.size();

	//cout << "mean num cases used: " << hmean << "\n";

}
