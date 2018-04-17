#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include "state.h"
#include "FitnessEstimator.h"
#include "Fitness.h"
#include "general_fns.h"

//class FitnessEstimator{
//public:
//	vector <int> FEpts; // points of fitness estimation (subset from data vals)
//	vector <float> TrainerFit; // fitness on trainer population
//	float fitness; // self fitness across trainers
//
//	FitnessEstimator(int length,vector<Randclass>& r,Data& d)
//	{
//		for (int i=0;i<length;++i){
//			FEpts.push_back(r[omp_get_thread_num()].rnd_int(0,d.vals.size()));
//		}
//	}
//
//	
//};
struct SortFE{
	bool operator() (const FitnessEstimator& i,const FitnessEstimator& j) { return (i.fitness<j.fitness);} 
};
struct SortGE{
	bool operator() (const FitnessEstimator& i,const FitnessEstimator& j) { return (i.genty<j.genty);} 
};
void setFEvals(vector<vector<float>>& FEvals, vector<float>& FEtarget,FitnessEstimator& FE, Data& d)
{
	FEvals.resize(0); FEvals.reserve(FE.FEpts.size());
	FEtarget.resize(0); FEtarget.reserve(FE.FEpts.size());
	for (int i=0;i<FE.FEpts.size();++i)
	{
		FEvals.push_back(d.vals[FE.FEpts[i]]);
		FEtarget.push_back(d.target[FE.FEpts[i]]);
	}
};
void FitnessFE(vector <FitnessEstimator>& FE, vector <ind>& trainers,params& p,Data& d,state& s)
{
	// calculate fitness of each Fitness Estimator on each trainer. 
	/*vector<float> exactfit(trainers.size());
		for (int j=0;j<trainers.size();++j)
		exactfit[j]=trainers[j].fitness;*/

	//get trainer exact fitness
	// trainer fitness must be calculated before this function is called
	vector<vector<float>> FEvals;
	vector<float> FEtarget;
	int ndata_t;
	for (int i=0;i<FE.size();++i){
		
		setFEvals(FEvals,FEtarget,FE[i],d);

		if(p.train || p.estimate_generality)
			ndata_t = FEvals.size()*p.train_pct;
		else
			ndata_t = FEvals.size();

		vector<float> FEness(trainers.size()); 

		for (int j=0;j<trainers.size();++j){

		if (!trainers[j].output.empty()){

			float abserror=0;
			float sq_error = 0;
			float meantarget = 0;
			float meanout = 0;
			float corr;
			float vaf;

			float target_std;
			vector<float> tmpoutput;
			for(unsigned int sim=0;sim<ndata_t;++sim)
			{	
				abserror += abs(FEtarget[sim]-trainers[j].output[FE[i].FEpts[sim]]);
				sq_error += pow(FEtarget[sim] - trainers[j].output[FE[i].FEpts[sim]],2); 
				meantarget += FEtarget.at(sim);
				meanout += trainers[j].output[FE[i].FEpts[sim]];
				tmpoutput.push_back(trainers[j].output[FE[i].FEpts[sim]]);
			}
				
				// mean absolute error
				abserror = abserror/ndata_t;
				sq_error /= ndata_t;
				meantarget = meantarget/ndata_t;
				meanout = meanout/ndata_t;
				//calculate correlation coefficient
				corr = getCorr(tmpoutput,FEtarget,meanout,meantarget,0,target_std);
				vaf = VAF(tmpoutput,FEtarget,meantarget,0);
				tmpoutput.clear();

				if(corr < p.min_fit)
					corr=p.min_fit;

				if(trainers[j].output.empty()) 
					FEness[j]=p.max_fit;
				/*else if (*std::max_element(pop.at(count).output.begin(),pop.at(count).output.end())==*std::min_element(pop.at(count).output.begin(),pop.at(count).output.end()))
					pop.at(count).fitness=p.max_fit;*/
				else if ( boost::math::isnan(abserror) || boost::math::isinf(abserror) || boost::math::isnan(corr) || boost::math::isinf(corr))
					FEness[j]=p.max_fit;
				else{
					if (!(p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0))
						FEness[j] = abserror;
					else if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0)
						FEness[j] = 1-corr;
					else if (p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0)
						FEness[j] = abserror/corr;
					else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0)
						FEness[j] = 1-vaf;
					else if (p.fit_type.compare("MSE")==0)
						FEness[j] = sq_error;
					if (p.norm_error)
						FEness[j] = FEness[j]/target_std;
				}

				if(FEness[j]>p.max_fit)
					FEness[j]=p.max_fit;
				else if(FEness[j]<p.min_fit)
					(FEness[j]=p.min_fit); 

		//Fitness(trainers,p,d,s,FE[i]); //note: FE[0] does not get used here
		if (!p.FE_rank)
			FE[i].fitness+=abs(trainers[j].fitness-FEness[j]);
		
		}
		}
		if (!p.FE_rank)
			FE[i].fitness=FE[i].fitness/trainers.size();
		else{
			// use FE ranking ability to assign fitness
			FE[i].fitness=0;
			
			float nsq = float(trainers.size()*trainers.size());
			for (int h=0;h<trainers.size();++h){
				for (int k=0;k<trainers.size();++k){
					if ((FEness[h] < FEness[k] && trainers[h].fitness > trainers[k].fitness) || 
						(FEness[h] > FEness[k] && trainers[h].fitness < trainers[k].fitness) )
						FE[i].fitness+=1;
				}
			}
			FE[i].fitness /= nsq;
		}
		/*==========================================================================
		                                    Generality Estimator
		==========================================================================*/

		if (p.estimate_generality){ // get generality using the second half of FEpts
			
			int ndata = FEvals.size();
			int ndata_v = ndata - ndata_t;

			vector<float> FEness_v(trainers.size()); 

			for (int j=0;j<trainers.size();++j){

			if (!trainers[j].output.empty()){

				float abserror_v=0;
				float sq_error_v = 0;
				float meantarget_v = 0;
				float meanout_v = 0;
				float corr_v;
				float vaf_v;

				float target_std_v;
				vector<float> tmpoutput_v;
				for(unsigned int sim=ndata_t;sim<ndata;++sim)
				{	
					abserror_v += abs(FEtarget[sim]-trainers[j].output[FE[i].FEpts[sim]]);
					sq_error_v += pow(FEtarget[sim] - trainers[j].output[FE[i].FEpts[sim]],2);
					meantarget_v += FEtarget.at(sim);
					meanout_v += trainers[j].output[FE[i].FEpts[sim]];
					tmpoutput_v.push_back(trainers[j].output[FE[i].FEpts[sim]]);
				}
				
					// mean absolute error
					abserror_v = abserror_v/ndata_v;
					sq_error_v /= ndata_v;
					meantarget_v = meantarget_v/ndata_v;
					meanout_v = meanout_v/ndata_v;
					//calculate correlation coefficient
					corr_v = getCorr(tmpoutput_v,FEtarget,meanout_v,meantarget_v,ndata_t,target_std_v);
					vaf_v = VAF(tmpoutput_v,FEtarget,meantarget_v,ndata_t);
					tmpoutput_v.clear();

					if(corr_v < p.min_fit)
						corr_v=p.min_fit;

					if(trainers[j].output.empty()) 
						FEness_v[j]=p.max_fit;
					else if ( boost::math::isnan(abserror_v) || boost::math::isinf(abserror_v) || boost::math::isnan(corr_v) || boost::math::isinf(corr_v))
						FEness_v[j]=p.max_fit;
					else{
						if (!(p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0))
							FEness_v[j] = abserror_v;
						else if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0)
							FEness_v[j] = 1-corr_v;
						else if (p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0)
							FEness_v[j] = abserror_v/corr_v;
						else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0)
							FEness_v[j] = 1-vaf_v;
						else if (p.fit_type.compare("MSE")==0)
							FEness_v[j] = sq_error_v;
						if (p.norm_error)
							FEness_v[j] = FEness_v[j]/target_std_v;
					}

					if(FEness_v[j]>p.max_fit)
						FEness_v[j]=p.max_fit;
					else if(FEness_v[j]<p.min_fit)
						(FEness_v[j]=p.min_fit); 

			//Fitness(trainers,p,d,s,FE[i]); //note: FE[0] does not get used here
			if (!p.FE_rank)
				FE[i].genty+=abs(trainers[j].genty - abs(FEness[j]-FEness_v[j])/FEness[j]);
		
			}
			}
			if (!p.FE_rank)
				FE[i].genty=FE[i].fitness/trainers.size();
			else{
				// use FE ranking ability to assign fitness
				FE[i].genty=0;
				float FE1_genty=0, FE2_genty=0; 
				float nsq = float(trainers.size()*trainers.size());
				for (int h=0;h<trainers.size();++h){
					FE1_genty = abs(FEness[h]-FEness_v[h])/FEness[h];
					for (int k=0;k<trainers.size();++k){
						FE2_genty = abs(FEness[k]-FEness_v[k])/FEness[k];
						if ((FE1_genty < FE2_genty && trainers[h].genty > trainers[k].genty) || 
							(FE1_genty > FE2_genty && trainers[h].genty < trainers[k].genty) )
							FE[i].genty+=1;
					}
				}
				FE[i].genty /= nsq;
			}

		}
	}
	
};
void InitPopFE(vector <FitnessEstimator>& FE,vector<ind> &pop,vector<ind>& trainers,params p,vector<Randclass>& r,Data& d,state& s)
{
	//vector <FitnessEstimator> FE; //fitness estimator population 
	//initialize estimator population
	FE.resize(0);
	FE.reserve(p.FE_pop_size);
	bool train = p.train;// || p.estimate_generality;
	for (int i=0;i<p.FE_pop_size;++i){
		FE.push_back(FitnessEstimator(p.FE_ind_size,r,d,train));			
	}
	
	//initialize trainer population
	trainers.reserve(p.FE_train_size);
	for (int i=0;i<p.FE_train_size;++i){
		trainers.push_back(pop[r[omp_get_thread_num()].rnd_int(0,pop.size()-1)]);
		//calc trainer fitness
	}
	//p.EstimateFitness = 0; // turn fitness estimation off (all data points used)
	//Fitness(trainers,p,d,s,FE[0]); //note: FE[0] does not get used here
	//p.EstimateFitness = 1; // turn fitness estimation back on
	//evaluate fitness of estimators
	p.EstimateFitness = 0; // turn fitness estimation off
	Fitness(trainers,p,d,s,FE[0]); //note: FE[0] does not get used here
	p.EstimateFitness = 1; // turn fitness estimation back on

	FitnessFE(FE,trainers,p,d,s);
	//perform selection
	if (p.estimate_generality){
		sort(FE.begin(),FE.end(),SortGE());
		stable_sort(FE.begin(),FE.end(),SortFE());
	}
	else
		sort(FE.begin(),FE.end(),SortFE());
};


void PickTrainers(vector<ind> pop, vector <FitnessEstimator>& FE,vector <ind>& trainers,params p,Data& d,state& s)
{
	vector <vector<float>> FEfits; //rows: predictors, cols: population
	vector<float> meanfits(pop.size());
	vector <float> varfits(pop.size());

	vector <vector<float>> FEgenty; //rows: predictors, cols: population
	vector<float> meangenty(pop.size());
	vector <float> vargenty(pop.size());
	//vector <ind> tmppop(1);
	// get FE fitness on each individual in population
	//vector<ind> tmppop;

	//for (int j=0;j<pop.size();++j){
	//	if(pop[j].fitness<p.max_fit){
	//		tmppop.push_back(pop[j]);
	//		//makenewcopy(pop[j],tmppop.back());
	//		if(tmppop.size()!=pop.size())
	//			tmppop.back().clrPhen();
	//	}
	//}
	//i: fitness predictors
	//j: solution population 
	for (int i=0;i<FE.size();++i){

		Fitness(pop,p,d,s,FE[i]);	

		if (p.estimate_generality){
			FEgenty.push_back(vector<float>(pop.size()));
			for (int j=0;j<pop.size();++j){
				FEgenty[i][j]=pop[j].genty;	
				/*if(i!=FE.size()-1)
					pop[j].clrPhen();*/
			}
		}

		FEfits.push_back(vector<float>(pop.size()));
		for (int j=0;j<pop.size();++j){
				FEfits[i][j]=pop[j].fitness;	
				if(i!=FE.size()-1)
					pop[j].clrPhen();
		}
		

		// convert FEfits to ranks
		/*for (int h=0;h<pop.size();++h){
			float maxfit = *max_element(FEfits[i].begin(),FEfits[i].end());
			FEfits[i][h] = FEfits[i][h]/maxfit*pop.size();
		}*/
	}
	float tmp;
	// calculate variance in fitness estimates
	for (int j=0;j<pop.size();++j){
		for (int i=0;i<FE.size();++i)
			meanfits[j]+=FEfits[i][j];

		meanfits[j]=meanfits[j]/FE.size(); // mean fitness over predictors
		
		for (int h=0;h<FE.size();++h){
			tmp = FEfits[h][j]-meanfits[j];
			varfits[j]+=pow(FEfits[h][j]-meanfits[j],2);
		}
		
		varfits[j]=varfits[j]/(FE.size()-1); // variance in fitness over predictors
		
		if(boost::math::isinf(varfits[j])) varfits[j]=0;
		
		pop[j].FEvar=varfits[j];

	}
	// calculate variance in generality estimates
	if(p.estimate_generality){
		for (int j=0;j<pop.size();++j){
			for (int i=0;i<FE.size();++i)
				meangenty[j]+=FEgenty[i][j];
			
			meangenty[j]=meangenty[j]/FE.size(); // mean generality over predictors

			for (int h=0;h<FE.size();++h){
				tmp = FEgenty[h][j]-meangenty[j];
				vargenty[j]+=pow(FEgenty[h][j]-meangenty[j],2);
			}
			
			vargenty[j]=vargenty[j]/(FE.size()-1); // variance in generality over predictors

			if(boost::math::isinf(vargenty[j])) vargenty[j]=0;

			pop[j].GEvar=vargenty[j];
		}
	}

	sort(pop.begin(),pop.end(),SortFEVar());

	//vector<ind>::iterator it = pop.begin();
	trainers.clear();
	int q=0;
	int train_size;
	
	if (p.estimate_generality) train_size = ceil(float(p.FE_train_size/2));
	else train_size = p.FE_train_size;

	while(trainers.size()<train_size && q<pop.size()){
		if(q==0){
			trainers.push_back(pop.at(q));
		}
		else if (pop.at(q).eqn.compare(trainers.back().eqn)!=0){
			trainers.push_back(pop.at(q));
		}
		q++;
	}
	q=0;
	if (p.estimate_generality){ //pick the second half of trainers based on generality variance
		sort(pop.begin(),pop.end(),SortGEVar());
		while(trainers.size()<p.FE_train_size && q<pop.size()){
			if (pop.at(q).eqn.compare(trainers.back().eqn)!=0)
				trainers.push_back(pop.at(q));
			q++;
		}
	}
	while(trainers.size()<p.FE_train_size){
		trainers.push_back(pop.at(q));
		q++;
	}
	//trainers.assign(tmppop.begin(),tmppop.begin()+p.FE_train_size);
	p.EstimateFitness = 0; // turn fitness estimation off
	Fitness(trainers,p,d,s,FE[0]); //note: FE[0] does not get used here
	p.EstimateFitness = 1; // turn fitness estimation back on
	//evaluate fitness of estimators
	FitnessFE(FE,trainers,p,d,s);
	// sort estimators
	if (p.estimate_generality){
		sort(FE.begin(),FE.end(),SortGE());
		stable_sort(FE.begin(),FE.end(),SortFE());
	}
	else
		sort(FE.begin(),FE.end(),SortFE());
};
void crossFE(vector <FitnessEstimator>& FE,params& p, vector<Randclass>& r)
{
	// select parents (tournament)
	int nt = 2;
	vector<int> parents;
	vector<int> choices(2);
	vector <FitnessEstimator> newFE;
	float bestfit, bestgenty; 
	int bestchoice;
	for (int i=0; i < FE.size(); ++i){
		
		for (int j=0; j<nt; ++j){
			
			choices[j] = r[omp_get_thread_num()].rnd_int(0,FE.size()-1);
			if (j==0) {
				bestfit = FE[choices[0]].fitness;
				bestgenty = FE[choices[0]].genty;
				bestchoice = choices[0];
			}
			else if(p.estimate_generality){
				if (r[omp_get_thread_num()].rnd_flt(0,1) < 0.5){
					if(FE[choices[j]].fitness<bestfit)
					{
						bestfit = FE[choices[j]].fitness;
						bestgenty = FE[choices[0]].genty;
						bestchoice = choices[j];
					}
				}
				else{
					if(FE[choices[j]].genty<bestgenty)
					{
						bestfit = FE[choices[j]].fitness;
						bestgenty = FE[choices[j]].genty;
						bestchoice = choices[j];
					}
				}
			}
			else if(FE[choices[j]].fitness<bestfit)
			{
				bestfit = FE[choices[j]].fitness;
				bestchoice = choices[j];
			}
		}
		parents.push_back(bestchoice);		
	}

	//// new pop
	newFE.resize(FE.size());
	for (int i=0;i<parents.size();++i){
		//newFE[i]= FE[parents[i]];	
		newFE[i].FEpts.resize(FE[i].FEpts.size());
	}
	//cross parents
//	int p1,p2;
//	int crosspt;
	for (int i=0; i < FE.size()-1; ++i){
		int point1 = r[omp_get_thread_num()].rnd_int(0,FE[0].FEpts.size()-1);
		
		newFE[i].FEpts.assign(FE[parents[i]].FEpts.begin(),FE[parents[i]].FEpts.begin()+point1);
		newFE[i].FEpts.insert(newFE[i].FEpts.end(),FE[parents[i+1]].FEpts.begin()+point1,FE[parents[i+1]].FEpts.end());

		newFE[i+1].FEpts.assign(FE[parents[i+1]].FEpts.begin(),FE[parents[i+1]].FEpts.begin()+point1);
		newFE[i+1].FEpts.insert(newFE[i+1].FEpts.end(),FE[parents[i]].FEpts.begin()+point1,FE[parents[i]].FEpts.end());

		++i;
	}
	FE = newFE;
	// keep strictly the best predictors	
	/*FE.insert(FE.end(),newFE.begin(),newFE.end());
	sort(FE.begin(),FE.end(),SortFE());
	FE.erase(FE.begin()+newFE.size(),FE.end());*/
};
void mutateFE(vector <FitnessEstimator>& FE,params p,Data& d,vector<Randclass>& r)
{
	int pt;
	int lastpt;
	if (p.train) lastpt = d.vals.size()/2-1;
	else lastpt = d.vals.size()-1;

	for (int i=0; i < FE.size(); ++i){
		pt = r[omp_get_thread_num()].rnd_int(0,FE[i].FEpts.size()-1);
		FE[i].FEpts[pt] = r[omp_get_thread_num()].rnd_int(0,lastpt);
	}
};
void EvolveFE(vector<ind> &pop, vector <FitnessEstimator>& FE,vector <ind>& trainers,params p,Data& d,state& s,vector<Randclass>& r) 
{
	vector <float> FEfitness(FE.size()); //fitness of Fitness Estimators
	
	vector <FitnessEstimator> newFE = FE;
	//cross estimators
	crossFE(newFE,p,r);

	//mutate estimators
	mutateFE(newFE,p,d,r);
	
	//evaluate fitness of estimators
	FitnessFE(newFE,trainers,p,d,s);

	// keep strictly the best predictors	
	/*FE.insert(FE.end(),newFE.begin(),newFE.end());
	sort(FE.begin(),FE.end(),SortFE());
	FE.erase(FE.begin()+newFE.size(),FE.end());*/

	//keep the elite predictor and the new predictors
	newFE.insert(newFE.end(),FE[0]);
	if (p.estimate_generality){
		sort(newFE.begin(),newFE.end(),SortGE());
		stable_sort(newFE.begin(),newFE.end(),SortFE());
	}
	else
		sort(newFE.begin(),newFE.end(),SortFE());

	FE.assign(newFE.begin(),newFE.end()-1);
	////perform selection
	//sort(FE.begin(),FE.end(),SortFE());
	
	/*//if time to add new fitness trainer
	PickTrainers(pop,FE,trainers,p,d,s);
	*/
};
//void EvolveGE(vector<ind> &pop, vector <FitnessEstimator>& FE,vector <ind>& trainers,params p,Data& d,state& s,vector<Randclass>& r)