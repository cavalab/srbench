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
//	FitnessEstimator(int length,vector<Randclass>& r,data& d)
//	{
//		for (int i=0;i<length;++i){
//			FEpts.push_back(r[omp_get_thread_num()].rnd_int(0,d.vals.size()));
//		}
//	}
//
//	
//};
struct SortGE{
	bool operator() (const FitnessEstimator& i,const FitnessEstimator& j) { return (i.fitness<j.fitness);} 
};

void setGEvals(vector<vector<float>>& GEvals, vector<float>& GEtarget,FitnessEstimator& GE, data& d)
{
	GEvals.resize(0); GEvals.reserve(GE.FEpts.size());
	GEtarget.resize(0); GEtarget.reserve(GE.FEpts.size());
	for (int i=0;i<GE.FEpts.size();++i)
	{
		GEvals.push_back(d.vals[GE.FEpts[i]]);
		GEtarget.push_back(d.target[GE.FEpts[i]]);
	}
};
void FitnessGE(vector <FitnessEstimator>& GE, vector <ind>& trainers,params& p,data& d,state& s)
{
	// calculate fitness of each Fitness Estimator on each trainer. 
	/*vector<float> exactfit(trainers.size());
		for (int j=0;j<trainers.size();++j)
		exactfit[j]=trainers[j].fitness;*/

	//get trainer exact fitness
	// trainer fitness must be calculated before this function is called
	vector<vector<float>> GEvals;
	vector<float> GEtarget;
	int ndata_t, ndata;
	for (int i=0;i<GE.size();++i){
		
		setGEvals(GEvals,GEtarget,GE[i],d);

		ndata_t = GEvals.size()/2;
		ndata = GEvals.size();

		vector<float> GEness_t(trainers.size()), GEness_v(trainers.size()); 

		for (int j=0;j<trainers.size();++j){

			if (!trainers[j].output.empty()){

				float abserror_t=0, abserror_v=0;
				float meantarget_t = 0, meantarget_v=0;
				float meanout_t = 0, meanout_v = 0;
				float corr_t, corr_v;
				float vaf_t, vaf_v;

				float target_std_t,target_std_v;
				vector<float> tmpoutput_t, tmpoutput_v;
				for(unsigned int sim=0;sim<ndata;++sim)
				{	
					if (sim < ndata_t){

						abserror_t += abs(GEtarget[sim]-trainers[j].output[GE[i].FEpts[sim]]);
						meantarget_t += GEtarget.at(sim);
						meanout_t += trainers[j].output[GE[i].FEpts[sim]];
						tmpoutput_t.push_back(trainers[j].output[GE[i].FEpts[sim]]);
					}
					else{
						abserror_v += abs(GEtarget[sim]-trainers[j].output_v[GE[i].FEpts[sim]-ndata_t]);
						meantarget_v += GEtarget.at(sim);
						meanout_v += trainers[j].output_v[GE[i].FEpts[sim]-ndata_t];
						tmpoutput_v.push_back(trainers[j].output_v[GE[i].FEpts[sim]-ndata_t]);
					}
				}
				
					// mean absolute error
					abserror_t = abserror_t/ndata_t;
					abserror_v = abserror_v/(ndata-ndata_t);
					meantarget_t = meantarget_t/ndata_t;
					meantarget_v = meantarget_v/(ndata-ndata_t);
					meanout_t = meanout_t/ndata_t;
					meanout_v = meanout_v/(ndata-ndata_t);
					//calculate correlation coefficient
					corr_t = getCorr(tmpoutput_t,GEtarget,meanout_t,meantarget_t,0,target_std_t);
					corr_v = getCorr(tmpoutput_v,GEtarget,meanout_v,meantarget_v,ndata_t,target_std_v);
					//variance accounted for
					vaf_t = VAF(tmpoutput_t,GEtarget,meantarget_t,0);	
				    vaf_v = VAF(tmpoutput_v,GEtarget,meantarget_v,ndata_t);

					tmpoutput_t.clear();
					tmpoutput_v.clear();

					if(corr_t < p.min_fit)
						corr_t=p.min_fit;
					if(corr_v < p.min_fit)
						corr_v=p.min_fit;

					/*if(trainers[j].output.empty()){ 
						GEness_t[j]=p.max_fit;
						GEness_v[j]=p.max_fit;
					}*/

					/*else if (*std::max_element(pop.at(count).output.begin(),pop.at(count).output.end())==*std::min_element(pop.at(count).output.begin(),pop.at(count).output.end()))
						pop.at(count).fitness=p.max_fit;*/
					if ( boost::math::isnan(abserror_t) || boost::math::isinf(abserror_t) || boost::math::isnan(corr_t) || boost::math::isinf(corr_t))
						GEness_t[j]=p.max_fit;
					else{
						if (p.fit_type==1)
							GEness_t[j] = abserror_t;
						else if (p.fit_type==2)
							GEness_t[j] = 1-corr_t;
						else if (p.fit_type==3)
							GEness_t[j] = abserror_t/corr_t;
						else if (p.fit_type==4)
							GEness_t[j] = 1-vaf_t;
						if (p.norm_error)
							GEness_t[j] = GEness_t[j]/target_std_t;
					}

					if(GEness_t[j]>p.max_fit)
						GEness_t[j]=p.max_fit;
					else if(GEness_t[j]<p.min_fit)
						(GEness_t[j]=p.min_fit); 

					if ( boost::math::isnan(abserror_v) || boost::math::isinf(abserror_v) || boost::math::isnan(corr_v) || boost::math::isinf(corr_v))
						GEness_v[j]=p.max_fit;
					else{
						if (p.fit_type==1)
							GEness_v[j] = abserror_v;
						else if (p.fit_type==2)
							GEness_v[j] = 1-corr_v;
						else if (p.fit_type==3)
							GEness_v[j] = abserror_v/corr_v;
						else if (p.fit_type==4)
							GEness_v[j] = 1-vaf_v;
						if (p.norm_error)
							GEness_v[j] = GEness_v[j]/target_std_v;
					}

					if(GEness_v[j]>p.max_fit)
						GEness_v[j]=p.max_fit;
					else if(GEness_v[j]<p.min_fit)
						(GEness_v[j]=p.min_fit); 

			//Fitness(trainers,p,d,s,GE[i]); //note: GE[0] does not get used here
			if (!p.GE_rank)
				GE[i].fitness+=abs(trainers[j].genty - abs(GEness_t[j]-GEness_v[j])/GEness_t[j]);
		
			}
		}
		if (!p.GE_rank)
			GE[i].fitness=GE[i].fitness/trainers.size();
		else{
			// use GE ranking ability to assign fitness
			GE[i].fitness=0;
			//float train1_genty = 0, train2_genty =0 ;
			float GE1_genty = 0, GE2_genty=0;
			float nsq = float(trainers.size()*trainers.size());
			for (int h=0;h<trainers.size();++h){
				//train1_genty = abs(trainers[h].fitness-trainers[h].fitness_v)/(trainers[h].fitness);
				GE1_genty = abs(GEness_t[h]-GEness_v[h])/GEness_t[h];
				
				for (int k=0;k<trainers.size();++k){
					//train2_genty = abs(trainers[k].fitness-trainers[k].fitness_v)/(trainers[k].fitness);
					GE2_genty = abs(GEness_t[k]-GEness_v[k])/GEness_t[k];

					if ((GE1_genty < GE2_genty && trainers[h].genty > trainers[k].genty) || 
						(GE1_genty > GE2_genty && trainers[h].genty < trainers[k].genty) )
						GE[i].fitness+=1;
				}
			}
			GE[i].fitness /= nsq;
		}
	}
	
};
void InitPopGE(vector <FitnessEstimator>& GE,vector<ind> &pop,vector<ind>& trainers,params p,vector<Randclass>& r,data& d,state& s)
{
	//vector <FitnessEstimator> GE; //fitness estimator population 
	//initialize estimator population
	GE.reserve(p.GE_pop_size);
	for (int i=0;i<p.GE_pop_size;++i){
		GE.push_back(FitnessEstimator(p.GE_ind_size,r,d,1));			
	}
	
	//initialize trainer population
	trainers.reserve(p.GE_train_size);
	for (int i=0;i<p.GE_train_size;++i){
		trainers.push_back(pop[r[omp_get_thread_num()].rnd_int(0,pop.size()-1)]);
	}
	//p.EstimateFitness = 0; // turn fitness estimation off (all data points used)
	//Fitness(trainers,p,d,s,GE[0]); //note: GE[0] does not get used here
	//p.EstimateFitness = 1; // turn fitness estimation back on
	//evaluate fitness of estimators
	p.EstimateFitness = 0; // turn fitness estimation off
	Fitness(trainers,p,d,s,GE[0]); //note: GE[0] does not get used here
	p.EstimateFitness = 1; // turn fitness estimation back on

	FitnessGE(GE,trainers,p,d,s);
	//perform selection
	sort(GE.begin(),GE.end(),SortGE());
};


void PickTrainersGE(vector<ind> pop, vector <FitnessEstimator>& GE,vector <ind>& trainers,params p,data& d,state& s)
{
	vector <vector<float>> GEfits; //rows: predictors, cols: population
	vector<float> meanfits(pop.size());
	vector <float> varfits(pop.size());
	//vector <ind> tmppop(1);
	// get GE fitness on each individual in population
	vector<ind> tmppop;

	for (int j=0;j<pop.size();++j){
		if(pop[j].fitness<p.max_fit){
			tmppop.push_back(ind());
			//makenewcopy(pop[j],tmppop.back());
			if(tmppop.size()!=pop.size())
				tmppop.back().clrPhen();
		}
	}
	//i: fitness predictors
	//j: solution population 
	for (int i=0;i<GE.size();++i){

		Fitness(tmppop,p,d,s,GE[i]);	
		GEfits.push_back(vector<float>(tmppop.size()));
		for (int j=0;j<tmppop.size();++j){
				GEfits[i][j]=tmppop[j].genty;	
				if(i!=GE.size()-1)
					tmppop[j].clrPhen();
			}
		// convert GEfits to ranks
		/*for (int h=0;h<tmppop.size();++h){
			float maxfit = *max_element(GEfits[i].begin(),GEfits[i].end());
			GEfits[i][h] = GEfits[i][h]/maxfit*tmppop.size();
		}*/
	}

	// calculate variance in fitness estimates
	for (int j=0;j<tmppop.size();++j){
		for (int i=0;i<GE.size();++i)
			meanfits[j]+=GEfits[i][j];

		meanfits[j]=meanfits[j]/GE.size(); // mean fitness over predictors

		for (int h=0;h<GE.size();++h){
			float tmp = GEfits[h][j]-meanfits[j];
			varfits[j]+=pow(GEfits[h][j]-meanfits[j],2);
		}

		varfits[j]=varfits[j]/(GE.size()-1); // variance in fitness over predictors
		if(boost::math::isinf(varfits[j])) varfits[j]=0;

		tmppop[j].GEvar=varfits[j];
	}
	
	sort(tmppop.begin(),tmppop.end(),SortGEVar());

	//vector<ind>::iterator it = pop.begin();
	trainers.clear();
	int q=0;
	while(trainers.size()<p.GE_train_size && q<tmppop.size()){
		if(q==0){
			trainers.push_back(tmppop.at(q));
		}
		else if (tmppop.at(q).eqn.compare(trainers.back().eqn)!=0){
			trainers.push_back(tmppop.at(q));
		}
		q++;
	}
	q=0;
	while(trainers.size()<p.GE_train_size){
		trainers.push_back(tmppop.at(q));
		q++;
	}
	//trainers.assign(tmppop.begin(),tmppop.begin()+p.GE_train_size);
	p.EstimateFitness = 0; // turn fitness estimation off
	Fitness(trainers,p,d,s,GE[0]); //note: GE[0] does not get used here
	p.EstimateFitness = 1; // turn fitness estimation back on
	//evaluate fitness of estimators
	FitnessGE(GE,trainers,p,d,s);
	sort(GE.begin(),GE.end(),SortGE());
};
void crossGE(vector <FitnessEstimator>& GE,vector<Randclass>& r)
{
	// select parents (tournament)
	int nt = 2;
	vector<int> parents;
	vector<int> choices(2);
	vector <FitnessEstimator> newGE;
	float bestfit; 
	int bestchoice;
	for (int i=0; i < GE.size(); ++i){
		
		for (int j=0; j<nt; ++j){
			
			choices[j] = r[omp_get_thread_num()].rnd_int(0,GE.size()-1);
			if (j==0) {
				bestfit = GE[choices[0]].fitness;
				bestchoice = choices[0];
			}
			else if(GE[choices[j]].fitness<bestfit)
			{
				bestfit = GE[choices[j]].fitness;
				bestchoice = choices[j];
			}
		}
		parents.push_back(bestchoice);		
	}

	//// new pop
	newGE.resize(GE.size());
	for (int i=0;i<parents.size();++i){
		//newGE[i]= GE[parents[i]];	
		newGE[i].FEpts.resize(GE[i].FEpts.size());
	}
	//cross parents
//	int p1,p2;
//	int crosspt;
	for (int i=0; i < GE.size()-1; ++i){
		int point1 = r[omp_get_thread_num()].rnd_int(0,GE[0].FEpts.size()-1);
		
		newGE[i].FEpts.assign(GE[parents[i]].FEpts.begin(),GE[parents[i]].FEpts.begin()+point1);
		newGE[i].FEpts.insert(newGE[i].FEpts.end(),GE[parents[i+1]].FEpts.begin()+point1,GE[parents[i+1]].FEpts.end());

		newGE[i+1].FEpts.assign(GE[parents[i+1]].FEpts.begin(),GE[parents[i+1]].FEpts.begin()+point1);
		newGE[i+1].FEpts.insert(newGE[i+1].FEpts.end(),GE[parents[i]].FEpts.begin()+point1,GE[parents[i]].FEpts.end());

		++i;
	}
	GE = newGE;
	// keep strictly the best predictors	
	/*GE.insert(GE.end(),newGE.begin(),newGE.end());
	sort(GE.begin(),GE.end(),SortGE());
	GE.erase(GE.begin()+newGE.size(),GE.end());*/
};
void mutateGE(vector <FitnessEstimator>& GE,params p,data& d,vector<Randclass>& r)
{
	int pt;
	int lastpt;
	if (p.train) lastpt = d.vals.size()/2-1;
	else lastpt = d.vals.size()-1;

	for (int i=0; i < GE.size(); ++i){
		pt = r[omp_get_thread_num()].rnd_int(0,GE[i].FEpts.size()-1);
		GE[i].FEpts[pt] = r[omp_get_thread_num()].rnd_int(0,lastpt);
	}
};
void EvolveGE(vector<ind> &pop, vector <FitnessEstimator>& GE,vector <ind>& trainers,params p,data& d,state& s,vector<Randclass>& r) 
{
	vector <float> GEfitness(GE.size()); //fitness of Fitness Estimators
	
	vector <FitnessEstimator> newGE = GE;
	//cross estimators
	crossGE(newGE,r);

	//mutate estimators
	mutateGE(newGE,p,d,r);
	
	//evaluate fitness of estimators
	FitnessGE(newGE,trainers,p,d,s);

	// keep strictly the best predictors	
	/*GE.insert(GE.end(),newGE.begin(),newGE.end());
	sort(GE.begin(),GE.end(),SortGE());
	GE.erase(GE.begin()+newGE.size(),GE.end());*/

	//keep the elite predictor and the new predictors
	newGE.insert(newGE.end(),GE[0]);
	sort(newGE.begin(),newGE.end(),SortGE());
	GE.assign(newGE.begin(),newGE.end()-1);
	////perform selection
	//sort(GE.begin(),GE.end(),SortGE());
	
	/*//if time to add new fitness trainer
	PickTrainers(pop,GE,trainers,p,d,s);
	*/
};
//void EvolveGE(vector<ind> &pop, vector <FitnessEstimator>& GE,vector <ind>& trainers,params p,data& d,state& s,vector<Randclass>& r)