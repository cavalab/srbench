#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "data.h"
#include <cstdlib>
#include <math.h>
//#include "evaluator.h"
#include "state.h"
#include "EvalEqnStr.h"
#include <unordered_map>
#include "Line2Eqn.h"
//#include "exprtk.hpp"

void FitnessLex(vector<ind>& pop,params& p,data& d,state& s)
{
	//boost::progress_timer timer;
	//int count;
	/*evaluator e;
	e.init(p,d);*/
	//int wtf = pop.size();

	//set up data table for conversion of symbolic variables
	unordered_map <string,float*> datatable;
	vector<float> dattovar(d.label.size());

	for (unsigned int i=0;i<d.label.size(); i++)
			datatable.insert(pair<string,float*>(d.label[i],&dattovar[i]));

	int ndata_t,ndata_v; // training and validation data sizes
	if (p.train){
		ndata_t = d.vals.size()/2;
		ndata_v = d.vals.size()-ndata_t;
	}
	else{
		ndata_t = d.vals.size();
		ndata_v=0;
	}


	//#pragma omp parallel for private(e)
	for(int count = 0; count<pop.size(); count++)
	{
			try 
			{
			pop.at(count).abserror = 0;
			pop.at(count).abserror_v = 0;
			float meanout=0;
			float meantarget=0;
			float meanout_v=0;
			float meantarget_v=0;
			//float sumout=0;
			//float meantarget=0; 
			//get equation and equation form
			pop.at(count).eqn = Line2Eqn(pop.at(count).line);
			getEqnForm(pop.at(count).eqn,pop.at(count).eqn_form);
			
			// set pointer to dattovar in symbolic functions

			//GET EFFECTIVE SIZE
			pop.at(count).eff_size=0;
			for(int m=0;m<pop.at(count).line.size();m++){

				if(pop.at(count).line.at(m)->on)
					pop.at(count).eff_size++;
			}
			// Get Complexity
			pop.at(count).complexity=0;
			for(int m=0;m<pop.at(count).line.size();m++){

				if(pop.at(count).line.at(m)->on)
				{
					pop.at(count).complexity++;

					if (pop.at(count).line.at(m)->type=='/')
						pop.at(count).complexity++;
					else if (pop.at(count).line.at(m)->type=='s' || pop.at(count).line.at(m)->type=='c')
						pop.at(count).eff_size=pop.at(count).eff_size+2;
					else if (pop.at(count).line.at(m)->type=='e' || pop.at(count).line.at(m)->type=='l')
						pop.at(count).eff_size=pop.at(count).eff_size+3;
				}
			}
			// Get Fitness
			for(int m=0;m<pop.at(count).line.size();m++){
				if(pop.at(count).line.at(m)->type=='v')
					{// set pointer to dattovar 
						float* set = datatable.at(static_pointer_cast<n_sym>(pop.at(count).line.at(m))->varname);
						if(set==NULL)
							cout<<"hmm";
						static_pointer_cast<n_sym>(pop.at(count).line.at(m))->setpt(set);
						/*if (static_pointer_cast<n_sym>(pop.at(count).line.at(m))->valpt==NULL)
							cout<<"wth";*/
					}
			}
			//cout << "Equation" << count << ": f=" << pop.at(count).eqn << "\n";
			bool pass=true;
			if(!pop.at(count).eqn.compare("unwriteable")==0){
				vector<float> outstack;
				pop.at(count).output.clear();
				pop.at(count).output_v.clear();
				float SStot=0;
				float SSreg=0;
				float SSres=0;
				float q = 0;
				float var_target = 0;
				float var_ind = 0;
				float v1,v2,cv1,cv2;
				// calculate error 
				for(unsigned int sim=0;sim<d.vals.size();sim++)
				{
					for (unsigned int j=0; j<p.allvars.size();j++)
						dattovar.at(j)= d.vals[sim][j];

					for(int k=0;k<pop.at(count).line.size();k++){
						/*if(pop.at(count).line.at(k)->type=='v'){
							if (static_pointer_cast<n_sym>(pop.at(count).line.at(k))->valpt==NULL)
								cout<<"WTF";
						}*/
						if (pop.at(count).line.at(k)->on)
							pop.at(count).line.at(k)->eval(outstack);
					}

					if(!outstack.empty()){
						if (p.train){
							if(sim<ndata_t){
								pop.at(count).output.push_back(outstack.back());
								pop.at(count).abserror += abs(d.target.at(sim)-pop.at(count).output.at(sim));
								meantarget += d.target.at(sim);
								meanout += pop.at(count).output[sim];
							}
							else
							{
								pop.at(count).output_v.push_back(outstack.back());
								pop.at(count).abserror_v += abs(d.target.at(sim)-pop.at(count).output_v.at(sim-ndata_t));
								meantarget_v += d.target.at(sim);
								meanout_v += pop.at(count).output_v[sim-ndata_t];
							}

						}
						else {
							pop.at(count).output.push_back(outstack.back());
							pop.at(count).abserror += abs(d.target.at(sim)-pop.at(count).output.at(sim));
							meantarget += d.target.at(sim);
							meanout += pop.at(count).output[sim];
						}
					}
					else{
						pass=false;
						break;
						}
					outstack.clear();
				}
				if (pass){
					// mean absolute error
					pop.at(count).abserror = pop.at(count).abserror/ndata_t;
					meantarget = meantarget/ndata_t;
					meanout = meanout/ndata_t;
					//calculate correlation coefficient
					for (unsigned int c = 0; c<pop.at(count).output.size(); c++)
					{

						v1 = d.target.at(c)-meantarget;
						v2 = pop.at(count).output.at(c)-meanout;
						
					/*	cv1 =  d.target.at(c)-meanout;
						cv2 = pop.at(count).output.at(c)-meantarget;*/

						//SSres += pow(d.target.at(c)-pop.at(count).output.at(c),2);
						q += v1*v2;
						var_target+=pow(v1,2);
						var_ind+=pow(v2,2);
					}
					string tmpeqn = pop.at(count).eqn;
					vector<float> tmpout = pop.at(count).output;
					q = q/(ndata_t-1); //unbiased esimator
					var_target=var_target/(ndata_t-1); //unbiased esimator
					var_ind =var_ind/(ndata_t-1); //unbiased esimator
					if(var_target==0 || var_ind==0)
						pop.at(count).corr = 0;
					else
						pop.at(count).corr = pow(q,2)/(var_target*var_ind);

					if (p.train)
					{
						q = 0;
						var_target = 0;
						var_ind = 0;
							// mean absolute error
						pop.at(count).abserror_v = pop.at(count).abserror_v/ndata_v;
						meantarget_v = meantarget_v/ndata_v;
						meanout_v = meanout_v/ndata_v;
						//calculate correlation coefficient
						for (unsigned int c = 0; c<pop.at(count).output_v.size(); c++)
						{

							v1 = d.target.at(c+ndata_t)-meantarget_v;
							v2 = pop.at(count).output_v.at(c)-meanout_v;
						
						/*	cv1 =  d.target.at(c)-meanout;
							cv2 = pop.at(count).output.at(c)-meantarget;*/

							//SSres += pow(d.target.at(c)-pop.at(count).output.at(c),2);
							q += v1*v2;
							var_target+=pow(v1,2);
							var_ind+=pow(v2,2);
						}
						string tmpeqn = pop.at(count).eqn;
						vector<float> tmpout = pop.at(count).output_v;
						q = q/(ndata_v-1); //unbiased esimator
						var_target=var_target/(ndata_v-1); //unbiased esimator
						var_ind =var_ind/(ndata_v-1); //unbiased esimator
						if(var_target==0 || var_ind==0)
							pop.at(count).corr_v = 0;
						else
							pop.at(count).corr_v = pow(q,2)/(var_target*var_ind);
					}

				}
			}
			else{
				pop.at(count).abserror=p.max_fit;
				pop.at(count).corr = p.min_fit;
				if (p.train){
					pop.at(count).abserror_v=p.max_fit;
					pop.at(count).corr_v = p.min_fit;
				}
				}
			
			if (!pass)
				pop.at(count).corr = 0;
						

			if(pop.at(count).corr < p.min_fit)
				pop.at(count).corr=p.min_fit;

		    if(pop.at(count).output.empty())
				pop.at(count).fitness=p.max_fit;
			/*else if (*std::max_element(pop.at(count).output.begin(),pop.at(count).output.end())==*std::min_element(pop.at(count).output.begin(),pop.at(count).output.end()))
				pop.at(count).fitness=p.max_fit;*/
			else if ( boost::math::isnan(pop.at(count).abserror) || boost::math::isinf(pop.at(count).abserror) || boost::math::isnan(pop.at(count).corr) || boost::math::isinf(pop.at(count).corr))
				pop.at(count).fitness=p.max_fit;
			else{
				if (p.fit_type==1)
					pop.at(count).fitness = pop.at(count).abserror;
				else if (p.fit_type==2)
					pop.at(count).fitness = 1-pop.at(count).corr;
				else if (p.fit_type==3)
					pop.at(count).fitness = pop.at(count).abserror/pop.at(count).corr;
			}

		
			if(pop.at(count).fitness>p.max_fit)
				pop.at(count).fitness=p.max_fit;
			else if(pop.at(count).fitness<p.min_fit)
				(pop.at(count).fitness=p.min_fit);

			if(p.train){
				if (!pass)
					pop.at(count).corr_v = 0;
						

				if(pop.at(count).corr_v < p.min_fit)
					pop.at(count).corr_v=p.min_fit;

				if(pop.at(count).output_v.empty())
					pop.at(count).fitness_v=p.max_fit;
				/*else if (*std::max_element(pop.at(count).output_v.begin(),pop.at(count).output_v.end())==*std::min_element(pop.at(count).output_v.begin(),pop.at(count).output_v.end()))
					pop.at(count).fitness_v=p.max_fit;*/
				else if ( boost::math::isnan(pop.at(count).abserror_v) || boost::math::isinf(pop.at(count).abserror_v) || boost::math::isnan(pop.at(count).corr_v) || boost::math::isinf(pop.at(count).corr_v))
					pop.at(count).fitness_v=p.max_fit;
				else{
					if (p.fit_type==1)
						pop.at(count).fitness_v = pop.at(count).abserror_v;
					else if (p.fit_type==2)
						pop.at(count).fitness_v = 1-pop.at(count).corr_v;
					else if (p.fit_type==3)
						pop.at(count).fitness_v = pop.at(count).abserror_v/pop.at(count).corr_v;
				}

		
				if(pop.at(count).fitness_v>p.max_fit)
					pop.at(count).fitness_v=p.max_fit;
				else if(pop.at(count).fitness_v<p.min_fit)
					(pop.at(count).fitness_v=p.min_fit);
			}
			else{
				pop.at(count).corr_v=pop.at(count).corr;
				pop.at(count).abserror_v=pop.at(count).abserror;
				pop.at(count).fitness_v=pop.at(count).fitness;
			}
			}
			catch(CException ex)
			{
				ex.Report();
				pop.at(count).fitness=p.max_fit;
			}
		/*}
		else
			pop.at(count).fitness=p.max_fit;*/
		
		//cout << count << "\t" << pop.at(count).fitness << endl; 
		
	}
	s.numevals[omp_get_thread_num()]=s.numevals[omp_get_thread_num()]+pop.size();
	//cout << "\nFitness Time: ";
		
}

void getEqnForm(std::string& eqn,std::string& eqn_form)
{
//replace numbers with the letter c
	//boost::regex re("\d|[\d\.\d]");
	//(([0-9]+)(\.)([0-9]+))|([0-9]+)
	boost::regex re("(([0-9]+)(\.)([0-9]+))|([0-9]+)");
	//(\d+\.\d+)|(\d+)
	eqn_form=boost::regex_replace(eqn,re,"c");
	//std::cout << eqn << "\t" << eqn_form <<"\n";
}
