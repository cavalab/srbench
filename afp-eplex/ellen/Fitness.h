#pragma once
#ifndef FITNESS_H
#define FITNESS_H

#include "FitnessEstimator.h"

class CException
{
public:
	char* message;
	CException( char* m ) { message = m; };
	void Report(){cout << "error calculating fitness\n";};

};

//void getEqnForm(std::string& eqn,std::string& eqn_form);
float getCorr(vector<float>& output,vector<float>& target,float meanout,float meantarget,int off,float& target_std);
float VAF(vector<float>& output,vector<float>& target,float meantarget,int off);
float std_dev(vector<float>& target,float& meantarget);
void Fitness(vector<ind>& pop,params& p,Data& d,state& s,FitnessEstimator& FE);
bool SlimFitness(ind& me,params& p,Data& d,state& s,FitnessEstimator& FE,  int linestart,  float orig_fit);

#endif