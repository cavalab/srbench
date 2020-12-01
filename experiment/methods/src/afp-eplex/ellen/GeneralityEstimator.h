// run fitness estimator evolution.
#pragma once
#ifndef GENERALITYESTIMATOR_H
#define GENERALITYESTIMATOR_H

void InitPopGE(vector <FitnessEstimator>& GE,vector<ind> &pop,vector<ind>& trainers,params p,vector<Randclass>& r,data& d,state& s);
void EvolveGE(vector<ind> &pop, vector <FitnessEstimator>& GE,vector <ind>& trainers,params p,data& d,state& s,vector<Randclass>& r);
void setGEvals(vector<vector<float> >& GEvals, vector<float>& GEtarget,FitnessEstimator& GE, data& d);
void PickTrainersGE(vector<ind> pop, vector <FitnessEstimator>& GE,vector <ind>& trainers,params p,data& d,state& s);
#endif
