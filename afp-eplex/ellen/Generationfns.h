#pragma once
#ifndef GENERATIONFNS_H
#define GENERATIONFNS_H
//#include "FitnessEstimator.h"
void Tournament(const vector<ind>&,vector<unsigned int>&,params&,vector<Randclass>& r);
void DC(vector<ind>& pop,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE);
void ApplyGenetics(vector<ind>&,vector<unsigned int>&,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE);
void Crossover(ind&,ind&,vector<ind>&,params&,vector<Randclass>& r);
void CrossoverP(ind&,ind&,ind&,ind&,params&,vector<Randclass>& r);
void Mutate(ind&,vector<ind>&,params&,vector<Randclass>& r,Data& d);
void MutateP(ind&,ind& tmpind,params&,vector<Randclass>& r);
void HillClimb(ind&,params&,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE);
void StochasticGradient(ind& oldind,params& p,vector<Randclass>& r,Data& d,state& s,
        FitnessEstimator& FE, int gen);
void EpiHC(ind&,params&,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE);
void AgeBreed(vector<ind>& pop,params& p,vector<Randclass>& r,Data& d,state&,FitnessEstimator& FE);
void AgeFitSurvival(vector<ind>& pop,params& p,vector<Randclass>& r);
void AgeFitGenSurvival(vector<ind>& pop,params& p,vector<Randclass>& r);
void LexicaseSelect(vector<ind>& pop,vector<unsigned int>& parloc,params& p,vector<Randclass>& r,Data& d,state& s);
#endif
