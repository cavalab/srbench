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
#include "EpiMut.h"

void EpiMut(ind& x,params& p,vector<Randclass>& r)
{
	//Mutate individual's epigenetic properties
	//ind newind;
	//makenewcopy(oldind,newind);
	for (int j=0;j<p.eHC_its; ++j) // for number of specified iterations
	{
	  for(unsigned int h = 0;h<x.line.size();++h)
		{
			if(r[omp_get_thread_num()].rnd_flt(0,1)<=p.eHC_prob)
				x.line.at(h).on = !x.line.at(h).on;
		}	
	}
	//oldind = newind;
}