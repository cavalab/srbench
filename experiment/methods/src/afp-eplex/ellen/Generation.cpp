#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "state.h"
#include "rnd.h"
#include "data.h"
#include "FitnessEstimator.h"
#include "Fitness.h"
#include "Generationfns.h"
#include "InitPop.h"
#include "EpiMut.h"
#include "ParetoSurvival.hpp"
#include "Prune.h"

void Generation(vector<ind>& pop,params& p,vector<Randclass>& r,Data& d,state& s,FitnessEstimator& FE)
{
	// std::cout << "Generation";

	//ind (*parloc)[p.popsize-1] = &pop;

	switch (p.sel)
	{
	case 1: // tournament selection
		{
			// std::cout << "tournament selection\n";
		//if (p.loud) boost::progress_timer timer;
		// return pop ids for parents
		vector<unsigned int> parloc(pop.size());
		//if (p.loud ) fcout << "     Tournament...";
		Tournament(pop,parloc,p,r);

		//elitism
		ind best;
		if (p.elitism){ // save best ind
			vector<ind>::iterator it_add = std::min_element(pop.begin(),pop.end(),SortFit());
			best = *it_add;
		}
		//assert (pop.size()== p.popsize) ;
		//if (p.loud ) fcout << "     Apply Genetics...";
		/*try
		{*/
		ApplyGenetics(pop,parloc,p,r,d,s,FE);
			//assert (pop.size()== p.popsize) ;
		/*}
		catch(...)
				{
					cout<<"error in ApplyGenetics\n";
					throw;
				}*/
		//if (p.loud ) fcout << "     Gen 2 Phen...";
		//if (p.loud ) fcout << "     Fitness...";


		// epigenetic mutation
		if (p.eHC_on && p.eHC_mut){
		for (int i = 0; i<pop.size(); ++i)
			EpiMut(pop.at(i),p,r);
		}

		Fitness(pop,p,d,s,FE);
		//get mutation/crossover stats
		s.setCrossPct(pop);

		//elitism
		if (p.elitism){ // replace worst ind with best ind
			vector<ind>::iterator it_rm = std::max_element(pop.begin(),pop.end(),SortFit());
			pop[it_rm-pop.begin()] = best;
		}
		/*if (p.pHC_on && p.ERC)
		{
				for(int i=0; i<pop.size(); ++i)
					HillClimb(pop.at(i),p,r,d,s);
		}
		if (p.eHC_on)
		{
				for(int i=0; i<pop.size(); ++i)
					EpiHC(pop.at(i),p,r,d,s);
		}*/
		break;
		}
	case 2: // deterministic crowding
		{
			int popsize;
			if (p.islands) popsize = p.popsize/p.nt;
			else popsize = p.popsize;

			for (int j=0; j<popsize;++j)
				DC(pop,p,r,d,s,FE);
			break;
		}
	case 3: // lexicase
		{
			vector<unsigned int> parloc(pop.size());

			if (p.lexage || std::find(p.lex_metacases.begin(), p.lex_metacases.end(), "age") != p.lex_metacases.end()) // if lex+afp
			{
				// add one new individual
				vector<ind> tmppop(1);
				tmppop[0].age=0;
				InitPop(tmppop,p,r);
				Fitness(tmppop,p,d,s,FE);
				pop.push_back(tmppop[0]);
			}


			LexicaseSelect(pop,parloc,p,r,d,s);
			//if (p.lex_age) vector<ind> tmppop(pop);

			//elitism
			ind best;
			if (p.elitism){ // save (aggregate) best ind
				vector<ind>::iterator it_add = std::min_element(pop.begin(),pop.end(),SortFit());
				best = *it_add;
			}

			ApplyGenetics(pop,parloc,p,r,d,s,FE);
			//FitnessLex(pop,parloc,p,r,d);

			// epigenetic mutation
			if (p.eHC_on && p.eHC_mut){
				for (int i = 0; i<pop.size(); ++i)
					EpiMut(pop.at(i),p,r);
			}

			// EDIT: this should be updated. "lexage" is confusing. in future release this will be
			// changed to something different
			Fitness(pop,p,d,s,FE);
			//get mutation/crossover stats
			s.setCrossPct(pop);

			if (p.lexage) // if lex+afp
			{// select new population with age-fitness pareto optimization
				if (p.PS_sel==1) {// age-fitness tournaments
					AgeFitSurvival(pop,p,r);
				}
				else{ // if using other objectives, use SPEA2 survival routine instead
					ParetoSurvival(pop,p,r,s);
				}
			}

			//elitism
			if (p.elitism & !p.lexage){ // replace (aggregate) worst ind with (aggregate) best ind
				vector<ind>::iterator it_rm = std::max_element(pop.begin(),pop.end(),SortFit());
				pop[it_rm-pop.begin()] = best;
			}
			break;
		}
	case 4: //Pareto survival
		{   //scheme:
			// produce a new population equal in size to the old
			// pool all individuals from both populations
			AgeBreed(pop,p,r,d,s,FE);

			// add one new individual
			vector<ind> tmppop(1);
			tmppop[0].age=0;
			InitPop(tmppop,p,r);

			Fitness(tmppop,p,d,s,FE);
			pop.push_back(tmppop[0]);

			// select new population with tournament size 2, based on pareto age-fitness
			if (p.PS_sel==1) {// age-fitness tournaments
				AgeFitSurvival(pop,p,r);
			}
			// else if (p.PS_sel==2)
			// 	AgeFitGenSurvival(pop,p,r);*/
			else{ // if using other objectives, use SPEA2 survival routine instead
				ParetoSurvival(pop,p,r,s);
			}

		break;

		}
	case 5: // random search
		{
			//elitism
			ind best;
			if (p.elitism){ // save (aggregate) best ind
				vector<ind>::iterator it_add = std::min_element(pop.begin(),pop.end(),SortFit());
				best = *it_add;
			}

			vector<unsigned int> parloc(pop.size());
			for (unsigned i = 0; i<parloc.size(); ++i){
				parloc[i] = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			}
			//if (p.loud ) fcout << "     Tournament...";
			ApplyGenetics(pop,parloc,p,r,d,s,FE);

			Fitness(pop,p,d,s,FE);

			//elitism
			if (p.elitism){ // replace (aggregate) worst ind with (aggregate) best ind
				vector<ind>::iterator it_rm = std::max_element(pop.begin(),pop.end(),SortFit());
				pop[it_rm-pop.begin()] = best;
			}
			break;
		}

	default:
		cout << "Bad p.sel parameter. " << endl;
		break;
	}
	if (p.classification && p.class_m4gp && p.class_prune)
		Prune(pop,p,r,d,s,FE);

	//if (p.loud ) fcout << "  Gentime...";
}
