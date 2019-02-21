#include "stdafx.h"
#include "pop.h"
#include "state.h"
#include "Generationfns.h"

void Breed(vector<ind>& pop,params& p,vector<Randclass>& r)
{
	//boost::progress_timer timer;
	float choice;
	
	vector <ind> tmppop;
	///if(!p.parallel)
	//{
	unsigned int numits=0;
	std::random_shuffle(pop.begin(),pop.end(),r[omp_get_thread_num()]);
	while(tmppop.size()<pop.size())
	{
		
		choice = r[omp_get_thread_num()].rnd_flt(0,1);

		if (choice<p.rep_wheel[1] && pop.size()>numits+1) // crossover, as long as there are at least two parents left
		{			
			// only breed if parents have different fitness 
			if(pop.at(numits).fitness!=pop.at(numits+1).fitness) 
			{				
				//cross parents to make two kids and push kids into tmppop
				Crossover(pop.at(numits),pop.at(numits+1),tmppop,p,r);
				tmppop.back().age=max(pop.at(numits).age,pop.at(numits+1).age);
				tmppop.at(tmppop.size()-2).age=max(pop.at(numits).age,pop.at(numits+1).age);
				pop.at(numits).age++;
				pop.at(numits+1).age++;
				
				numits+=2;
				
			}
			else
			{
				Mutate(pop.at(numits),tmppop,p,r);
				Mutate(pop.at(numits+1),tmppop,p,r);
				numits+=2;
			}
		}
		else if (choice < p.rep_wheel[2]) //mutation
		{
			Mutate(pop.at(numits),tmppop,p,r);
			numits++;
		}
		
	}
	//}
	//else //parallel version
	//{
	//	// pick operations
	//	// pick parloc indices for each operation
	//	// in parallel, run operations with parlocs
	//	tmppop.resize(pop.size());

	//	int kidstomake=0;
	//	vector<int> kid1;
	//	vector<int> kid2;
	//	vector<int> numfns;
	//	vector <int> par1;
	//	vector <int> par2;
	//	float choice;
	//	std::random_shuffle(parloc.begin(),parloc.end(),r[0]);
	//	while(kidstomake<pop.size())
	//	{
	//		choice = r[omp_get_thread_num()].rnd_flt(0,1);
	//		if(choice <= p.rep_wheel[0]) // reproduction
	//		{
	//			//cout<<"rep\n";
	//			//par1.push_back(parloc[kidstomake]);
	//			//par2.push_back(0);
	//			kid1.push_back(kidstomake);
	//			kid2.push_back(0);
	//			kidstomake++;
	//			numfns.push_back(1);
	//		}
	//		else if (choice<=p.rep_wheel[1] && parloc.size()>kidstomake+1) // crossover, as long as there are at least two parents left
	//		{				
	//			//cout<<"crs\n";
	//			if(pop.at(parloc[kidstomake]).fitness!=pop.at(parloc[kidstomake+1]).fitness) 
	//			{				
	//				kid1.push_back(kidstomake);
	//				kid2.push_back(kidstomake+1);
	//				kidstomake+=2;
	//				numfns.push_back(2);
	//			}
	//		}
	//		else if (choice <= p.rep_wheel[2]) //mutation
	//		{
	//			//cout<<"mut\n";
	//			//par1.push_back(parloc[kidstomake]);
	//			//par2.push_back(0);
	//			kid1.push_back(kidstomake);
	//			kid2.push_back(0);
	//			kidstomake++;
	//			numfns.push_back(3);
	//		}
	//	}
	//	//cout<<"performing genetics\n";
	//	//#pragma omp parallel for 
	//	for(int q=0;q<numfns.size();q++)
	//	{
	//		switch(numfns[q])
	//		{
	//		case 1:

	//			tmppop.at(kid1.at(q)) = pop.at(parloc.at(kid1.at(q)));
	//			break;
	//		case 2:
	//			//cout<<"crossP\n";
	//			CrossoverP(pop.at(parloc.at(kid1.at(q))),pop.at(parloc.at(kid2.at(q))),tmppop.at(kid1.at(q)),tmppop.at(kid2.at(q)));
	//			break;
	//		case 3:
	//			//cout<<"mutateP\n";
	//			MutateP(pop.at(parloc.at(kid1.at(q))),tmppop.at(kid1.at(q)));
	//			
	//			break;
	//		}
	//	}

	//}

	while(tmppop.size()>pop.size())
		tmppop.erase(tmppop.end()-1);
	//replace pop with tmppop
	pop.swap(tmppop);
	//clear tmppop

}