// Gen2Phen.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "pop.h"
#include "data.h"
#include "params.h"
#include "instructionset.h"

using namespace std;

// functions

// pointer array to functions
void (*pf[22])(ind &)= {absf,add,cosf,DEL0,DEL1,divL,divR,DNL,DNR,FLIP,mul,NOOP,sinf,subL,subR,totheL,totheR,UP,UP2,UP3,expf,logf};
void getEqn_Form(std::string& eqn,std::string& eqn_form);

void Gen2Phen(vector<ind> &pop,params& p)
{
	//boost::progress_timer timer;
	//int i; j;
	//int count=0;

	//#pragma omp parallel for 
	for(int i=0;i<pop.size();i++)
	{
		pop.at(i).eqn = p.sim_nom_mod; //set equation embryo
		if (pop.at(i).ptr.size()!=2)	
			pop.at(i).ptr.resize(2);
		pop.at(i).ptr[0] = 1;
		pop.at(i).ptr[1] = p.sim_nom_mod.size()-2;
		//pop.at(i).eff_size = 0;

		ind tmpind = pop.at(i); //temporary updating ind
		int eff_size = 0;
		for(unsigned int j=0; j<tmpind.line.size(); j++)
		{
			
			if (tmpind.epiline.at(j))
			{
				if (tmpind.line.at(j)>99)
				{
			//			std::cout << "insert" << endl;
					ins(tmpind,tmpind.line.at(j)-100);
				}
				else{
			//		std:cout << "function" << *pf[pop.at(i).line.at(j)-1] << endl;
					pf[tmpind.line.at(j)-1](tmpind);
				}
				if (tmpind.eqn.size()<=p.max_dev_len) //check developmental length before continuing on
				{
					pop.at(i)=tmpind;
					eff_size++;
					//cout << eff_size << "\n";
				}
				else
					tmpind = pop.at(i); // revert tmpind back to a valid equation size (last successful step)

			}
			//cout << "new model: " << pop.at(i).eqn << endl;
		}
		getEqn_Form(pop.at(i).eqn,pop.at(i).eqn_form);
		pop.at(i).eff_size = eff_size;

	//d.parser.compile(pop.at(i).eqn,pop.at(i).expression);
	}

	//cout << "\nGen2Phen time: ";
	
}
void getEqn_Form(std::string& eqn,std::string& eqn_form)
{
//replace numbers with the letter c
	//boost::regex re("\d|[\d\.\d]");
	//(([0-9]+)(\.)([0-9]+))|([0-9]+)
	boost::regex re("(([0-9]+)(\.)([0-9]+))|([0-9]+)");
	//(\d+\.\d+)|(\d+)
	eqn_form=boost::regex_replace(eqn,re,"c");
	//std::cout << eqn << "\t" << eqn_form <<"\n";
}
