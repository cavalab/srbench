#include "stdafx.h"
#include "pop.h"
#include "general_fns.h"


void MutateP(ind& par,ind& tmpind,params& p,vector<Randclass>& r)
{
	try
	{
	//cout<<"in MutateP\n";
	vector<unsigned int> ichange;
	//ind kid;
	//cout<<"calc ichange \n";
	for(unsigned int i = 0;i<par.line.size();i++)
	{
		if(r[omp_get_thread_num()].rnd_flt(0,1)<=p.mut_ar)
			ichange.push_back(i);
	}
	//cout<<"end ichange\n";
	for(unsigned int j=0;j<ichange.size();j++)
	{
		if(par.line.at(ichange.at(j))>=100) // insert function
		{
			//if constant, perturb with gaussian noise
			if (is_number(par.args.at(par.line.at(ichange.at(j))-100)))
			{
				float num = std::stof(par.args.at(par.line.at(ichange.at(j))-100));
				//cout<<"r call in is_number if \n";
				num = num/2 + r[omp_get_thread_num()].gasdev()*num/2;
				par.args.at(par.line.at(ichange.at(j))-100) = to_string(static_cast<long double>(num));
			}
			//else if variable, pick random variable replacement
			else
			{ 
				par.args.at(par.line.at(ichange.at(j))-100) = p.allvars.at(r[omp_get_thread_num()].rnd_int(0,p.allvars.size()-1));
			}

		}
		else
		{
			NewInstruction(par,ichange.at(j),p,r);
		}
	}
		par.origin='m';
		tmpind = par;
	}
	catch(...)
	{
		cout<<"caught.\n";

	}
}

