/*
* the cpp file for the instruction set
"implied parenthesis" are removed in this version
*/ 
#include "stdafx.h"
#include "pop.h"
#include "params.h"

using namespace std;
extern params p;

void absf(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"abs(");

	newind.ptr.back() += 5;
}
void add(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"+0)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+2; 
	newind.ptr.back() += 3;
}
void cosf(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"cos(");

	newind.ptr.back() += 5;
}
void DEL0(ind &newind)
{
}
void DEL1(ind &newind)
{
}
void divL(ind &newind)
{
	newind.eqn.insert(newind.ptr.back(),")");
	newind.eqn.insert(newind.ptr.front(),"(1/");

	newind.ptr.front()++; 
	newind.ptr.back()  = newind.ptr.front()+1;
}
void divR(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"/1)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+2; 
	newind.ptr.back() += 3;
}
void DNL(ind &newind)
{
}
void DNR(ind &newind)
{
}
void FLIP(ind &newind)
{
}
void ins(ind &newind, int loc)
{
	string arg = newind.args.at(loc);
	newind.eqn.replace(newind.ptr.front(),newind.ptr.back()-newind.ptr.front()+1,arg); 
	newind.ptr.back() = newind.ptr.front()+arg.size()-1;
	
}

void mul(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"*1)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+2; 
	newind.ptr.back() += 3;
}
void NOOP(ind &newind)
{ //do nothing
}
void sinf(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"(sin");

	newind.ptr.back() += 5;
}
void subL(ind &newind)
{
	
	newind.eqn.insert(newind.ptr.back(),")");
	newind.eqn.insert(newind.ptr.front(),"(0-");

	newind.ptr.front()++; 
	newind.ptr.back()  = newind.ptr.front();

}
void subR(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"-0)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+2; 
	newind.ptr.back() += 3;
}
void totheL(ind &newind)
{
}
void totheR(ind &newind)
{
}
void UP(ind &newind)
{
	//cout << "UP" << endl;
	if (newind.ptr.back()-newind.ptr.front()+1 != newind.eqn.size())
	{
		//vector<int> ptr(newind.ptr);
		vector<int> newptr(newind.ptr);
		string eqn = newind.eqn;	

		newptr.front() = eqn.rfind('(',newind.ptr.front()-1);
		newptr.back() = eqn.find(')',newind.ptr.back()+1);

		eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);

		// check parenthesis in new equation
		int n_in = 0;
		int n_out= 0;
		std::string ::size_type pos_in = 0;
		std::string ::size_type pos_out = 0;

		while( (pos_in = eqn.find( '(', pos_in )) 
					 != std::string::npos ) {
    		n_in++;
    		pos_in++; 
		}
		while( (pos_out = eqn.find( ')', pos_out )) 
					 != std::string::npos ) {
    		n_out++;
    		pos_out++; 
		}

		while (n_in != n_out)
		{
			if (n_in > n_out)
			{
				
				pos_out = newptr.back()-newptr.front()+1;
				pos_in = pos_out;

				newptr.back() = newind.eqn.find(')',newptr.back()+1);
				n_out++;
				eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
				
				while( (pos_in = eqn.find( '(', pos_in)) 
					 != std::string::npos ) {
    				n_in++;
    				pos_in++;

				}
			}
			else
			{
				pos_in = newptr.front();
				newptr.front() = newind.eqn.rfind('(',newptr.front()-1);			
				n_in++;

				eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
				
				pos_in -= newptr.front();
				string tmpeqn; 
				tmpeqn.assign(newind.eqn,newptr.front(),pos_in);

			
				// check parenthesis in new equation
				pos_out=0;
				while( (pos_out = tmpeqn.find( ')', pos_out )) 
							 != std::string::npos ) {
    				n_out++;
    				pos_out++; 
				}
			}
			
		
		
		}
		//newind.eqn.swap(eqn);
		newind.ptr.swap(newptr);
	}
}
void UP2(ind &newind)
{
	UP(newind);
	UP(newind);
}
void UP3(ind &newind)
{
	UP(newind);
	UP(newind);
	UP(newind);
}
