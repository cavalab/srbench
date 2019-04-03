/*
* the cpp file for the instruction set
*/ 
#include "stdafx.h"
#include "pop.h"
#include "params.h"

using namespace std;


//void absf(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"))");
//	newind.eqn.insert(newind.ptr.front(),"(abs(");
//
//	newind.ptr.back() += 7;
//}
//void add(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"+(0))");
//	newind.eqn.insert(newind.ptr.front(),"(");
//
//	newind.ptr.front() = newind.ptr.back()+3; 
//	newind.ptr.back() += 5;
//}
//void cosf(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"))");
//	newind.eqn.insert(newind.ptr.front(),"(sin(");
//
//	newind.ptr.back() += 7;
//}
//void DEL0(ind &newind)
//{
//}
//void DEL1(ind &newind)
//{
//}
//void divL(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back(),")");
//	newind.eqn.insert(newind.ptr.front(),"((1)/");
//
//	newind.ptr.front()++; 
//	newind.ptr.back()  = newind.ptr.front()+2;
//}
//void divR(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"/(1))");
//	newind.eqn.insert(newind.ptr.front(),"(");
//
//	newind.ptr.front() = newind.ptr.back()+3; 
//	newind.ptr.back() += 5;
//}
void DNL(ind &newind)
{
}
void DNR(ind &newind)
{
}
//void FLIP(ind &newind)
//{
//}
//void ins(ind &newind, int loc)
//{
//	string arg = "(" + newind.args.at(loc) + ")";
//	newind.eqn.replace(newind.ptr.front(),newind.ptr.back()-newind.ptr.front()+1,arg); 
//	newind.ptr.back() = newind.ptr.front()+arg.size()-1;
//	
//}
//void mul(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"*(1))");
//	newind.eqn.insert(newind.ptr.front(),"(");
//
//	newind.ptr.front() = newind.ptr.back()+3; 
//	newind.ptr.back() += 5;
//}
//void NOOP(ind &newind)
//{ //do nothing
//}
//void sinf(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"))");
//	newind.eqn.insert(newind.ptr.front(),"(sin(");
//
//	newind.ptr.back() += 7;
//}
//void subL(ind &newind)
//{
//	
//	newind.eqn.insert(newind.ptr.back(),")");
//	newind.eqn.insert(newind.ptr.front(),"((0)-");
//
//	newind.ptr.front()++; 
//	newind.ptr.back()  = newind.ptr.front()+2;
//
//}
//void subR(ind &newind)
//{
//	newind.eqn.insert(newind.ptr.back()+1,"-(0))");
//	newind.eqn.insert(newind.ptr.front(),"(");
//
//	newind.ptr.front() = newind.ptr.back()+3; 
//	newind.ptr.back() += 5;
//}
//void totheL(ind &newind)
//{
//}
//void totheR(ind &newind)
//{
//}
//void UP(ind &newind)
//{
//	//cout << "UP" << endl;
//	if (newind.ptr.back()-newind.ptr.front()+1 != newind.eqn.size())
//	{
//		//vector<int> ptr(newind.ptr);
//		vector<int> newptr(newind.ptr);
//		string eqn = newind.eqn;	
//
//		newptr.front() = eqn.rfind('(',newind.ptr.front()-1);
//		newptr.back() = eqn.find(')',newind.ptr.back()+1);
//
//		eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//
//		// check parenthesis in new equation
//		int n_in = 0;
//		int n_out= 0;
//		std::string ::size_type pos_in = 0;
//		std::string ::size_type pos_out = 0;
//
//		while( (pos_in = eqn.find( '(', pos_in )) 
//					 != std::string::npos ) {
//    		n_in++;
//    		pos_in++; 
//		}
//		while( (pos_out = eqn.find( ')', pos_out )) 
//					 != std::string::npos ) {
//    		n_out++;
//    		pos_out++; 
//		}
//
//		while (n_in != n_out)
//		{
//			if (n_in > n_out)
//			{
//				
//				pos_out = newptr.back()-newptr.front()+1;
//				pos_in = pos_out;
//
//				newptr.back() = newind.eqn.find(')',newptr.back()+1);
//				n_out++;
//				eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				
//				while( (pos_in = eqn.find( '(', pos_in)) 
//					 != std::string::npos ) {
//    				n_in++;
//    				pos_in++;
//
//				}
//			}
//			else
//			{
//				pos_in = newptr.front();
//				newptr.front() = newind.eqn.rfind('(',newptr.front()-1);			
//				n_in++;
//
//				eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				
//				pos_in -= newptr.front();
//				string tmpeqn; 
//				tmpeqn.assign(newind.eqn,newptr.front(),pos_in);
//
//			
//				// check parenthesis in new equation
//				pos_out=0;
//				while( (pos_out = tmpeqn.find( ')', pos_out )) 
//							 != std::string::npos ) {
//    				n_out++;
//    				pos_out++; 
//				}
//			}
//			
//		
//		
//		}
//		//newind.eqn.swap(eqn);
//		newind.ptr.swap(newptr);
//	}
//}
//void UP2(ind &newind)
//{
//	UP(newind);
//	UP(newind);
//}
//void UP3(ind &newind)
//{
//	UP(newind);
//	UP(newind);
//	UP(newind);
//}


void add(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"+0)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+3; 
	newind.ptr.back() += 3;
}
void subR(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"-0)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+3; 
	newind.ptr.back()  = newind.ptr.front();
}
void mul(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"*1)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+3; 
	newind.ptr.back()  = newind.ptr.front();
}
void divR(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,"/1)");
	newind.eqn.insert(newind.ptr.front(),"(");

	newind.ptr.front() = newind.ptr.back()+3; 
	newind.ptr.back()  = newind.ptr.front();
}
void subL(ind &newind)
{
	
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"(0-");

	newind.ptr.front()++; 
	newind.ptr.back()  = newind.ptr.front();

}
void divL(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"(1/");

	newind.ptr.front()++; 
	newind.ptr.back()  = newind.ptr.front();
}
void cosf(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"cos(");

	newind.ptr.back() += 5;
}
void sinf(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"sin(");

	newind.ptr.back() += 5;
}
void absf(ind &newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"abs(");

	newind.ptr.back() += 5;
}
void expf(ind& newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"exp(");

	newind.ptr.back() += 5;
}
void logf(ind& newind)
{
	newind.eqn.insert(newind.ptr.back()+1,")");
	newind.eqn.insert(newind.ptr.front(),"log(");

	newind.ptr.back() += 5;
}
void DEL0(ind &newind)
{
	newind.eqn.replace(newind.ptr.front(),newind.ptr.back()-newind.ptr.front()+1,"0"); 
	newind.ptr.back() = newind.ptr.front();
}
void DEL1(ind &newind)
{
	newind.eqn.replace(newind.ptr.front(),newind.ptr.back()-newind.ptr.front()+1,"1"); 
	newind.ptr.back() = newind.ptr.front();
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


void NOOP(ind &newind)
{ //do nothing
}



void totheL(ind &newind)
{
}
void totheR(ind &newind)
{
}
//void DNL(ind &newind)
//{	
//	std::string tmp;
//	tmp = newind.eqn.substr(newind.ptr[0],newind.ptr[1]-newind.ptr[0]+1);
//	if (tmp.find('(') != std::string::npos && 
//		tmp.find(')') != std::string::npos) // only descend if there is nesting to descend
//	{
//		ind tmpind = newind;
//		vector<int> newptr(newind.ptr);
//		string eqn = newind.eqn;	
//		char tmp = eqn[newptr.front()];
//		if (eqn[newptr.front()]=='(')
//		{
//			newptr.front() ++;
//			if (eqn.find_first_of("+-*//^",newptr.front()) != string::npos)
//				newptr.back() = eqn.find_first_of("+-*//^",newptr.front())-1;
//			else
//				newptr.back()--;
//			if (newptr.back() < 0 || newptr.front()<0)
//				cout<< "mistake\n";
//
//		}
//		else
//		{
//			// if a unary function is highlighted, pointer becomes the argument contained within
//			if (eqn.find("abs",newptr.front()) != std::string::npos ||
//				eqn.find("sin",newptr.front()) != std::string::npos ||
//				eqn.find("cos",newptr.front()) != std::string::npos ||
//				eqn.find("exp",newptr.front()) != std::string::npos ||
//				eqn.find("log",newptr.front()) != std::string::npos)
//			{
//
//				newptr.front() = eqn.find('(',newind.ptr.front()+1)+1;
//				newptr.back()--;
//				
//				if (newptr.back() < 0 || newptr.front()<0)
//				cout<< "mistake\n";
//				
//
//			}
//			else
//			{
//				std::cout<<"Something's wrong\n";
//
//			}
//
//		}
//		
//		// check parenthesis in new equation
//		eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				int n_in = 0;
//				int n_out= 0;
//				std::string ::size_type pos_in = 0;
//				std::string ::size_type pos_out = 0;
//
//				while( (pos_in = eqn.find( '(', pos_in )) 
//							 != std::string::npos ) {
//    				n_in++;
//    				pos_in++; 
//				}
//				while( (pos_out = eqn.find( ')', pos_out )) 
//							 != std::string::npos ) {
//    				n_out++;
//    				pos_out++; 
//				}
//
//				while (n_in != n_out)
//				{
//					if (n_in > n_out)
//					{
//				
//						pos_out = newptr.back()-newptr.front()+1;
//						pos_in = pos_out;
//
//						newptr.back() = newind.eqn.find(')',newptr.back()+1);
//						n_out++;
//						eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				
//						while( (pos_in = eqn.find( '(', pos_in)) 
//							 != std::string::npos ) {
//    						n_in++;
//    						pos_in++;
//
//						}
//					}
//					else
//					{
//						pos_in = newptr.front();
//						newptr.front() = newind.eqn.rfind('(',newptr.front()-1);			
//						n_in++;
//
//						eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				
//						pos_in -= newptr.front();
//						string tmpeqn; 
//						tmpeqn.assign(newind.eqn,newptr.front(),pos_in);
//
//			
//						// check parenthesis in new equation
//						pos_out=0;
//						while( (pos_out = tmpeqn.find( ')', pos_out )) 
//									 != std::string::npos ) {
//    						n_out++;
//    						pos_out++; 
//						}
//					}
//			
//		
//		
//				}
//				if (newptr.back() < 0 || newptr.front()<0)
//				cout<< "mistake\n";
//		newind.ptr.swap(newptr);
//
//		if (newind.eqn.find('(',0)==std::string::npos)
//			std::cout<< "mistake\n";
//	}
//}
//void DNR(ind &newind)
//{
//	std::string tmp;
//	tmp = newind.eqn.substr(newind.ptr[0],newind.ptr[1]-newind.ptr[0]+1);
//	if (tmp.find('(') != std::string::npos && 
//		tmp.find(')') != std::string::npos) // only descend if there is nesting to descend
//	{
//		ind tmpind = newind;
//		vector<int> newptr(newind.ptr);
//		string eqn = newind.eqn;	
//		char tmp = eqn[newptr.back()];
//		if (eqn[newptr.front()]=='(')
//		{
//			newptr.back() --;
//			tmp = eqn.find_last_of("+-*/^",newptr.back());
//			if (eqn.find_last_of("+-*/^",newptr.back()) != string::npos)
//				newptr.front() = eqn.find_last_of("+-*//^",newptr.back())+1;
//			else
//				newptr.front()++;
//			if (newptr.back() < 0 || newptr.front()<0)
//				cout<< "mistake\n";
//
//		}
//		else
//		{
//			// if a unary function is highlighted, pointer becomes the argument contained within
//			if (eqn.find("abs",newptr.front()) != std::string::npos ||
//				eqn.find("sin",newptr.front()) != std::string::npos ||
//				eqn.find("cos",newptr.front()) != std::string::npos ||
//				eqn.find("exp",newptr.front()) != std::string::npos ||
//				eqn.find("log",newptr.front()) != std::string::npos)
//			{
//
//				newptr.front() = eqn.find('(',newind.ptr.front()+1)+1;
//				newptr.back()--;
//				
//				if (newptr.back() < 0 || newptr.front()<0)
//				cout<< "mistake\n";
//				
//
//			}
//			else
//			{
//				std::cout<<"Something's wrong\n";
//
//			}
//
//		}
//		
//		// check parenthesis in new equation
//		eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				int n_in = 0;
//				int n_out= 0;
//				std::string ::size_type pos_in = 0;
//				std::string ::size_type pos_out = 0;
//
//				while( (pos_in = eqn.find( '(', pos_in )) 
//							 != std::string::npos ) {
//    				n_in++;
//    				pos_in++; 
//				}
//				while( (pos_out = eqn.find( ')', pos_out )) 
//							 != std::string::npos ) {
//    				n_out++;
//    				pos_out++; 
//				}
//
//				while (n_in != n_out)
//				{
//					if (n_in > n_out)
//					{
//				
//						pos_out = newptr.back()-newptr.front()+1;
//						pos_in = pos_out;
//
//						newptr.back() = newind.eqn.find(')',newptr.back()+1);
//						n_out++;
//						eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				
//						while( (pos_in = eqn.find( '(', pos_in)) 
//							 != std::string::npos ) {
//    						n_in++;
//    						pos_in++;
//
//						}
//					}
//					else
//					{
//						pos_in = newptr.front();
//						newptr.front() = newind.eqn.rfind('(',newptr.front()-1);			
//						n_in++;
//
//						eqn.assign(newind.eqn,newptr.front(),newptr.back()-newptr.front()+1);
//				
//						pos_in -= newptr.front();
//						string tmpeqn; 
//						tmpeqn.assign(newind.eqn,newptr.front(),pos_in);
//
//			
//						// check parenthesis in new equation
//						pos_out=0;
//						while( (pos_out = tmpeqn.find( ')', pos_out )) 
//									 != std::string::npos ) {
//    						n_out++;
//    						pos_out++; 
//						}
//					}
//			
//		
//		
//				}
//				if (newptr.back() < 0 || newptr.front()<0)
//				cout<< "mistake\n";
//		newind.ptr.swap(newptr);
//
//		if (newind.eqn.find('(',0)==std::string::npos)
//			std::cout<< "mistake\n";
//	}
//}
void UP(ind &newind)
{
	//cout << "UP" << endl;
	if (newind.ptr.front()>1 && newind.ptr.back()< newind.eqn.size()-1) // if there is an up to go 
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