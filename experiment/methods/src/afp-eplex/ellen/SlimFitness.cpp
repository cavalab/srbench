#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "data.h"
#include "rnd.h"
#include "state.h"
#include "Line2Eqn.h"
#include "EvalEqnStr.h"
#include <unordered_map>
#if defined(_WIN32)
	#include <regex>
#else
	#include <boost/regex.hpp>
#endif
#include "FitnessEstimator.h"
#include "Fitness.h"

//void getEqnForm(std::string& eqn,std::string& eqn_form);
//float getCorr(vector<float>& output,vector<float>& target,float meanout,float meantarget,int off);
//int getComplexity(string& eqn);

//void getEqnForm(std::string& eqn,std::string& eqn_form)
//{
////replace numbers with the letter c
//#if defined(_WIN32)
//	std::regex e ("(([0-9]+)(\.)([0-9]+))|([0-9]+)");
//	std::basic_string<char> tmp = "c";
//	std::regex_replace (std::back_inserter(eqn_form), eqn.begin(), eqn.end(), e,tmp);
//#else
//	boost::regex e ("(([0-9]+)(\.)([0-9]+))|([0-9]+)");
//	eqn_form = boost::regex_replace(eqn,e,"c");
//#endif 
//	//(\d+\.\d+)|(\d+)
//	//eqn_form = eqn;
//	//eqn_form=std::tr1::regex_replace(eqn,e,tmp.c_str(),std::tr1::regex_constants::match_default);
//	//std::string result;
//
//
//	//std::regex_replace(std::back_inserter(eqn_form),eqn.begin(),eqn.end(),e,"c",std::regex_constants::match_default);
//	//std::regex_replace(std::back_inserter(eqn_form),eqn.begin(),eqn.end(),e,"c",std::tr1::regex_constants::match_default
//    //std::cout << result;
//	//std::cout << eqn << "\t" << eqn_form <<"\n";
//}
//int getComplexity(string& eqn)
//{
//	int complexity=0;
//	char c;
//	for(int m=0;m<eqn.size();m++){
//		c=eqn[m];
//		
//		if(c=='/')
//			complexity=complexity+2;
//		else if (c=='s'){
//			if(m+2<eqn.size()){
//				if ( eqn[m+1]=='i' && eqn[m+2] == 'n'){
//					complexity=complexity+3;
//					m=m+2;
//				}
//			}
//		}
//		else if (c=='c'){
//			if(m+2<eqn.size()){
//				if ( eqn[m+1]=='o' && eqn[m+2] == 's'){
//					complexity=complexity+3;
//					m=m+2;
//				}
//			}
//		}
//		else if (c=='e'){
//			if(m+2<eqn.size()){
//				if ( eqn[m+1]=='x' && eqn[m+2] == 'p'){
//					complexity=complexity+4;
//					m=m+2;
//				}
//			}
//		}
//		else if (c=='l'){
//			if(m+2<eqn.size()){
//				if ( eqn[m+1]=='o' && eqn[m+2] == 'g'){
//					complexity=complexity+4;
//					m=m+2;
//				}
//			}
//		}
//		else if (isalpha(c) && (m+1)<eqn.size()){
//			bool pass=true;
//			while ((m+1)<eqn.size() && pass){
//				if (isalpha(eqn[m+1])) m++; 
//				else pass=0;
//			}
//			complexity++;
//		}
//		else
//			complexity++;
//	}
//
//	return complexity;
//}
//void eval(node& n,vector<float>& outstack)
//{
//	switch(n.type) 
//	{
//	case 'n':
//		outstack.push_back(n.value);
//		break;
//	case 'v':
//		if (n.valpt==NULL)
//			cout<<"problem";
//		else
//			outstack.push_back(*n.valpt);
//		break;
//	case '+':
//		if(outstack.size()>=2){
//				float n1 = outstack.back(); outstack.pop_back();
//				float n2 = outstack.back(); outstack.pop_back();
//
//				outstack.push_back(n2+n1);
//		}
//		break;
//	case '-':
//		if(outstack.size()>=2){
//				float n1 = outstack.back(); outstack.pop_back();
//				float n2 = outstack.back(); outstack.pop_back();
//
//				outstack.push_back(n2-n1);
//		}
//		break;
//	case '*':
//		if(outstack.size()>=2){
//				float n1 = outstack.back(); outstack.pop_back();
//				float n2 = outstack.back(); outstack.pop_back();
//
//				outstack.push_back(n2*n1);
//		}
//		break;
//	case '/':
//		if(outstack.size()>=2){
//				float n1 = outstack.back(); outstack.pop_back();
//				float n2 = outstack.back(); outstack.pop_back();
//				if(abs(n1)<0.0001)
//					outstack.push_back(0);
//				else
//					outstack.push_back(n2/n1);
//		}
//		break;
//	case 's':
//		if(outstack.size()>=1){
//				float n1 = outstack.back(); outstack.pop_back();
//				outstack.push_back(sin(n1));
//		}
//		break;
//	case 'c':
//		if(outstack.size()>=1){
//				float n1 = outstack.back(); outstack.pop_back();
//				outstack.push_back(cos(n1));
//		}
//		break;
//	case 'e':
//		if(outstack.size()>=1){
//				float n1 = outstack.back(); outstack.pop_back();
//				outstack.push_back(exp(n1));
//		}
//		break;
//	case 'l':
//		if(outstack.size()>=1){
//				float n1 = outstack.back(); outstack.pop_back();
//				if (abs(n1)<0.0001)
//					outstack.push_back(0);
//				else
//					outstack.push_back(log(n1));
//		}
//		break;
//	}
//}

