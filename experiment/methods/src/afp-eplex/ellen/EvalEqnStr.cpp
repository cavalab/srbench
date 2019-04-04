// ShuntTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <tchar.h>
#include<stdlib.h>
#include<ctype.h>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
using namespace std;

#define MAXOPSTACK 64
#define MAXNUMSTACK 64

float eval_uminus(float a1, float a2) 
{
	return -a1;
}
float eval_exponent(float a1, float a2)
{
	return a2<0 ? 0 : (a2==0?1:a1*eval_exponent(a1, a2-1));
}
float eval_mul(float a1, float a2) 
{
	return a1*a2;
}
float eval_div(float a1, float a2) 
{
	if(!a2) {
		return 0;
		//fprintf(stderr, "ERROR: Division by zero\n");
		//exit(EXIT_FAILURE);
	}
	return a1/a2;
}
float eval_mod(float a1, float a2) 
{
	if(!a2) {
		//fprintf(stderr, "ERROR: Division by zero\n");
		//exit(EXIT_FAILURE);
		return 0;
	}
	return float(int(a1)%int(a2));
}
float eval_add(float a1, float a2) 
{
	return a1+a2;
}
float eval_sub(float a1, float a2) 
{
	return a1-a2;
}
float eval_sin(float a1, float a2)
{
	return sin(a1);
}
float eval_cos(float a1, float a2)
{
	return cos(a1);
}
float eval_exp(float a1, float a2)
{
	return exp(a1);
}
float eval_log(float a1, float a2)
{
	return log(a1);
}

enum {ASSOC_NONE=0, ASSOC_LEFT, ASSOC_RIGHT};

struct op_s {
	std::string op;
	int prec;
	int assoc;
	bool unary;
	float (*eval)(float a1, float a2);
	~op_s() {}
} ops[]={
	{"_", 11, ASSOC_RIGHT, 1, eval_uminus},
	{"^", 9, ASSOC_RIGHT, 0, eval_exponent},
	{"*", 8, ASSOC_LEFT, 0, eval_mul},
	{"/", 8, ASSOC_LEFT, 0, eval_div},
	{"%", 8, ASSOC_LEFT, 0, eval_mod},
	{"+", 5, ASSOC_LEFT, 0, eval_add},
	{"-", 5, ASSOC_LEFT, 0, eval_sub},
	{"(", 0, ASSOC_NONE, 0, NULL},
	{")", 0, ASSOC_NONE, 0, NULL},
	{"sin",10,ASSOC_RIGHT,1,eval_sin},
	{"cos",10,ASSOC_RIGHT,1,eval_cos},
	{"exp",10,ASSOC_RIGHT,1,eval_exp},
	{"log",10,ASSOC_RIGHT,1,eval_log}
};

struct op_s *getop(string ch,unsigned int& p)
{
	
	int i;
	for(i=0; i<sizeof ops/sizeof ops[0]; ++i) {
		if(ops[i].op[0]==ch[0]) 
		{
			if (ops[i].op.size()==1)
				return ops+i;
			else if (ops[i].op.compare(ch.substr(0,ops[i].op.size()))==0)
			{
				p += ops[i].op.size()-1;
				return ops+i;
			}
		}

	}
	return NULL;
}




struct op_s *pop_opstack(vector <op_s*>& opstack)
{
	if(opstack.empty()) {
		fprintf(stderr, "ERROR: Operator stack empty\n");
		exit(EXIT_FAILURE);
	}
	struct op_s *op_tmp = opstack.back();
	opstack.pop_back();
	return op_tmp;
}

void shunt_op(struct op_s *op,vector <op_s*>& opstack,vector<float>& numstack)
{
	struct op_s *pop;
	float n1, n2;
	if(op->op.compare("(")==0) {
		opstack.push_back(op);
		return;
	} else if(op->op.compare(")")==0) {
		while(opstack.size()>0 && opstack[opstack.size()-1]->op.compare("(")!=0) {
			pop=opstack.back(); opstack.pop_back();
			n1=numstack.back(); numstack.pop_back();
			if(pop->unary) numstack.push_back(pop->eval(n1, 0));
			else {
				n2=numstack.back(); numstack.pop_back();
				numstack.push_back(pop->eval(n2, n1));
			}
		}
		if(!(pop=pop_opstack(opstack)) || pop->op.compare("(")!=0) {
			fprintf(stderr, "ERROR: Stack error. No matching \'(\'\n");
			exit(EXIT_FAILURE);
		}
		return;
	}

	if(op->assoc==ASSOC_RIGHT) {
		while(opstack.size() && op->prec<opstack[opstack.size()-1]->prec) {
			pop=opstack.back(); opstack.pop_back();
			n1=numstack.back(); numstack.pop_back();
			if(pop->unary) numstack.push_back(pop->eval(n1, 0));
			else {
				n2=numstack.back(); numstack.pop_back();
				numstack.push_back(pop->eval(n2, n1));
			}
		}
	} else {
		while(opstack.size() && op->prec<=opstack[opstack.size()-1]->prec) {
			pop=opstack.back(); opstack.pop_back();
			n1=numstack.back(); numstack.pop_back();
			if(pop->unary) numstack.push_back(pop->eval(n1, 0));
			else {
				n2=numstack.back(); numstack.pop_back();
				numstack.push_back(pop->eval(n2, n1));
			}
		}
	}
	opstack.push_back(op);
}
string getnum(string &s,unsigned int& i)
{
	int count=0;
	std::string::const_iterator it = s.begin();
	while (it != s.end() && (isdigit(*it) || (*it=='.') || (*it=='e'))) 
	{ 
		if(*it=='e') {
			if(*(it+1)=='-')
			{
				it+=4;
				count+=4;
			}
			else
			{
				it+=3;
				count+=3;
			}
		}

		++it; 
		count++;		
	}
	i += count-1;
    return s.substr(0,count);
}

string getdata(unordered_map<string,float*>& mymap, string &s,unsigned int& i)
{ //return data from data table

	std::unordered_map<string,float*>::iterator it;
	
	for(unsigned int j=0;j<=s.size();j++)
	{
		it = mymap.find(s.substr(0,j));
	 if (it != mymap.end())
	 {
		 i+=j-1;
		return to_string(static_cast<long double>(*(it->second)));
	 }
	}
	 return NULL;
}
bool is_letter(char c)
{
    return (('a' <= c) && (c <= 'z')) || (('A' <= c) && (c <= 'Z'));
}



//int main(int argc, char *argv[])
float EvalEqnStr(string& expr,unordered_map<string,float*>& datatable)
{
	string tstart;
	struct op_s startop={"X", 0, ASSOC_NONE, 0, NULL};	/* Dummy operator to mark start */
	struct op_s *op=NULL;
	float n1, n2;
	struct op_s *lastop=&startop;

	std::vector <op_s*> opstack; 
	std::vector<float> numstack;
	
	for(unsigned int i =0; i<expr.size(); ++i) {
		if(tstart.empty()) 
		{
			if(op=getop(expr.substr(i,expr.size()-i),i)) 
			{
				if(lastop && (lastop==&startop || lastop->op.compare(")")!=0)) 
				{
					if(op->op.compare("-")==0) op=getop("_",i);
					else if(op->op.compare("(")!=0 && 
						    op->op.compare("sin")!=0 &&
							op->op.compare("cos")!=0 &&
							op->op.compare("exp")!=0 &&
							op->op.compare("log")!=0) 
					{

						cout<< "ERROR: Illegal use of binary operator " + op->op +"\n";



						exit(EXIT_FAILURE);
					}
				}
				shunt_op(op,opstack,numstack);
				lastop=op;
			} 
			else if(isdigit(expr[i]) || expr[i]=='.') 
				tstart=getnum(expr.substr(i,expr.size()-i),i);
			else if(is_letter(expr[i]))//data variable
				tstart=getdata(datatable,expr.substr(i,expr.size()-i),i);
			else if(!isspace(expr[i])) 
			{
				fprintf(stderr, "ERROR: Syntax error\n");
				return EXIT_FAILURE;
			}
		} 
		else 
		{
			if(isspace(expr[i])) 
			{
				numstack.push_back(stof(tstart));
				//push_numstack(atoi(tstart));
				tstart.clear();
				lastop=NULL;
			} 
			else if((op=getop(expr.substr(i,expr.size()-i),i)))
			{
				//push_numstack(atoi(tstart));
				numstack.push_back(stof(tstart));
				tstart.clear();
				shunt_op(op,opstack,numstack);
				lastop=op;
			} 
			else if(!isdigit(expr[i])) 
			{
				fprintf(stderr, "ERROR: Syntax error\n");
				return EXIT_FAILURE;
			}
		}
	}
	if(!tstart.empty()) 
		numstack.push_back(stof(tstart));

	while(opstack.size()) {
		op=opstack.back(); opstack.pop_back();
		n1=numstack.back(); numstack.pop_back();
		if(op->unary) 
			numstack.push_back(op->eval(n1, 0));
		else {
			n2=numstack.back(); numstack.pop_back();
			numstack.push_back(op->eval(n2, n1));
		}
	}
	if(numstack.size()!=1) {
		fprintf(stderr, "ERROR: Number stack has %d elements after evaluation. Should be 1.\n", numstack.size());
		return EXIT_FAILURE;
	}
	//printf("%f\n", numstack[0]);

	return numstack[0];
}


