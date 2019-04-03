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
/* The authors of this work have released all rights to it and placed it
in the public domain under the Creative Commons CC0 1.0 waiver
(http://creativecommons.org/publicdomain/zero/1.0/).

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Retrieved from: http://en.literateprograms.org/Shunting_yard_algorithm_(C)?oldid=18970
*/


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
		fprintf(stderr, "ERROR: Division by zero\n");
		exit(EXIT_FAILURE);
	}
	return a1/a2;
}
float eval_mod(float a1, float a2) 
{
	if(!a2) {
		fprintf(stderr, "ERROR: Division by zero\n");
		exit(EXIT_FAILURE);
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

//struct op_s *opstack[MAXOPSTACK];
std::vector <op_s*> opstack; 
//int nopstack=0;

std::vector<float> numstack;
//int nnumstack=0;

//void push_opstack(struct op_s *op)
//{
//	if(nopstack>MAXOPSTACK-1) {
//		fprintf(stderr, "ERROR: Operator stack overflow\n");
//		exit(EXIT_FAILURE);
//	}
//	opstack[nopstack++]=op;
//}

struct op_s *pop_opstack()
{
	if(opstack.empty()) {
		fprintf(stderr, "ERROR: Operator stack empty\n");
		exit(EXIT_FAILURE);
	}
	struct op_s *op_tmp = opstack.back();
	opstack.pop_back();
	return op_tmp;
}

//void push_numstack(int num)
//{
//	if(numstack.size()>MAXNUMSTACK-1) {
//		fprintf(stderr, "ERROR: Number stack overflow\n");
//		exit(EXIT_FAILURE);
//	}
//	numstack[numstack.size()++]=num;
//}

//int pop_numstack()
//{
//	if(!numstack.size()) {
//		fprintf(stderr, "ERROR: Number stack empty\n");
//		exit(EXIT_FAILURE);
//	}
//	return numstack[--nnumstack];
//}


void shunt_op(struct op_s *op)
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
		if(!(pop=pop_opstack()) || pop->op.compare("(")!=0) {
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
	while (it != s.end() && (isdigit(*it) || (*it=='.'))) { ++it; count++;};
	i += count-1;
    return s.substr(0,count);
}

string getdata(unoredered_map& myymap, string &s,unsigned int& i)
{ //return data from data table

	//unordered_map <string,int> mymap;
	////mymap.
	//mymap.insert(pair<string,int>("alpha",10));
	//mymap.insert(pair<string,int>("beta",20));
	//mymap.insert(pair<string,int>("G1",30));
	std::unordered_map<string,int*>::iterator it;
	
	for(unsigned int j=0;j<=s.size();j++)
	{
		it = mymap.find(s.substr(0,j));
	 if (it != mymap.end())
	 {
		 i+=j-1;
		return to_string(static_cast<long long>(it->(*second)));
	 }
	}
	 return NULL;
}
bool is_letter(char c)
{
    return (('a' <= c) && (c <= 'z')) || (('A' <= c) && (c <= 'Z'));
}
//int main(int argc, char *argv[])
void eval_eqn_str(string& expr,unordered_map& mymap)
{
	string expr = string(argv[1]);
	string tstart;
	struct op_s startop={"X", 0, ASSOC_NONE, 0, NULL};	/* Dummy operator to mark start */
	struct op_s *op=NULL;
	float n1, n2;
	struct op_s *lastop=&startop;

	if(argc<2) {
		fprintf(stderr, "Usage: %s <expression>\n", argv[0]);
		return EXIT_FAILURE;
	}

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
						fprintf(stderr, "ERROR: Illegal use of binary operator (%c)\n", op->op);
						exit(EXIT_FAILURE);
					}
				}
				shunt_op(op);
				lastop=op;
			} 
			else if(isdigit(expr[i]) || expr[i]=='.') 
				tstart=getnum(expr.substr(i,expr.size()-i),i);
			else if(is_letter(expr[i]))//data variable
				tstart=getdata(expr.substr(i,expr.size()-i),i);
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
				shunt_op(op);
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
	printf("%f\n", numstack[0]);

	return EXIT_SUCCESS;
}


