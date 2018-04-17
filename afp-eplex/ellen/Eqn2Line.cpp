#include "stdafx.h"
#include "op_node.h"
#include <exception>
#include <string>
#include <unordered_map>

using namespace std;


enum {ASSOC_NONE=0, ASSOC_LEFT, ASSOC_RIGHT};

struct op_s {
	std::string op;
	int prec;
	int assoc;
	~op_s() {}
} ops[]={
	{"_", 11, ASSOC_RIGHT},
	{"^", 9, ASSOC_RIGHT},
	{"*", 8, ASSOC_LEFT},
	{"/", 8, ASSOC_LEFT},
	{"%", 8, ASSOC_LEFT},
	{"+", 5, ASSOC_LEFT},
	{"-", 5, ASSOC_LEFT},
	{"(", 0, ASSOC_NONE},
	{")", 0, ASSOC_NONE},
	{"sin",10,ASSOC_RIGHT},
	{"cos",10,ASSOC_RIGHT},
	{"exp",10,ASSOC_RIGHT},
	{"log",10,ASSOC_RIGHT}
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
void op2node(string& ch,vector<node>& eqnstack)
{

if(ch.compare("+")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_add())); 
	eqnstack.push_back(node('+'));
else if (ch.compare("-")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_sub()));
	eqnstack.push_back(node('-'));
else if(ch.compare("/")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_div()));
	eqnstack.push_back(node('/'));
else if (ch.compare("*")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_mul()));
	eqnstack.push_back(node('*'));
else if(ch.compare("sin")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_sin()));
	eqnstack.push_back(node('s'));
else if (ch.compare("cos")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_cos()));
	eqnstack.push_back(node('c'));
else if (ch.compare("exp")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_exp()));
	eqnstack.push_back(node('e'));
else if (ch.compare("log")==0)
	//eqnstack.push_back(shared_ptr<node>(new n_log()));
	eqnstack.push_back(node('l'));
else 
	cout << "op2node failed.\n";
}
void shunt_op(struct op_s *op,vector <op_s*>& opstack,vector<float>& numstack,vector<node>& eqnstack,string& RPN)
{
	struct op_s *pop;
//	float n1, n2;
	if(op->op.compare("(")==0) {
		opstack.push_back(op);
		return;
	} else if(op->op.compare(")")==0) {
		while(opstack.size()>0 && opstack[opstack.size()-1]->op.compare("(")!=0) {
			pop=opstack.back(); 
			RPN +=opstack.back()->op;
			op2node(opstack.back()->op,eqnstack);
			opstack.pop_back();
			/*n1=numstack.back(); numstack.pop_back();

			if(pop->unary) numstack.push_back(pop->eval(n1, 0));
			else {
				n2=numstack.back(); numstack.pop_back();
				numstack.push_back(pop->eval(n2, n1));
			}*/
		}
		if(!(pop=pop_opstack(opstack)) || pop->op.compare("(")!=0) {
			fprintf(stderr, "ERROR: Stack error. No matching \'(\'\n");
			exit(EXIT_FAILURE);
		}
		return;
	}

	if(op->assoc==ASSOC_RIGHT) {
		while(opstack.size() && op->prec<opstack[opstack.size()-1]->prec) {
			pop=opstack.back(); 
			RPN +=opstack.back()->op;
			op2node(opstack.back()->op,eqnstack);
			opstack.pop_back();
			/*n1=numstack.back(); numstack.pop_back();
			if(pop->unary) numstack.push_back(pop->eval(n1, 0));
			else {
				n2=numstack.back(); numstack.pop_back();
				numstack.push_back(pop->eval(n2, n1));
			}*/
		}
	} else {
		while(opstack.size() && op->prec<=opstack[opstack.size()-1]->prec) {
			pop=opstack.back(); 
			RPN +=opstack.back()->op;
			op2node(opstack.back()->op,eqnstack);
			opstack.pop_back();
			/*n1=numstack.back(); numstack.pop_back();
			if(pop->unary) numstack.push_back(pop->eval(n1, 0));
			else {
				n2=numstack.back(); numstack.pop_back();
				numstack.push_back(pop->eval(n2, n1));
			}*/
		}
	}
	opstack.push_back(op);
	
}
string getnum(const string &s,unsigned int& i)
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
		++count;		
	}
	i += count-1;
    return s.substr(0,count);
}

string getdata(unordered_map<string,float*>& mymap, string &s,unsigned int& i)
{ //return data from data table

	std::unordered_map<string,float*>::iterator it;
	
	for(unsigned int j=0;j<=s.size();++j)
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


void Eqn2Line(string& expr,vector<node>& eqnstack)
{
	string RPN;
	string tstart;
	struct op_s startop={"X", 0, ASSOC_NONE};	/* Dummy operator to mark start */
	struct op_s *op=NULL;
//	float n1, n2;
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
				shunt_op(op,opstack,numstack,eqnstack,RPN);
				lastop=op;
			} 
			else if(isdigit(expr[i]) || expr[i]=='.') 
				tstart=getnum(expr.substr(i,expr.size()-i),i);
			else if(is_letter(expr[i]))//data variable
			{
				tstart=expr[i];
				bool tmp = 1;				
				while(i+1<expr.size() && tmp){
					if (is_letter(expr[i+1])){
						tstart+=expr[i+1];
						++i;
					}
					else tmp=0;
				}
			}
				//getdata(datatable,expr.substr(i,expr.size()-i),i);
			else if(!isspace(expr[i])) 
			{
				fprintf(stderr, "ERROR: Syntax error\n");
				//return EXIT_FAILURE;
			}
		} 
		else 
		{
			if(isspace(expr[i])) 
			{
				//numstack.push_back(stof(tstart));
				RPN+=tstart;
				if (isdigit(tstart[0]))
					//eqnstack.push_back(shared_ptr<node>(new n_num(stof(tstart))));
					eqnstack.push_back(node(stof(tstart)));
				else if (is_letter(tstart[0]))
					//eqnstack.push_back(shared_ptr<node>(new n_sym(tstart)));
					eqnstack.push_back(node(tstart));
				else
					cout << "oops.\n";
				tstart.clear();
				lastop=NULL;
			} 
			else if((op=getop(expr.substr(i,expr.size()-i),i)))
			{
				//push_numstack(atoi(tstart));
				//numstack.push_back(stof(tstart));
				RPN+=tstart;

				if (isdigit(tstart[0]))
					//eqnstack.push_back(shared_ptr<node>(new n_num(stof(tstart))));
					eqnstack.push_back(node(stof(tstart)));
				else if (is_letter(tstart[0]))
					//eqnstack.push_back(shared_ptr<node>(new n_sym(tstart)));
					eqnstack.push_back(node(tstart));
				else
					cout << "oops.\n";

				tstart.clear();
				shunt_op(op,opstack,numstack,eqnstack,RPN);
				lastop=op;
			} 
			else if(!isdigit(expr[i])) 
			{
				fprintf(stderr, "ERROR: Syntax error\n");
			//	return EXIT_FAILURE;
			}
		}
	}
	if(!tstart.empty()) {
		//numstack.push_back(stof(tstart));
		RPN+=tstart;
		/*eqnstack.push_back(shared_ptr<node>(new n_num(stof(tstart))));*/
		eqnstack.push_back(node(stof(tstart)));
	}

	while(opstack.size()) {
		op=opstack.back(); 
		RPN+=op->op;
		op2node(opstack.back()->op,eqnstack);
		opstack.pop_back();
	}

}


