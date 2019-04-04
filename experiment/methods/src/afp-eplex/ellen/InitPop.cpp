#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include "general_fns.h"
#include "InitPop.h"
//#include "RPN_class.h"
//extern params p;
//extern vector<Randclass> r;
using namespace std;
/*Initialize population
		make genotypes
		genotype to phenotype
		calculate fitness
*/


void InitPop(vector<ind> &pop,params& p, vector<Randclass>& r)
{
	//boost::progress_timer timer;
	//#pragma omp parallel for
	// std::cout << "p.min_len: " << p.min_len;
  // std::cout << "p.max_len_init: " << p.max_len_init;
	// cout << "p.allvars size: " << p.allvars.size();
	for(int i=0;i<pop.size();++i)
	{
		// WGL: need to edit this to work with EHC and classification both ON

		if (!p.init_trees){ // random initialization
			makeline(pop.at(i),p,r);
			if (p.eHC_on)
			{
				for (int j=0;j<pop.at(i).line.size();++j)
				{
					float tmp = r[omp_get_thread_num()].rnd_flt(0,1);
					//cout << "tmp: " << tmp << " p.eHC: " << p.eHC_init;
					if (tmp > p.eHC_init)
					{
						pop.at(i).line[j].on=false;
					}
				}
			}
		}
		else{ // init trees
			int linelen = r[omp_get_thread_num()].rnd_int(p.min_len,p.max_len_init);
			// cout << "linelen:" << linelen << "\n";
			if (p.eHC_on){
				int onlen = max(p.min_len, int(linelen*p.eHC_init));
				makeline_rec(pop.at(i).line,p,r,onlen);
				int offlen = linelen-onlen;
				for (int j=0;j<offlen;++j){
					int loc = r[omp_get_thread_num()].rnd_int(0,pop.at(i).line.size()-1);
					 InsInstruction(pop.at(i),loc,p,r);
					pop.at(i).line[loc].on=false;
				}
			}
			else if (p.classification){
				// choose dimensions of trees between 1 and max_len_init/min_len
				int dims;
                
				if (p.class_m4gp) {
					dims = r[omp_get_thread_num()].rnd_int(1, linelen);
				}
				else
					dims = p.number_of_classes;
                //cout << "dims: " << dims << "\n";

				int remain_len = linelen;
				vector<node> tmp_line;
				//int default_size = remain_len/dims;
				// create dims trees and push them together
				for (unsigned j=0; j<dims; ++j){
					int treelen = remain_len/(dims-j);// + r[omp_get_thread_num()].gasdev();
					if (treelen<1)
						treelen=1;
					else if (treelen>p.max_len_init)
						treelen=p.max_len_init;
                    //cout << "making tree of length " << treelen << "\n";
					makeline_rec(tmp_line,p,r,treelen);
					pop[i].line.insert(pop[i].line.end(),tmp_line.begin(),tmp_line.end());
					remain_len -= tmp_line.size();
					tmp_line.clear();
				}


			}
			else // make normal trees
				makeline_rec(pop.at(i).line,p,r,linelen);
		}
		//remove dangling numbers and variables from end
		/*while((pop.at(i).line.back()->type=='n' || pop.at(i).line.back()->type=='v') && pop.at(i).line.size()>1)
			pop.at(i).line.pop_back();*/

		pop.at(i).origin = 'i';

		assert(!pop.at(i).line.empty());
		//assert(pop[i].line.size() >= p.min_len);

	}
	// cout <<"\nInit Pop done ";
}

void makeline(ind& newind,params& p,vector<Randclass>& r)
{
	// construct line
	int linelen = r[omp_get_thread_num()].rnd_int(p.min_len,p.max_len_init);

	int choice=0;

	//uniform_int_distribution<int> dist(0,21);
	vector<float> wheel(p.op_weight.size());
	//wheel.resize(p.op_weight.size());
	if (p.weight_ops_on) //fns are weighted
	{
		partial_sum(p.op_weight.begin(), p.op_weight.end(), wheel.begin());
	}

	for (int x = 0; x<linelen; x++)
	{
		if (p.weight_ops_on) //fns are weighted
		{
			float tmp = r[omp_get_thread_num()].rnd_flt(0,1);
			if (tmp < wheel.at(0))
				choice=0;
			else
			{
				for (unsigned int k=1;k<wheel.size();++k)
				{
					if(tmp<wheel.at(k) && tmp>=wheel.at(k-1)){
						choice = k;
						break;
					}
				}
			}
		}
		else
			choice = r[omp_get_thread_num()].rnd_int(0,p.op_choice.size()-1);

		string varchoice;
		int seedchoice;
		vector<node> tmpstack;

		switch (p.op_choice.at(choice))
			{
			case 0: //load number
				if(p.ERC){ // if ephemeral random constants are on
					if (!p.cvals.empty()){ // if there are constants defined by the user
						if (r[omp_get_thread_num()].rnd_flt(0,1)<.5)
							//newind.line.push_back(shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
							newind.line.push_back(node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
						else{
							if(p.ERCints)
								//newind.line.push_back(shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));
								newind.line.push_back(node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
							else
								//newind.line.push_back(shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
								newind.line.push_back(node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
						}
					}
					else{
						if(p.ERCints)
								//newind.line.push_back(shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));
						newind.line.push_back(node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
							else
								//newind.line.push_back(shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
						newind.line.push_back(node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
					}
				}
				else if (!p.cvals.empty()) // if ERCs are off, but there are constants defined by the user
				{
					//newind.line.push_back(shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
					newind.line.push_back(node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
				}
				break;
			case 1: //load variable
				varchoice = p.allvars.at(r[omp_get_thread_num()].rnd_int(0,p.allvars.size()-1));
				//varchoice = d.label.at(r[omp_get_thread_num()].rnd_int(0,d.label.size()-1));
				newind.line.push_back(node(varchoice));
				break;
			case 2: // +
				//newind.line.push_back(shared_ptr<node>(new n_add()));
				newind.line.push_back(node('+'));
				break;
			case 3: // -
				//newind.line.push_back(shared_ptr<node>(new n_sub()));
				newind.line.push_back(node('-'));
				break;
			case 4: // *
				//newind.line.push_back(shared_ptr<node>(new n_mul()));
				newind.line.push_back(node('*'));
				break;
			case 5: // /
				//newind.line.push_back(shared_ptr<node>(new n_div()));
				newind.line.push_back(node('/'));
				break;
			case 6: // sin
				//newind.line.push_back(shared_ptr<node>(new n_sin()));
				newind.line.push_back(node('s'));
				break;
			case 7: // cos
				//newind.line.push_back(shared_ptr<node>(new n_cos()));
				newind.line.push_back(node('c'));
				break;
			case 8: // exp
				//newind.line.push_back(shared_ptr<node>(new n_exp()));
				newind.line.push_back(node('e'));
				break;
			case 9: // log
				//newind.line.push_back(shared_ptr<node>(new n_log()));
				newind.line.push_back(node('l'));
				break;
			case 10: // seed
				seedchoice = r[omp_get_thread_num()].rnd_int(0,p.seedstacks.size()-1);
				//copystack(p.seedstacks.at(seedchoice),tmpstack);
				tmpstack = p.seedstacks.at(seedchoice);

				for(int i=0;i<tmpstack.size(); ++i)
				{
					if (x<p.max_len_init){
						newind.line.push_back(tmpstack[i]);
						x++;
					}
				}
				tmpstack.clear();
				break;
			case 11: //sqrt
				newind.line.push_back(node('q'));
				break;
			case 12: //square
				newind.line.push_back(node('2'));
				break;
			case 13: //cube
				newind.line.push_back(node('3'));
				break;
			case 14: //power
				newind.line.push_back(node('^'));
				break;
			case 15: // equals
				newind.line.push_back(node('='));
				break;
			case 16: // does not equal
				newind.line.push_back(node('!'));
				break;
			case 17: // less than
				newind.line.push_back(node('<'));
				break;
			case 18: // greater than
				newind.line.push_back(node('>'));
				break;
			case 19: //less than or equal to
				newind.line.push_back(node('{'));
				break;
			case 20: //greater than or equal to
				newind.line.push_back(node('}'));
				break;
			case 21: //if-then
				newind.line.push_back(node('i'));
				break;
			case 22: //if-then-else
				newind.line.push_back(node('t'));
				break;
			case 23: // and
				newind.line.push_back(node('&'));
				break;
			case 24: //or
				newind.line.push_back(node('|'));
				break;
			}
	}
	/*while (newind.line.back()->type=='n'||newind.line.back()->type=='v')
		newind.line.pop_back();*/
}
void makeline_rec(vector<node>& line,params& p,vector<Randclass>& r, int linelen)
{
	// recursive version of makeline that creates lines that are complete with respect to stack operations
	// in other words the entire line is guaranteed to be a valid syntax tree
	// construct line



	int choice=0;
	// set output type based on problem definition ('f' = float, 'b' = boolean)
	char type = 'f';
	if (p.classification && p.class_bool)
		type = 'b';

	int tmp = maketree(line,linelen,1,0,type,p,r);

}
void getChoice(int& choice, int min_arity, int max_arity, char type, params& p,vector<Randclass>& r)
{
	vector<int> choices;
	vector<float> op_weight;
	int tmpchoice;
	for (int i=0;i<p.op_arity.size();++i)
	{
		if (p.op_arity[i] >= min_arity && p.op_arity[i] <= max_arity && p.return_type[i]==type)
		{
			choices.push_back(i);
			if(p.weight_ops_on)
				op_weight.push_back(p.op_weight[i]);
		}
	}
	if(!choices.empty()){
		vector<float> wheel(choices.size());
		//wheel.resize(p.op_weight.size());
		if (p.weight_ops_on) //fns are weighted
			partial_sum(op_weight.begin(), op_weight.end(), wheel.begin());

		if (p.weight_ops_on) //fns are weighted
		{
			float tmp = r[omp_get_thread_num()].rnd_flt(0,*std::max_element(wheel.begin(),wheel.end()));
			if (tmp < wheel.at(0))
				tmpchoice=0;
			else
			{
				for (unsigned int k=1;k<wheel.size();++k)
				{
					if(tmp<wheel.at(k) && tmp>=wheel.at(k-1)){
						tmpchoice = k;
						break;
					}
				}
			}
		}
		else{
			// cout << "choices size: " << choices.size();
			tmpchoice =r[omp_get_thread_num()].rnd_int(0,choices.size()-1);
		}

		choice = choices[tmpchoice];
	}
	else
		choice = -1;

}

void push_back_node(vector <node>& line, int choice, params& p,vector<Randclass>& r)
{
	string varchoice;

	switch (p.op_choice.at(choice))
	{
		case 0: //load number
			if(p.ERC){ // if ephemeral random constants are on
				if (!p.cvals.empty()){ // if there are constants defined by the user
					if (r[omp_get_thread_num()].rnd_flt(0,1)<.5)
						//line.push_back(shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
						line.push_back(node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
					else{
						if(p.ERCints)
							/*line.push_back(shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));*/
							line.push_back(node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
						else
							//line.push_back(shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
							line.push_back(node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
					}
				}
				else{
					if(p.ERCints)
							//line.push_back(shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));
							line.push_back(node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
						else
							//line.push_back(shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
							line.push_back(node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
				}
			}
			else if (!p.cvals.empty()) // if ERCs are off, but there are constants defined by the user
			{
				//line.push_back(shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
				line.push_back(node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
			}
			break;
		case 1: //load variable
			varchoice = p.allvars.at(r[omp_get_thread_num()].rnd_int(0,p.allvars.size()-1));
			//varchoice = d.label.at(r[omp_get_thread_num()].rnd_int(0,d.label.size()-1));
			/*line.push_back(shared_ptr<node>(new n_sym(varchoice)));*/
			line.push_back(node(varchoice));
			break;
		case 2: // +
			//line.push_back(shared_ptr<node>(new n_add()));
			line.push_back(node('+'));
			break;
		case 3: // -
			//line.push_back(shared_ptr<node>(new n_sub()));
			line.push_back(node('-'));
			break;
		case 4: // *
			//line.push_back(shared_ptr<node>(new n_mul()));
			line.push_back(node('*'));
			break;
		case 5: // /
			//line.push_back(shared_ptr<node>(new n_div()));
			line.push_back(node('/'));
			break;
		case 6: // sin
			//line.push_back(shared_ptr<node>(new n_sin()));
			line.push_back(node('s'));
			break;
		case 7: // cos
			//line.push_back(shared_ptr<node>(new n_cos()));
			line.push_back(node('c'));
			break;
		case 8: // exp
			//line.push_back(shared_ptr<node>(new n_exp()));
			line.push_back(node('e'));
			break;
		case 9: // log
			//line.push_back(shared_ptr<node>(new n_log()));
			line.push_back(node('l'));
			break;
		case 11: // sqrt
			//line.push_back(shared_ptr<node>(new n_log()));
			line.push_back(node('q'));
			break;
		case 12: // square
			//line.push_back(shared_ptr<node>(new n_log()));
			line.push_back(node('2'));
			break;
		case 13: // cube
			//line.push_back(shared_ptr<node>(new n_log()));
			line.push_back(node('3'));
			break;
		case 14: //power
			line.push_back(node('^'));
			break;
		case 15: // equals
			line.push_back(node('='));
			break;
		case 16: // does not equal
			line.push_back(node('!'));
			break;
		case 17: // less than
			line.push_back(node('<'));
			break;
		case 18: // greater than
			line.push_back(node('>'));
			break;
		case 19: //less than or equal to
			line.push_back(node('{'));
			break;
		case 20: //greater than or equal to
			line.push_back(node('}'));
			break;
		case 21: //if-then
			line.push_back(node('i'));
			break;
		case 22: //if-then-else
			line.push_back(node('t'));
			break;
		case 23: // and
			line.push_back(node('&'));
			break;
		case 24: //or
			line.push_back(node('|'));
			break;
	}
}
void push_front_node(vector <node>& line, int choice, params& p,vector<Randclass>& r)
{
	string varchoice;

	switch (p.op_choice.at(choice))
	{
		case 0: //load number
			if(p.ERC){ // if ephemeral random constants are on
				if (!p.cvals.empty()){ // if there are constants defined by the user
					if (r[omp_get_thread_num()].rnd_flt(0,1)<.5)
						//line.insert(line.begin(),shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
						line.insert(line.begin(),node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
					else{
						if(p.ERCints)
							/*line.insert(line.begin(),shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));*/
							line.insert(line.begin(),node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
						else
							//line.insert(line.begin(),shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
							line.insert(line.begin(),node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
					}
				}
				else{
					if(p.ERCints)
							//line.insert(line.begin(),shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));
							line.insert(line.begin(),node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
						else
							//line.insert(line.begin(),shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
							line.insert(line.begin(),node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
				}
			}
			else if (!p.cvals.empty()) // if ERCs are off, but there are constants defined by the user
			{
				//line.insert(line.begin(),shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
				line.insert(line.begin(),node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
			}
			break;
		case 1: //load variable
			varchoice = p.allvars.at(r[omp_get_thread_num()].rnd_int(0,p.allvars.size()-1));
			//varchoice = d.label.at(r[omp_get_thread_num()].rnd_int(0,d.label.size()-1));
			/*line.insert(line.begin(),shared_ptr<node>(new n_sym(varchoice)));*/
			line.insert(line.begin(),node(varchoice));
			break;
		case 2: // +
			//line.insert(line.begin(),shared_ptr<node>(new n_add()));
			line.insert(line.begin(),node('+'));
			break;
		case 3: // -
			//line.insert(line.begin(),shared_ptr<node>(new n_sub()));
			line.insert(line.begin(),node('-'));
			break;
		case 4: // *
			//line.insert(line.begin(),shared_ptr<node>(new n_mul()));
			line.insert(line.begin(),node('*'));
			break;
		case 5: // /
			//line.insert(line.begin(),shared_ptr<node>(new n_div()));
			line.insert(line.begin(),node('/'));
			break;
		case 6: // sin
			//line.insert(line.begin(),shared_ptr<node>(new n_sin()));
			line.insert(line.begin(),node('s'));
			break;
		case 7: // cos
			//line.insert(line.begin(),shared_ptr<node>(new n_cos()));
			line.insert(line.begin(),node('c'));
			break;
		case 8: // exp
			//line.insert(line.begin(),shared_ptr<node>(new n_exp()));
			line.insert(line.begin(),node('e'));
			break;
		case 9: // log
			//line.insert(line.begin(),shared_ptr<node>(new n_log()));
			line.insert(line.begin(),node('l'));
			break;
		case 11: // sqrt
			line.insert(line.begin(),node('q'));
			break;
		case 12: //  square
			line.insert(line.begin(),node('2'));
			break;
		case 13: // cube
			line.insert(line.begin(),node('3'));
			break;
		case 14: // power
			line.insert(line.begin(),node('^'));
			break;
		case 15: // equals
			line.insert(line.begin(),node('='));
			break;
		case 16: // does not equal
			line.insert(line.begin(),node('!'));
			break;
		case 17: // less than
			line.insert(line.begin(),node('<'));
			break;
		case 18: // greater than
			line.insert(line.begin(),node('>'));
			break;
		case 19: //less than or equal to
			line.insert(line.begin(),node('{'));
			break;
		case 20: //greater than or equal to
			line.insert(line.begin(),node('}'));
			break;
		case 21: //if-then
			line.insert(line.begin(),node('i'));
			break;
		case 22: //if-then-else
			line.insert(line.begin(),node('t'));
			break;
		case 23: //and
			line.insert(line.begin(),node('&'));
			break;
		case 24: //or
			line.insert(line.begin(),node('|'));
			break;
	}
}
int maketree(vector<node>& line, int level, bool exactlevel, int lastnode,char type,params& p,vector<Randclass>& r)
{
	int choice;
//	int splitnodes;
	int thisnode = lastnode+1;
	int startsize = line.size();
	if (level == 1) { // choose a terminal because of the level limitation
		getChoice(choice, 0, 0, type, p, r);
	}
	else if (exactlevel){
		// choose an instruction other than a terminal with arity <= level-1
		getChoice(choice,1,level-1,type,p,r);
		if (choice==-1)
			getChoice(choice,0,0,type,p,r);
	}
	else
		getChoice(choice,0,level-1,type,p,r);
	while (choice == -1 && level <= p.max_len)
		getChoice(choice, 0, ++level-1, type, p, r);
	if (choice == -1)
		cout << "bug\n";
	// insert choice into line
	//cout << "choice: " << p.op_choice[choice] << "\n";
        
    push_front_node(line,choice,p,r);
	//int a = p.op_arity[choice];
	int a = line[0].arity();
	vector<char> types(a);
	for (size_t i = 0; i < a; ++i)
	{
		if (i < line[0].arity_float)
			types[i] = 'f';
		else
			types[i] = 'b';
	}
	int newlevel;
	int nodes=0;
	if (a !=0){
		//level = max(2,level-1);
		level = level - 1; // subtract one from level to represent the added node
	}
	/*else if (a == 0)
		cout << "debug";*/


	for (int i=1;i<=a;++i)
	{
		if (i==a)
			newlevel = level-nodes;
		else {
			// make subtree approximately balanced with normal distribution
			newlevel = int((level - 1) / a + r[omp_get_thread_num()].gasdev());
			newlevel = min(newlevel, level - 1);
			newlevel = max(newlevel, 1);
			//newlevel = r[omp_get_thread_num()].rnd_int(1, level - 1);
		}

		nodes += maketree(line,newlevel,exactlevel,thisnode,types[i-1],p,r);
	}
	//for (int i = 1; i <= a_b; ++i)
	//{
	//	if (i == a_b)
	//		newlevel = level - nodes;
	//	else
	//		newlevel = r[omp_get_thread_num()].rnd_int(1, level - 1);
	//	nodes += maketree(line, newlevel, exactlevel, thisnode, 'b', p, r);
	//}
	return line.size()-startsize;
}
//void makestack(ind& newind,params& p,vector<Randclass>& r)
//{
//	// construct line
//	// obtain a seed from the system clock:
//
//	// random number generator
//	//mt19937_64 engine(p.seed);
//
//	int linelen = r[omp_get_thread_num()].rnd_int(p.min_len,p.max_len_init);
//
//	vector <string> load_choices(p.allblocks);
//
//	int choice=0;
//
//	//uniform_int_distribution<int> dist(0,21);
//	if (p.ERC)
//	{
//		float ERC;
//		for (int j=0;j<p.numERC;++j)
//		{
//			ERC = r[omp_get_thread_num()].rnd_flt((float)p.minERC,(float)p.maxERC);
//			if(p.ERCints)
//				load_choices.push_back(to_string(static_cast<long long>(ERC)));
//			else
//				load_choices.push_back(to_string(static_cast<long double>(ERC)));
//		}
//	}
//	vector<float> wheel(p.op_weight.size());
//	//wheel.resize(p.op_weight.size());
//	if (p.weight_ops_on) //fns are weighted
//	{
//		partial_sum(p.op_weight.begin(), p.op_weight.end(), wheel.begin());
//	}
//
//	for (int x = 0; x<linelen; x++)
//	{
//		if (p.weight_ops_on) //fns are weighted
//		{
//			float tmp = r[omp_get_thread_num()].rnd_flt(0,1);
//			if (tmp < wheel.at(0))
//				choice=0;
//			else
//			{
//				for (unsigned int k=1;k<wheel.size();++k)
//				{
//					if(tmp<wheel.at(k) && tmp>=wheel.at(k-1))
//						choice = k;
//				}
//			}
//		}
//		else
//			choice = r[omp_get_thread_num()].rnd_int(0,p.op_choice.size()-1);
//
//
//		if (choice==0) // number reference to argument from all blocks in next position in line
//		{
//			if(p.ERCints)
//				newind.line.push_back(ops(p.op_choice.at(choice),r[omp_get_thread_num()].rnd_int((float)p.minERC,(float)p.maxERC)));
//			else
//				newind.line.push_back(ops(p.op_choice.at(choice),r[omp_get_thread_num()].rnd_flt((float)p.minERC,(float)p.maxERC)));
//		}
//		else if (choice==1) // pick pointer values from mapped variables in data struct
//		{
//		}
//		else
//			newind.line.push_back(ops(p.op_choice.at(choice)));
//	}
//}
