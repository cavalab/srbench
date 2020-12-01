#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include <locale>

float Round(float d)
{
  return floor(d + 0.5);
}

void find_root_nodes(vector<node>& line, vector<unsigned>& roots)
{
	// find "root" nodes of floating point program, where roots are final values that output something directly to the stack
	int total_arity = -1; //end node is always a root


	for (vector<node>::iterator i = line.end()-1; i!=line.begin(); --i){
		if ((*i).on){
			if (total_arity <= 0 ){ // root node
				roots.push_back(i-line.begin());
				total_arity=0;
			}
			else
				--total_arity;

			total_arity += (*i).arity_float;
		}
	}
	if (roots.empty())
		roots.push_back(line.size()-1);
}
bool is_number(const std::string& s)
{
	std::locale loc;
    std::string::const_iterator it = s.begin();
    while (it != s.end() && (std::isdigit(*it,loc) || (*it=='-') || (*it=='.'))) ++it;
    return !s.empty() && it == s.end();
}

void MutInstruction(ind& newind,int loc,params& p,vector<Randclass>& r,Data& d)
{

	vector <string> load_choices(p.allblocks);
	int choice=0;
	//cout<<"iterator definition\n";
	//std::vector<int>::iterator it;
	//it = newind.line.begin();
	//cout<<"end def \n";

	//uniform_int_distribution<int> dist(0,21);
	if (p.ERC)
	{
		float ERC;
		for (int j=0;j<p.numERC;++j)
		{
			ERC = r[omp_get_thread_num()].rnd_flt((float)p.minERC,(float)p.maxERC);
			if(p.ERCints)
				load_choices.push_back(to_string(static_cast<long long>(ERC)));
			else
				load_choices.push_back(to_string(static_cast<long double>(ERC)));
		}
	}
	vector<float> wheel(p.op_weight.size());
	//wheel.resize(p.op_weight.size());
	if (p.weight_ops_on) //fns are weighted
	{
		partial_sum(p.op_weight.begin(), p.op_weight.end(), wheel.begin());
	}

	if (p.weight_ops_on) //fns are weighted
	{
		float tmp = r[omp_get_thread_num()].rnd_flt(0,1);
		if (tmp < wheel.at(0))
			choice=0;
		else
		{
			for (unsigned int k=1;k<wheel.size();++k)
			{
				if(tmp<wheel.at(k) && tmp>=wheel.at(k-1))
					choice = k;
			}
		}
	}
	else
		choice = r[omp_get_thread_num()].rnd_int(0,p.op_list.size()-1);

		string varchoice;
//		int seedchoice;
		vector<node> tmpstack;
		switch (p.op_choice.at(choice))
		{
		case 0: //load number
			/*if(p.ERCints)
				newind.line.at(loc)=shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
			else
				newind.line.at(loc)=shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
			break;*/
			if(p.ERC){ // if ephemeral random constants are on
				if (!p.cvals.empty()){
					if (r[omp_get_thread_num()].rnd_flt(0,1)<.5)
						/*newind.line.at(loc)=(shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));*/
					newind.line.at(loc) = node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)));
					else{
						if(p.ERCints)
							/*newind.line.at(loc)=(shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));*/
              newind.line.at(loc)=node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC));
            else
							//newind.line.at(loc)=(shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
			        newind.line.at(loc)=node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC));
					}
				}
				else{
					if(p.ERCints)
							/*newind.line.at(loc)=(shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));*/
					newind.line.at(loc)=node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC));
						else
							//newind.line.at(loc)=(shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
							newind.line.at(loc)=node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC));
				}
			}
			else if (!p.cvals.empty())
			{
				/*newind.line.at(loc)=(shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));*/
				newind.line.at(loc)=node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)));
			}
			break;
		case 1: //load variable
			varchoice = d.label.at(r[omp_get_thread_num()].rnd_int(0,d.label.size()-1));
			//newind.line.at(loc)=shared_ptr<node>(new n_sym(varchoice));
			newind.line.at(loc)=node(varchoice);
			break;
		case 2: // +
			//newind.line.at(loc)=shared_ptr<node>(new n_add());
			newind.line.at(loc) = node('+');
			break;
		case 3: // -
			//newind.line.at(loc)=shared_ptr<node>(new n_sub());
			newind.line.at(loc) = node('-');
			break;
		case 4: // *
			//newind.line.at(loc)=shared_ptr<node>(new n_mul());
			newind.line.at(loc) = node('*');
			break;
		case 5: // /
			//newind.line.at(loc)=shared_ptr<node>(new n_div());
			newind.line.at(loc) = node('/');
			break;
		case 6: // sin
			//newind.line.at(loc)=shared_ptr<node>(new n_sin());
			newind.line.at(loc) = node('s');
			break;
		case 7: // cos
			//newind.line.at(loc)=shared_ptr<node>(new n_cos());
			newind.line.at(loc) = node('c');
			break;
		case 8: // exp
			//newind.line.at(loc)=shared_ptr<node>(new n_exp());
			newind.line.at(loc) = node('e');
			break;
		case 9: // log
			//newind.line.at(loc)=shared_ptr<node>(new n_log());
			newind.line.at(loc) = node('l');
			break;
		//case 10: // seed
		//	seedchoice = r[omp_get_thread_num()].rnd_int(0,p.seedstacks.size()-1);
		//	copystack(p.seedstacks.at(seedchoice),tmpstack);

		//	for(int i=0;i<tmpstack.size(); ++i)
		//	{
		//		if (x<p.max_len){
		//			newind.line.push_back(tmpstack[i]);
		//			x++;
		//		}
		//	}
		//	tmpstack.clear();
		//	break;
		case 11: // sqrt
			newind.line.at(loc) = node('q');
			break;
    case 12: // square
      newind.line.at(loc) = node('2');
      break;
    case 13: // cube
      newind.line.at(loc) = node('3');
      break;
		case 14: // exponent
			newind.line.at(loc) = node('^');
			break;
		case 15: // equals
			newind.line.at(loc) = node('=');
			break;
		case 16: // does not equal
			newind.line.at(loc) = node('!');
			break;
		case 17: // less than
			newind.line.at(loc) = node('<');
			break;
		case 18: // greater than
			newind.line.at(loc) = node('>');
			break;
		case 19: //less than or equal to
			newind.line.at(loc) = node('{');
			break;
		case 20: //greater than or equal to
			newind.line.at(loc) = node('}');
			break;
		case 21: //if-then
			newind.line.at(loc) = node('i');
			break;
		case 22: //if-then-else
			newind.line.at(loc) = node('t');
			break;
		case 23: //and
			newind.line.at(loc) = node('&');
			break;
		case 24: //or
			newind.line.at(loc) = node('|');
			break;
		}


}
void InsInstruction(ind& newind,int loc,params& p,vector<Randclass>& r)
{

	vector <string> load_choices(p.allblocks);
	int choice=0;
   std::vector<node>::iterator it;
	it = newind.line.begin();
	//cout<<"iterator definition\n";
	//std::vector<int>::iterator it;
	//it = newind.line.begin();
	//cout<<"end def \n";

	//uniform_int_distribution<int> dist(0,21);
	if (p.ERC)
	{
		float ERC;
		for (int j=0;j<p.numERC;++j)
		{
			ERC = r[omp_get_thread_num()].rnd_flt((float)p.minERC,(float)p.maxERC);
			if(p.ERCints)
				load_choices.push_back(to_string(static_cast<long long>(ERC)));
			else
				load_choices.push_back(to_string(static_cast<long double>(ERC)));
		}
	}
	vector<float> wheel(p.op_weight.size());
	//wheel.resize(p.op_weight.size());
	if (p.weight_ops_on) //fns are weighted
	{
		partial_sum(p.op_weight.begin(), p.op_weight.end(), wheel.begin());
	}

	if (p.weight_ops_on) //fns are weighted
	{
		float tmp = r[omp_get_thread_num()].rnd_flt(0,1);
		if (tmp < wheel.at(0))
			choice=0;
		else
		{
			for (unsigned int k=1;k<wheel.size();++k)
			{
				if(tmp<wheel.at(k) && tmp>=wheel.at(k-1))
					choice = k;
			}
		}
	}
	else
		choice = r[omp_get_thread_num()].rnd_int(0,p.op_list.size()-1);

		string varchoice;
//		int seedchoice;
		vector<node> tmpstack;
		switch (p.op_choice.at(choice))
		{
		case 0: //load number
			/*if(p.ERCints)
				newind.line.at(loc)=shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
			else
				newind.line.at(loc)=shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
			break;*/
			if(p.ERC){ // if ephemeral random constants are on
				if (!p.cvals.empty()){
					if (r[omp_get_thread_num()].rnd_flt(0,1)<.5)
						//newind.line.insert(it+loc,shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
					newind.line.insert(it+loc,node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
					else{
						if(p.ERCints)
							//newind.line.insert(it+loc,shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));
							newind.line.insert(it+loc,node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
						else
							//newind.line.insert(it+loc,shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
						newind.line.insert(it+loc,node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
					}
				}
				else{
					if(p.ERCints)
							//newind.line.insert(it+loc,shared_ptr<node>(new n_num((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC))));
							newind.line.insert(it+loc,node((float)r[omp_get_thread_num()].rnd_int(p.minERC,p.maxERC)));
						else
							//newind.line.insert(it+loc,shared_ptr<node>(new n_num(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC))));
							newind.line.insert(it+loc,node(r[omp_get_thread_num()].rnd_flt(p.minERC,p.maxERC)));
				}
			}
			else if (!p.cvals.empty())
			{
				//newind.line.insert(it+loc,shared_ptr<node>(new n_num(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1)))));
				newind.line.insert(it+loc,node(p.cvals.at(r[omp_get_thread_num()].rnd_int(0,p.cvals.size()-1))));
			}
			break;
		case 1: //load variable
			varchoice = p.allvars.at(r[omp_get_thread_num()].rnd_int(0,p.allvars.size()-1));
			//newind.line.insert(it+loc,shared_ptr<node>(new n_sym(varchoice)));
			newind.line.insert(it+loc,node(varchoice));
			break;
		case 2: // +
			//newind.line.insert(it+loc,shared_ptr<node>(new n_add()));
			newind.line.insert(it+loc,node('+'));
			break;
		case 3: // -
			//newind.line.insert(it+loc,shared_ptr<node>(new n_sub()));
			newind.line.insert(it+loc,node('-'));
			break;
		case 4: // *
			//newind.line.insert(it+loc,shared_ptr<node>(new n_mul()));
			newind.line.insert(it+loc,node('*'));
			break;
		case 5: // /
			//newind.line.insert(it+loc,shared_ptr<node>(new n_div()));
			newind.line.insert(it+loc,node('/'));
			break;
		case 6: // sin
			//newind.line.insert(it+loc,shared_ptr<node>(new n_sin()));
			newind.line.insert(it+loc,node('s'));
			break;
		case 7: // cos
			//newind.line.insert(it+loc,shared_ptr<node>(new n_cos()));
			newind.line.insert(it+loc,node('c'));
			break;
		case 8: // exp
			//newind.line.insert(it+loc,shared_ptr<node>(new n_exp()));
			newind.line.insert(it+loc,node('e'));
			break;
		case 9: // log
			//newind.line.insert(it+loc,shared_ptr<node>(new n_log()));
			newind.line.insert(it+loc,node('l'));
			break;
		//case 10: // seed
		//	seedchoice = r[omp_get_thread_num()].rnd_int(0,p.seedstacks.size()-1);
		//	copystack(p.seedstacks.at(seedchoice),tmpstack);

		//	for(int i=0;i<tmpstack.size(); ++i)
		//	{
		//		if (x<p.max_len){
		//			newind.line.push_back(tmpstack[i]);
		//			x++;
		//		}
		//	}
		//	tmpstack.clear();
		//	break;
		case 11: // sqrt
			newind.line.insert(it+loc,node('q'));
			break;
    case 12: // square
      newind.line.insert(it+loc,node('2'));
      break;
    case 13: // cube
      newind.line.insert(it+loc,node('3'));
      break;
    case 14: // exponent
			newind.line.insert(it+loc,node('^'));
			break;
		case 15: // equals
			newind.line.insert(it+loc,node('='));
			break;
		case 16: // does not equal
			newind.line.insert(it+loc,node('!'));
			break;
		case 17: // less than
			newind.line.insert(it+loc,node('<'));
			break;
		case 18: // greater than
			newind.line.insert(it+loc,node('>'));
			break;
		case 19: //less than or equal to
			newind.line.insert(it+loc,node('{'));
			break;
		case 20: //greater than or equal to
			newind.line.insert(it+loc,node('}'));
			break;
		case 21: //if-then
			newind.line.insert(it+loc,node('i'));
			break;
		case 22: //if-then-else
			newind.line.insert(it+loc,node('t'));
			break;
		case 23: //and
			newind.line.insert(it+loc,node('&'));
			break;
		case 24: //or
			newind.line.insert(it+loc,node('|'));
			break;
		}


}




//void makenew(ind& newind)
//{
//	for (int i=0;i<newind.line.size();++i)
//	{
//		/*if (newind.line.at(i).use_count()>1)
//		{*/
//			string varname;
//			float value;
//			bool onval;
//			switch (newind.line.at(i)->type){
//			case 'n':
//				value = static_pointer_cast<n_num>(newind.line.at(i))->value;
//				onval = newind.line.at(i)->on;
//				newind.line.at(i) = shared_ptr<node>(new n_num(value));
//				newind.line.at(i)->on=onval;
//				break;
//			case 'v':
//				varname = static_pointer_cast<n_sym>(newind.line.at(i))->varname;
//				onval = newind.line.at(i)->on;
//				newind.line.at(i) = shared_ptr<node>(new n_sym(varname));
//				newind.line.at(i)->on=onval;
//				break;
//			case '+': // +
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_add());
//				newind.line.at(i)->on=onval;
//				break;
//			case '-': // -
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_sub());
//				newind.line.at(i)->on=onval;
//				break;
//			case '*': // *
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_mul());
//				newind.line.at(i)->on=onval;
//				break;
//			case '/': // /
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_div());
//				newind.line.at(i)->on=onval;
//				break;
//			case 's': // sin
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_sin());
//				newind.line.at(i)->on=onval;
//				break;
//			case 'c': // cos
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_cos());
//				newind.line.at(i)->on=onval;
//				break;
//			case 'e': // exp
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_exp());
//				newind.line.at(i)->on=onval;
//				break;
//			case 'l': // log
//				onval = newind.line.at(i)->on;
//				newind.line.at(i)=shared_ptr<node>(new n_log());
//				newind.line.at(i)->on=onval;
//				break;
//				}
//		//}
//		if (newind.line.at(i).use_count()==0)
//		{
//			cerr << "shared pointer use count is zero\n";
//		}
//}
//}
//void makenewcopy(ind& oldind, ind& newind)
//{
//	for (int i=0;i<oldind.line.size();++i)
//	{
//		/*if (newind.line.at(i).use_count()>1)
//		{*/
//			string varname;
//			float value;
//			bool onval;
//			switch (oldind.line.at(i)->type){
//			case 'n':
//				value = static_pointer_cast<n_num>(oldind.line.at(i))->value;
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back( shared_ptr<node>(new n_num(value)));
//				newind.line.at(i)->on=onval;
//				break;
//			case 'v':
//				varname = static_pointer_cast<n_sym>(oldind.line.at(i))->varname;
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back( shared_ptr<node>(new n_sym(varname)));
//				newind.line.at(i)->on=onval;
//				break;
//			case '+': // +
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_add()));
//				newind.line.at(i)->on=onval;
//				break;
//			case '-': // -
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_sub()));
//				newind.line.at(i)->on=onval;
//				break;
//			case '*': // *
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_mul()));
//				newind.line.at(i)->on=onval;
//				break;
//			case '/': // /
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_div()));
//				newind.line.at(i)->on=onval;
//				break;
//			case 's': // sin
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_sin()));
//				newind.line.at(i)->on=onval;
//				break;
//			case 'c': // cos
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_cos()));
//				newind.line.at(i)->on=onval;
//				break;
//			case 'e': // exp
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_exp()));
//				newind.line.at(i)->on=onval;
//				break;
//			case 'l': // log
//				onval = oldind.line.at(i)->on;
//				newind.line.push_back(shared_ptr<node>(new n_log()));
//				newind.line.at(i)->on=onval;
//				break;
//				}
//		//}
//		if (newind.line.at(i).use_count()==0)
//		{
//			cerr << "shared pointer use count is zero\n";
//		}
//}
//}
//void copystack(vector<shared_ptr<node>>& line, vector<shared_ptr<node>>& newline)
//{
//	for (int i=0;i<line.size();++i)
//	{
//		string varname;
//		float value;
//		bool onval;
//		switch (line.at(i)->type){
//		case 'n':
//			value = static_pointer_cast<n_num>(line.at(i))->value;
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_num(value)));
//			newline.at(i)->on=onval;
//			break;
//		case 'v':
//			varname = static_pointer_cast<n_sym>(line.at(i))->varname;
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_sym(varname)));
//			newline.at(i)->on=onval;
//			break;
//		case '+': // +
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_add()));
//			newline.at(i)->on=onval;
//			break;
//		case '-': // -
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_sub()));
//			newline.at(i)->on=onval;
//			break;
//		case '*': // *
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_mul()));
//			newline.at(i)->on=onval;
//			break;
//		case '/': // /
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_div()));
//			newline.at(i)->on=onval;
//			break;
//		case 's': // sin
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_sin()));
//			newline.at(i)->on=onval;
//			break;
//		case 'c': // cos
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_cos()));
//			newline.at(i)->on=onval;
//			break;
//		case 'e': // exp
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_exp()));
//			newline.at(i)->on=onval;
//			break;
//		case 'l': // log
//			onval = line.at(i)->on;
//			newline.push_back(shared_ptr<node>(new n_log()));
//			newline.at(i)->on=onval;
//			break;
//		}
//	}
//}
