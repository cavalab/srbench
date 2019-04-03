#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include "general_fns.h"
#include "InitPop.h"
#include "general_fns.h"

void Mutate(ind& par, vector<ind>& tmppop, params& p, vector<Randclass>& r, Data& d)
{
	vector<unsigned int> ichange;
	vector<ind> kid(1,par);
	//makenew(kid[0]);
	kid[0].origin='m';
	kid[0].parentfitness=par.fitness;
	kid[0].parent_id.resize(0);
	kid[0].parent_id.push_back(par.tag);
	kid[0].clrPhen();

	if (p.mutate==1){ // point mutation
		for(unsigned int i = 0;i<kid[0].line.size();++i)
		{
			if(r[omp_get_thread_num()].rnd_flt(0,1)<=p.mut_ar)
				ichange.push_back(i);
		}
		for(unsigned int j=0;j<ichange.size();++j)
		{
			if(kid[0].line.at(ichange.at(j)).type=='n') // perturb constant
			{
				if (p.ERC)
				{
						float num = kid[0].line.at(ichange.at(j)).value;
						num = num/2 + r[omp_get_thread_num()].gasdev()*num/2;
						kid[0].line.at(ichange.at(j)).value = num;
				}
			}
			else if (kid[0].line.at(ichange.at(j)).type=='v'){
					//else if variable, pick random variable replacement
				kid[0].line.at(ichange.at(j)).varname = d.label.at(r[omp_get_thread_num()].rnd_int(0,d.label.size()-1));
			}
			else
			{
				bool onstate = kid[0].line.at(ichange.at(j)).on;
				MutInstruction(kid[0],ichange.at(j),p,r,d);
				//inherit epigenetic state of gene location
				kid[0].line.at(ichange.at(j)).on=onstate;
			}
		}
	}
	else if (p.mutate==2){ //subtree mutation
		int pt1,begin1,end1;
		int action; // 1: delete tree; 2: swap tree; 3: add tree (m3gp only)
		// specialization for m3gp
			if (p.classification && p.class_m4gp){

				if ( r[omp_get_thread_num()].rnd_flt(0.0,1.0) <= 0.333 && kid[0].line.size()<p.max_len){
					action=3;
					begin1 = 1;
					end1 = 0;
				}
				else{
					// find root nodes of kid[0]
					vector<unsigned> roots;
					find_root_nodes(kid[0].line, roots);
					if (r[omp_get_thread_num()].rnd_flt(0.0,1.0) <= 0.5 && roots.size()>1){
						action = 1;
						pt1 = roots[r[omp_get_thread_num()].rnd_int(0,roots.size()-1)];
					}
					else { //swap
						action = 2;
						pt1 = r[omp_get_thread_num()].rnd_int(0,kid[0].line.size()-1);
					}
				}

			}
			//else if (p.classification && p.class_m4gp) //just add tree to end
			//	action = 3;
			else{
				// choose action
				action = 2;
				//if(kid[0].line.size()<p.min_len) // if kid is small, make a swap
				//	action = 2;
				//else
				//	action = r[omp_get_thread_num()].rnd_int(1,2);
				// choose point of mutation
				if (action==1){ // if deletion, don't choose the head
					int tmp = 2;
					while (!kid[0].line[kid[0].line.size()-tmp].on)
						--tmp;
					pt1 = r[omp_get_thread_num()].rnd_int(0,kid[0].line.size()-tmp);
				}
				else if (action==2)
					pt1 = r[omp_get_thread_num()].rnd_int(0,kid[0].line.size()-1);
			}

			if (action!=3){
				end1 = pt1;
				int sum_arity = kid[0].line[pt1].arity_float;
				while (sum_arity > 0 && pt1 > 0)
				{
					--pt1;
					--sum_arity;
					sum_arity+=kid[0].line[pt1].arity_float;
				}
				begin1 = pt1;
			}


			if(action==1 && kid[0].line.size()-(end1-(begin1-1)) >= p.min_len)// delete subtree
				kid[0].line.erase(kid[0].line.begin()+begin1,kid[0].line.begin()+end1+1);
			else{ // create new subtree
				vector<node> st;
				int linelen;
				if (action==3){ // add a tree that does not violate max program size
					linelen = r[omp_get_thread_num()].rnd_int(1,p.max_len-kid[0].line.size());
				}
				else{ // swap tree with one within a normal distribution of the current tree size with std dev = 1/2 tree size
					linelen = std::max(p.min_len - (int(kid[0].line.size()) - (end1-(begin1-1))), end1-(begin1-1) + int(Round(r[omp_get_thread_num()].gasdev() * 0.5 * (end1-(begin1-1)))));
					// new tree should at least result in the size of the final tree being the minimum length
					//linelen = std::max(linelen, );
				}
					//linelen = r[omp_get_thread_num()].rnd_int(p.min_len,p.max_len-kid[0].line.size()+end1-(begin1-1));

				//int tmp = p.min_len - (int(kid[0].line.size()) - (end1-(begin1-1)));

				makeline_rec(st,p,r,linelen);
				// if the subtree is too small, make it bigger
				while (st.size()+(int(kid[0].line.size()) - (end1-(begin1-1))) < p.min_len){
					++linelen;
					st.clear();
					makeline_rec(st,p,r,linelen);
				}

				if (action==2){ // swap new subtree
					kid[0].line.erase(kid[0].line.begin()+begin1,kid[0].line.begin()+end1+1);
					kid[0].line.insert(kid[0].line.begin()+begin1,st.begin(),st.end());
				}
			    else if (action==3) // add tree to end (m3gp only)
					kid[0].line.insert(kid[0].line.end(),st.begin(),st.end());

				//assert(kid[0].line.size() >= p.min_len);
			}


	}

	tmppop.push_back(kid[0]);
	kid.clear();

}
