#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include "general_fns.h"
#include <boost/uuid/uuid_io.hpp>
void Crossover(ind& p1,ind& p2,vector<ind>& tmppop,params& p,vector<Randclass>& r)
{
	vector<ind> parents; parents.reserve(2);
	parents.push_back(p1);
	parents.push_back(p2);

	//makenew(parents[0]);
	//makenew(parents[1]);

	vector<ind> kids(2);

	int r2,off1,off2,offset,head;
	//std::vector<int>::iterator it;
	int tmpinssize = 0;
	vector<int> psize;
	psize.push_back(p1.line.size());
	psize.push_back(p2.line.size());

	if(p.cross==1) //alternation
	{
		for (int r1=0; r1 < 2; r1++)
		{
			if(r1==0)
				r2=1;
			else
				r2=0;

			//if(parents[r1].line.size()>parents[r2].line.size())
			//{
			//	off1 = r[omp_get_thread_num()].rnd_int(0,parents[r1].line.size()-parents[r2].line.size()-1);
			//	off2 = 0;
			//}
			//else if (parents[r2].line.size()>parents[r1].line.size())
			//{
			//	off1 = 0;
			//	off2 = r[omp_get_thread_num()].rnd_int(0,parents[r2].line.size()-parents[r1].line.size()-1);
			//}
			//else
			//{
			//	off1=0;
			//	off2=0;
			//}
			//head = r1;
			//offset=off1;
			//// assign beginning of parent to kid if it is longer
			//kids.at(r1).line.insert(kids.at(r1).line.end(),parents[r1].line.begin(),parents[r1].line.begin()+offset);

			//for (unsigned int i=0;i<std::min(parents[r1].line.size(),parents[r2].line.size());++i)
			//{
			//	if (r[omp_get_thread_num()].rnd_flt(0,1)<p.cross_ar)
			//	{
			//		if(head==r1)
			//		{
			//			head=r2;
			//			offset=off2;
			//		}
			//		else
			//		{
			//			head=r1;
			//			offset=off1;
			//		}
			//	}
			//
			//	kids.at(r1).line.push_back(parents[head].line.at(i+offset));
			//}
			//if(kids.at(r1).line.size() < parents[r1].line.size())
			//{
			//	int gap = kids.at(r1).line.size() < parents[r1].line.size()+1;
			//	kids.at(r1).line.insert(kids.at(r1).line.end(),parents[r1].line.end()-gap,parents[r1].line.end());
			//}
				//kids.at(r1).line.push_back(parents[r1].line.at(kids.at(r1).line.size()));

			// new version uses alignment deviation rather than an initial random offset.
			head = r1;
			offset = 0;
			for (unsigned int i=0;i<parents[r1].line.size(); ++i)
			{
				if (r[omp_get_thread_num()].rnd_flt(0,1)<p.cross_ar)
				{
					if(head==r1)
					{
						head=r2;

					}
					else
					{
						head=r1;
						//offset=r[omp_get_thread_num()].gasdev()*parents[head].line.size()*p.cross_ar;
					}
					if (p.align_dev)
						offset=r[omp_get_thread_num()].gasdev();//*parents[head].line.size()*p.cross_ar;
				}


				if (i+offset>=parents[head].line.size() || i+offset <= 0)
					offset = 0;

				if (i < parents[head].line.size() && kids.at(r1).line.size() < p.max_len)
					kids.at(r1).line.push_back(parents[head].line.at(i+offset));


			}
			/*if(kids.at(r1).line.size() < parents[r1].line.size())
			{
				int gap = parents[r1].line.size()-kids.at(r1).line.size() +1;
				kids.at(r1).line.insert(kids.at(r1).line.end(),parents[r1].line.end()-gap,parents[r1].line.end());
			}*/
		}
	}
	else if (p.cross==2) // one-point crossover
	{
		if (p.align_dev){
			bool tryagain = true;
			int max_tries = 5;
			int tries = 0;
			while (tryagain && tries < max_tries) {
				int point1 = r[omp_get_thread_num()].rnd_int(0,min(p1.line.size(),p2.line.size()));
				int point2 = point1 + r[omp_get_thread_num()].gasdev(); //*abs(int(p1.line.size()-p2.line.size()))/10;
				int tmp1 = point1;
				int tmp2 = point2;

				point1 += r[omp_get_thread_num()].gasdev(); //*abs(int(p1.line.size()-p2.line.size()))/10;
				int tmp1_1 = point1;

				if (point1 < 0)
					point1 = 0;
				else if (point1 > p1.line.size()) {
					point1 = p1.line.size();
				}


				if (point2 < 0)
					point2 = 0;
				else if (point2 > p2.line.size()) {
					int p2size = p2.line.size();
					point2 = p2.line.size();
				}


				kids[0].line.assign(parents[0].line.begin(),parents[0].line.begin()+point1);
				kids[0].line.insert(kids[0].line.end(),parents[1].line.begin()+point2,parents[1].line.end());

				kids[1].line.assign(parents[1].line.begin(),parents[1].line.begin()+point2);
				kids[1].line.insert(kids[1].line.end(),parents[0].line.begin()+point1,parents[0].line.end());

				if (kids[0].line.empty() || kids[1].line.empty() || kids[0].line.size()>p.max_len || kids[1].line.size()>p.max_len)
					tryagain = true;
				else
					tryagain = false;

				++tries;

			}
			if (tryagain)
			{
				int point1 = min(p1.line.size(),p2.line.size())/2;
				int point2 = point1;

				kids[0].line.assign(parents[0].line.begin(),parents[0].line.begin()+point1);
				kids[0].line.insert(kids[0].line.end(),parents[1].line.begin()+point2,parents[1].line.end());

				kids[1].line.assign(parents[1].line.begin(),parents[1].line.begin()+point2);
				kids[1].line.insert(kids[1].line.end(),parents[0].line.begin()+point1,parents[0].line.end());
			}
		}
		else{
			int point1 = r[omp_get_thread_num()].rnd_int(0,min(p1.line.size(),p2.line.size()));

			kids[0].line.assign(parents[0].line.begin(),parents[0].line.begin()+point1);
			kids[0].line.insert(kids[0].line.end(),parents[1].line.begin()+point1,parents[1].line.end());

			kids[1].line.assign(parents[1].line.begin(),parents[1].line.begin()+point1);
			kids[1].line.insert(kids[1].line.end(),parents[0].line.begin()+point1,parents[0].line.end());
		}

		//int empty_count=0;
		//while(kids[0].line.empty() || kids[1].line.empty() || kids[0].line.size()>p.max_len || kids[1].line.size()>p.max_len && empty_count<10){
		//	int point1 = r[omp_get_thread_num()].rnd_int(0,p1.line.size());
		//	int point2 = r[omp_get_thread_num()].rnd_int(0,p2.line.size());

		//	kids[0].line.assign(parents[0].line.begin(),parents[0].line.begin()+point1);
		//	kids[0].line.insert(kids[0].line.end(),parents[1].line.begin()+point2,parents[1].line.end());

		//	kids[1].line.assign(parents[1].line.begin(),parents[1].line.begin()+point2);
		//	kids[1].line.insert(kids[1].line.end(),parents[0].line.begin()+point1,parents[0].line.end());
		//
		//	++empty_count;
		//}
		//if (empty_count==10) // split parents half and half
		//{
		//	int point1 = (p1.line.size()-1)/2;
		//	int point2 = (p2.line.size()-1)/2;

		//	kids[0].line.assign(parents[0].line.begin(),parents[0].line.begin()+point1);
		//	kids[0].line.insert(kids[0].line.end(),parents[1].line.begin()+point2,parents[1].line.end());

		//	kids[1].line.assign(parents[1].line.begin(),parents[1].line.begin()+point2);
		//	kids[1].line.insert(kids[1].line.end(),parents[0].line.begin()+point1,parents[0].line.end());
		//}

	}
	else if (p.cross == 3) // sub-tree-like crossover
	{

		for (int r1=0; r1 < 2; ++r1)
		{
			if(r1==0)
				r2=1;
			else
				r2=0;

			int pt1, pt2;

			// specialization for m3gp
			if (p.classification && p.class_m4gp && r[omp_get_thread_num()].rnd_flt(0.0,1.0) > 0.5){
				// find root nodes of r1
				vector<unsigned> roots_r1;
				find_root_nodes(parents[r1].line, roots_r1);
				pt1 = roots_r1[r[omp_get_thread_num()].rnd_int(0,roots_r1.size()-1)];

				vector<unsigned> roots_r2;
				find_root_nodes(parents[r2].line, roots_r2);
				pt2 = roots_r2[r[omp_get_thread_num()].rnd_int(0,roots_r2.size()-1)];
			}
			else{
				pt1 = r[omp_get_thread_num()].rnd_int(0,parents[r1].line.size()-1);
				pt2 = r[omp_get_thread_num()].rnd_int(0,parents[r2].line.size()-1);
			}

			int end1 = pt1;
			int sum_arity = parents[r1].line[pt1].arity_float;
			while (sum_arity > 0 && pt1 > 0)
			{
				--pt1;
				--sum_arity;
				sum_arity+=parents[r1].line[pt1].arity_float;

			}
			int begin1 = pt1;

			int end2 = pt2;
			sum_arity = parents[r2].line[pt2].arity_float;
			while (sum_arity > 0 && pt2 > 0)
			{
				--pt2;
				--sum_arity;
				sum_arity+=parents[r2].line[pt2].arity_float;
			}
			int begin2 = pt2;
			ind st1, st2;
			st1.line.assign(parents[r1].line.begin()+begin1,parents[r1].line.begin()+end1+1);
			st2.line.assign(parents[r2].line.begin()+begin2,parents[r2].line.begin()+end2+1);

			kids[r1].line.assign(parents[r1].line.begin(),parents[r1].line.begin()+begin1);
			kids[r1].line.insert(kids[r1].line.end(),parents[r2].line.begin()+begin2,parents[r2].line.begin()+end2+1);
			kids[r1].line.insert(kids[r1].line.end(),parents[r1].line.begin()+end1+1,parents[r1].line.end());

			if (kids[r1].line.size()>p.max_len)
				kids[r1].line = parents[r1].line;
		}


	}
	//tmpinssize=0;
	//for(unsigned int t=0; t<kids[0].line.size();t++)
	//	if(kids[0].line.at(t)>99) tmpinssize++;
	//if(tmpinssize!=kids[0].args.size())
	//	cout << "size mismatch" << endl;
	assert(~kids[0].line.empty() && ~kids[1].line.empty());

	/*assert(kids[0].line.size() >= p.min_len);
	assert(kids[1].line.size() >= p.min_len);*/

	tmpinssize=0;

	kids[0].origin = 'c';
	kids[0].parentfitness = parents[0].fitness;
	kids[1].origin = 'c';
	kids[1].parentfitness = parents[1].fitness;

	// database tracking
	kids[0].parent_id.resize(0);
	kids[0].parent_id.push_back(parents[0].tag);
	kids[0].parent_id.push_back(parents[1].tag);
	kids[1].parent_id.resize(0);
	kids[1].parent_id.push_back(parents[1].tag);
	kids[1].parent_id.push_back(parents[0].tag);
	//makenew(kids[0]);
	//makenew(kids[1]);

	tmppop.push_back(kids[0]);
	tmppop.push_back(kids[1]);

	//kids.clear();
	//parents.clear();
}
