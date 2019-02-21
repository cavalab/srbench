#include "stdafx.h"
#include "pop.h"

void CrossoverP(ind& p1,ind& p2,ind& tmpind1,ind& tmpind2,params& p,vector<Randclass>& r)
{ 
	//cout<<"in crossoverP\n";// produces only 1 child.
	vector<ind> parents;
	parents.push_back(p1);
	parents.push_back(p2);

	vector<ind> kids(2);

	int r2,off1,off2,offset,head,it;
		int tmpinssize = 0;
	//if(p.cross==1)
	//{
		for (int r1=0; r1 < 2; r1++)
		{
			if(r1==0)
				r2=1;
			else
				r2=0;
		
			if(parents[r1].line.size()>parents[r2].line.size())
			{
				off1 = r[omp_get_thread_num()].rnd_int(0,parents[r1].line.size()-parents[r2].line.size()-1);
				off2 = 0;
			}
			else if (parents[r2].line.size()>parents[r1].line.size())
			{
				off1 = 0;
				off2 = r[omp_get_thread_num()].rnd_int(0,parents[r2].line.size()-parents[r1].line.size()-1);
			}
			else
			{
				off1=0; 
				off2=0;
			}
			head = r1;
			offset=off1;
			it = 0;
			// assign beginning of parent to kid if it is longer
			while(kids.at(r1).line.size() < offset)
			{
				if (parents[r1].line.at(it)>=100)
				{
					kids.at(r1).args.push_back(parents[r1].args.at(parents[r1].line.at(it)-100));
					kids.at(r1).line.push_back(kids.at(r1).args.size()+99);
				}
				else
					kids.at(r1).line.push_back(parents[r1].line.at(it));
				++it;
			}


			for (unsigned int i=0;i<std::min(parents[r1].line.size(),parents[r2].line.size());i++)
			{
				if (r[omp_get_thread_num()].rnd_flt(0,1)<p.cross_ar)
				{
					if(head==r1)
					{
						head=r2;
						offset=off2;
					}
					else
					{
						head=r1;
						offset=off1;
					}
				}

			
				if(parents[head].line.at(i+offset)>99)
				{// grab arguments
					kids.at(r1).args.push_back(parents[head].args.at(parents[head].line.at(i+offset)-100));
					kids.at(r1).line.push_back(kids.at(r1).args.size()+99);
				}
				else
					kids.at(r1).line.push_back(parents[head].line.at(i+offset));
			
				tmpinssize=0;
				for(unsigned int t=0; t<kids[r1].line.size();t++)
					if(kids[r1].line.at(t)>99) tmpinssize++;
				if(tmpinssize!=kids[r1].args.size())
					cout << "size mismatch" << endl;
			}
		
			
			while(kids.at(r1).line.size() < parents[r1].line.size())
			{
				if (parents[r1].line.at(kids.at(r1).line.size())>=100)
				{
					kids.at(r1).args.push_back(parents[r1].args.at(parents[r1].line.at(kids.at(r1).line.size())-100));
					kids.at(r1).line.push_back(kids.at(r1).args.size()+99);
				}
				else
					kids.at(r1).line.push_back(parents[r1].line.at(kids.at(r1).line.size()));
			}
		}
	//}
	//tmpinssize=0;
	//for(unsigned int t=0; t<kids[0].line.size();t++)
	//	if(kids[0].line.at(t)>99) tmpinssize++;
	//if(tmpinssize!=kids[0].args.size())
	//	cout << "size mismatch" << endl;

	kids.at(0).origin = 'c';
	kids.at(1).origin = 'c';
	tmpind1 = kids.at(0);
	tmpind2 = kids.at(1);

}