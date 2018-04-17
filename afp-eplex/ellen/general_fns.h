#pragma once
#ifndef GENERAL_FNS_H
#define GENERAL_FNS_H
//#include "rnd.h"


bool is_number(const std::string& s);
void MutInstruction(ind& newind,int loc,params&,vector<Randclass>& r,Data& d);
void InsInstruction(ind& newind,int loc,params&,vector<Randclass>& r);
void find_root_nodes(vector<node>& line, vector<unsigned>& roots);
float Round(float d);

//void makenew(ind& newind);
//void makenewcopy(ind& oldind, ind& newind);
//void copystack(vector<shared_ptr<node> >& line, vector<shared_ptr<node> >& newline);


#endif
