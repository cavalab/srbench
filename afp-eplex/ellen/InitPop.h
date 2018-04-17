#pragma once
#ifndef INITPOP_H
#define INITPOP_H
//#include "pop.h"

void InitPop(vector<ind>&, params&, vector<Randclass>&);
void makeline(ind&,params& p,vector<Randclass>& r);
void makeline_rec(vector<node>&,params& p,vector<Randclass>& r,int linelen);
int maketree(vector <node>& line, int level, bool exactlevel, int lastnode,char type,params& p,vector<Randclass>& r);

#endif