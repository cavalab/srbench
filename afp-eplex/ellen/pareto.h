//#include "pop.h"
#pragma once
#ifndef PARETO_H
#define PARETO_H

//#include <vector>
//#include "pop.h"
using namespace std;
void pareto_fc(vector<ind>& pop);
void pareto_fa(vector<ind>& pop);
void pareto_fga(vector<ind>& pop);
void pareto(vector<ind>& pop,std::vector<float>& data);
void pareto2(std::vector<float>& data, int& r1, int& r2);
#endif