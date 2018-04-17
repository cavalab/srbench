#include "stdafx.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include "Datapoint.h"
#include "StablesortAlgo.h"

int StablesortAlgo::computeFrontier(std::vector<Datapoint*>& dataset){

  if (dataset[0]->dim() > 2){
    std::cerr << "[error] Stablesort Algorithm only works with K=2" << std::endl;
    exit(1);
  }



  // d is a copy of the dataset (we'll sort it but keep original dataset in original order)
  std::vector<Datapoint*> d = dataset; 

  sort(d.begin(),d.end(),SortFirstDim());
  stable_sort(d.begin(),d.end(),SortSecondDim());

  int numPareto=1;
  d[0]->setParetoStatus(1);
  Datapoint* currentPareto = d[0];

  for (size_t n=1; n<d.size();++n){
    if (*d[n] < *currentPareto){
      d[n]->setParetoStatus(0); // d[n] is dominated
    }
    else {
      d[n]->setParetoStatus(1); // d[n] is not dominated, so it'll be pareto
      numPareto++;
      currentPareto = d[n];
    }
  }
  return numPareto;
}
