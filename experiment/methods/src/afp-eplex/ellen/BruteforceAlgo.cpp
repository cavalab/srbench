#include "stdafx.h"
#include <vector>
#include "Datapoint.h"
#include "BruteforceAlgo.h"

int BruteforceAlgo::computeFrontier(std::vector<Datapoint*>& dataset){
  int numPareto=0;

  for (size_t n=0; n<dataset.size(); ++n){
    for (size_t m=0; m<dataset.size(); ++m){
      if (*dataset[n] < *dataset[m]){
	dataset[n]->incrementDominated();
      }
    }
  }

  for (size_t n=0; n<dataset.size(); ++n){
    if (dataset[n]->numDominated() == 0){
      dataset[n]->setParetoStatus(1);
      numPareto++;
    }
    else {
      dataset[n]->setParetoStatus(0);
    }
  }

  return numPareto;
}
