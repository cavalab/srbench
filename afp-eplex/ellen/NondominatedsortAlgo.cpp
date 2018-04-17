#include "stdafx.h"
#include <vector>
#include "Datapoint.h"
#include "NondominatedsortAlgo.h"

int NondominatedsortAlgo::computeFrontier(std::vector<Datapoint*>& dataset){
  int numPareto=0;

  // Pairwise comparisons
  for (size_t n=0; n<dataset.size(); ++n){
    for (size_t m=0; m<dataset.size(); ++m){
      if (*dataset[n] < *dataset[m]){
	dataset[n]->incrementDominated();
	dataset[m]->addToDominatingSet(n);
      }
    }
  }

  // Find the first Pareto front
  std::vector<size_t> front;
  std::vector<size_t> front2;
  int tmpLevel = -10;   // temporary value for Pareto level; will re-adjust after total number of levels is known
  for (size_t n=0; n<dataset.size(); ++n){
    if (dataset[n]->numDominated() == 0){
      dataset[n]->setParetoStatus(tmpLevel);
      front.push_back(n);
      numPareto++;
    }
  }

  // Iteratively peel off Pareto fronts
  while (!front.empty()){
    tmpLevel--;
    for (size_t i=0; i<front.size(); ++i){

      std::vector<size_t>::const_iterator e=dataset[front[i]]->endDominatingSet();
      for (std::vector<size_t>::const_iterator s = dataset[front[i]]->beginDominatingSet(); s!=e;++s){
	dataset[*s]->decrementDominated();
	if (dataset[*s]->numDominated() == 0){
	  front2.push_back(*s);
	  dataset[*s]->setParetoStatus(tmpLevel);
	}
      }
    }
    front = front2;
    front2.clear();
  }

  // Re-adjust pareto-level so that we have positive integers and lowest level=0
  for (size_t n=0; n<dataset.size(); ++n){
    int oldLevel = dataset[n]->getParetoStatus();
    if (oldLevel != -1){
      dataset[n]->setParetoStatus(oldLevel-tmpLevel-1);      
    }

  }

  return numPareto;
}
