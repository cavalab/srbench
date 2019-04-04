#ifndef PARETO_ALGO_H_
#define PARETO_ALGO_H_

//////////////////////////////
// ParetoAlgo: Abstract base class for pareto frontier algorithms

#include <vector>
#include "Datapoint.h"

class ParetoAlgo {
 public: 
  virtual int computeFrontier(std::vector<Datapoint*>& ) = 0;
  virtual ~ParetoAlgo(){};
};

#endif
