#ifndef PARETO_STABLESORT_ALGO_H_
#define PARETO_STABLESORT_ALGO_H_

//////////////////////////////
// StablesortAlgo: 
// Fast O(N*log(N)) pareto algo for K=2 cases

#include <vector>
#include "Datapoint.h"
#include "ParetoAlgo.h"

class StablesortAlgo : public ParetoAlgo {
 public: 
  virtual int computeFrontier(std::vector<Datapoint*>& );
};

#endif
