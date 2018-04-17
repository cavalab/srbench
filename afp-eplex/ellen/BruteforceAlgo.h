#ifndef PARETO_BRUTEFORCE_ALGO_H_
#define PARETO_BRUTEFORCE_ALGO_H_

//////////////////////////////
// BruteforceAlgo: 
// Look at all pairwise datapoints O(N^2) and check if which isn't dominated

#include <vector>
#include "Datapoint.h"
#include "ParetoAlgo.h"

class BruteforceAlgo : public ParetoAlgo {
 public: 
  virtual int computeFrontier(std::vector<Datapoint*>& );
};

#endif
