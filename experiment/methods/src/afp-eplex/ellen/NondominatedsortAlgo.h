#ifndef PARETO_NONDOMINATEDSORT_ALGO_H_
#define PARETO_NONDOMINATEDSORT_ALGO_H_

//////////////////////////////
// NondominatedsortAlgo: 
// Look at all pairwise datapoints O(N^2) and output Pareto levels
// where the higest level is the Pareto front, followed by second front 
// taken by peeling off the first front, etc.
//
// Reference: 
//   K. Deb (2001), Multi-Objective Optimization using Evolutionarty Algorithms,
//   Wiley & Sons Publishing -- NSGA method, pg. 43

#include <vector>
#include "Datapoint.h"
#include "ParetoAlgo.h"

class NondominatedsortAlgo : public ParetoAlgo {
 public: 
  virtual int computeFrontier(std::vector<Datapoint*>& );
};

#endif
