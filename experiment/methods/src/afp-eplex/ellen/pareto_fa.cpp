//////////////////////////////////////////////////////////////
// PARETO: computes the pareto frontier given a set of points
//////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "pop.h"
#include "Datapoint.h"
#include "ParetoAlgo.h"
#include "StablesortAlgo.h"
#include "BruteforceAlgo.h"
#include "NondominatedsortAlgo.h"
//#include "pop.h"
using namespace std;

void usage(){
  std::cerr << "[usage] pareto (-a algorithmName) -l file1 -l file2 ... -l fileK \n \
  Computes pareto frontier given N K-dimensional datapoints. \n \
  There are K>1 files, each consisting of a column of N numbers. \n \
  The flag -l indicates larger number is better (maximize); \n \
  alternatively, the -s flag means smaller is better (minimize). \n \
  The output consists of N labels, where 1=pareto and 0=not. \n\n \
  Flag -a specifies the algorithm to use. \n \
  Supported algorithms: \n \
    -a bruteforce (default for K>2) \n \
    -a stablesort (default for K=2) \n \
    -a nondominatedsort (generates ranking, with higher number meaning higher level of pareto front)\n \
  " << std::endl;

  exit(1);
}

void pareto_fa(vector<ind>& pop){

  /////////////////////////////////
  // Command line parameters
  //std::vector< std::pair<char*,int> > files; 
	std::string algoName; // name of specific paretro algorithm to use

  ////////////////////////////////////
  // Read data
  //int K = files.size(); // # of files, i.e. K-dimensional problem
  int K = 2;
  if (K<2){
    std::cerr << "[error] Insufficient dimension (2 or more input files needed). " << std::endl;
    usage();
  }

  std::vector<Datapoint*> dataset; //dataset will contain N K-dim datapoints
  //size_t idmax_file1; // for sanity check to ensure equal file lengths
  for (size_t f=0; f<pop.size(); ++f){
		Datapoint *d = new Datapoint(f);
		dataset.push_back(d);
		dataset[f]->addNumber(-pop.at(f).fitness); 
		dataset[f]->addNumber(-pop.at(f).age);
  }


  //////////////////////////////////////
  // Choose the Pareto Algorithm to use
  /*if (algoName.empty()){
    if (K==2){
      algoName="stablesort";
    }
    else {
      algoName="bruteforce";
    }
  }*/
  algoName="stablesort";
  ParetoAlgo* algo = NULL;
  if (algoName == "stablesort"){
    algo = new StablesortAlgo();
  }
  else if (algoName == "bruteforce") {
    algo = new BruteforceAlgo();
  }
  else if (algoName == "nondominatedsort"){
    algo = new NondominatedsortAlgo();
  }
  else {
    std::cout << "[error] Unknown AlgoName: " << algoName << std::endl;
  }


  ////////////////////////////////////
  // Compute the Pareto Frontier! 
  int numPareto = algo->computeFrontier(dataset);
 // std::cerr << "#pareto: " << numPareto << " of " << dataset.size() << " total, by the " << algoName << " algorithm " << std::endl;
  for (size_t d=0; d<dataset.size(); ++d){
    //dataset[d]->print();
	  pop.at(d).rank = dataset[d]->getParetoStatus();
  }

  for(std::vector<Datapoint*>::iterator j=dataset.begin();j!=dataset.end();++j)
	  delete(*j);
  delete(algo);
}
