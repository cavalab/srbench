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

void pareto(vector<ind>& pop,std::vector<float>& data){

  /////////////////////////////////
  // Command line parameters
  //std::vector< std::pair<char*,int> > files; 
	std::string algoName; // name of specific paretro algorithm to use

  ////////////////////////////////////
  // Read data

  std::vector<Datapoint*> dataset; //dataset will contain N K-dim datapoints
  //size_t idmax_file1; // for sanity check to ensure equal file lengths
  for (size_t f=0; f<pop.size(); ++f){
		Datapoint *d = new Datapoint(f);
		dataset.push_back(d);
		for (size_t i=0; i<data.size()/pop.size(); ++i)
			dataset[f]->addNumber(-data.at(f*(data.size()/pop.size())+i)); 
  }


  //////////////////////////////////////
  // Choose the Pareto Algorithm to use
  if (algoName.empty()){
    if (data.size()/pop.size()==2){
      algoName="stablesort";
    }
    else {
      algoName="bruteforce";
    }
  }
  //algoName="bruteforce";
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
