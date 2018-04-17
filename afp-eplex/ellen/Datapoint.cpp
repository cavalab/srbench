#include "stdafx.h"
#include <iostream>
#include "Datapoint.h"

Datapoint::Datapoint(size_t id) : id(id), paretoStatus(-1), dominatedCount(0) {}

void Datapoint::addNumber(float number){
  vec.push_back(number);
}

void Datapoint::print() const {
  std::cout << id << " " << paretoStatus << " : " ;
  for (size_t k=0; k<vec.size();++k){
    std::cout << vec[k] << " ";
  }
  std::cout << std::endl;
}


