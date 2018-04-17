/*
&the header file for the instruction set
*/ 
#ifndef INSTRUCTIONSET_H
#define INSTRUCTIONSET_H
#include "pop.h"

void ins(ind &newind, int loc);	//special
void absf(ind &newind);		//0
void add(ind &newind);		//1  
void cosf(ind &newind);		//2	
void DEL0(ind &newind);		//3
void DEL1(ind &newind);		//4	
void divL(ind &newind);		//5
void divR(ind &newind);		//6
void DNL(ind &newind);		//7
void DNR(ind &newind);		//8
void FLIP(ind &newind);		//9
void mul(ind &newind);		//11
void NOOP(ind &newind);	//12	
void sinf(ind &newind);	//13
void subL(ind &newind);	//14
void subR(ind &newind);	//15
void totheL(ind &newind);	//16
void totheR(ind &newind);	//17
void UP(ind &newind);		//18
void UP2(ind &newind);		//19
void UP3(ind &newind); //20
void expf(ind& newind); //21
void logf(ind& newind); //22

#endif