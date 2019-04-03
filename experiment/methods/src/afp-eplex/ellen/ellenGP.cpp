// Devilep.cpp : Defines the entry point for the console application.
// This is the main program for running Developmental Linear Epigenetic Programming, or Devilep. 

#include "stdafx.h"
// mine
#include <string.h>
#include <string>
//#include "pop.h"

//#include <exception>
//#include "logger.h"

//#include "data.h"
//#include "InitPop.h"
//#include "Fitness.h"
//#include "stdafx.h"
//#include "state.h"
//#include "Generation.h"
//#include "HillClimb.h"
//#include "instructionset.h"
//#include "evaluator.h"
//#include "strdist.h"
//#include <ctime>
//#include <cstring>
#include "runEllenGP.h"
using namespace std;

//// global parameters structure
//struct params p; 
//struct Data d;
//struct state s;
//class logger fcout;
////struct evaluator e;
//vector <Randclass> r;


//void load_params(params &p, std::ifstream& is);
//void load_data(Data &d, std::ifstream& is);
//bool stopcondition(float&);
//void printstats(tribe&,int&,int&);

//template<typename T>
//int _tmain(int argc, _TCHAR *argv[])
int main(int argc, char** argv)
{
	/*try 
	{*/
		string paramfile(argv[1]);
		string datafile(argv[2]);
		//runEllenGP(paramfile.c_str(),datafile.c_str(),0);
		runEllenGP(argv[1],argv[2],0,0);
	/*}
	catch(exception& er) 
	{
		cout << "Error: " << er.what() << "\n Exiting in 5 seconds...\n";
		#if defined(_WIN32)
			_sleep(5000);
		#else
			sleep(5);
		#endif
		exit(1);

	}
	catch(...)
	{
		cout << "Exception Occurred. Exiting in 5 seconds...\n";
		#if defined(_WIN32)
			_sleep(5000);
		#else
			sleep(5);
		#endif
		exit(1);
	}*/
	
	return 0;
}
