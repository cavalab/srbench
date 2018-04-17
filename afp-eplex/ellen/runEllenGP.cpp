#include "stdafx.h"
#include <string>
// mine
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "data.h"
#include "state.h"
#include "logger.h"

#include "InitPop.h"
#include "FitnessEstimator.h"
#include "Fitness.h"
#include "Generation.h"
#include "instructionset.h"
//#include "omp.h"
#include "Generationfns.h"
#include "strdist.h"
#include <time.h>
#include <cstring>
#include "p_archive.h"
#include "Eqn2Line.h"
#include "general_fns.h"
#include "load_params.h"
#include "load_data.h"
#include "printing.h"
// #include "runEllenGP.h"
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

using namespace std;
#if defined(_WIN32)
	#include <direct.h>
	#define GetCurrentDir _getcwd
#else
	#include <unistd.h>
	#include <iomanip>
	#define GetCurrentDir getcwd
#endif

// includes for python integration
#include <boost/python.hpp>
namespace bp = boost::python;
#include <numpy/ndarrayobject.h>
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// global parameters structure

bool check_genty(vector<ind>& pop,params& p){

	for(int count = 0; count<pop.size(); ++count)
	{
		float tmp = abs(pop[count].fitness-pop[count].fitness_v)/pop[count].fitness;
		if (pop.at(count).genty != tmp && pop.at(count).genty != p.max_fit)
			return 0;
	}
	return 1;
}
void printGenome(tribe& T,int gen,string& logname,Data& d,params& p)
{
	// print genome to file in csv format
	// create Data file
	// print format: print in head-to-tail fashion with grid locations
	// x y gene
	// gene encoding: +:1,-:2,*:3,/:4,sin:5,cos:6,exp:7,log:8,constant:9,vars:10-length(vars)

	//total genes
	 std::ofstream gout_all;
	 std::ofstream gout_on;
	 std::ofstream gout_off;
	 string filename0 = logname.substr(0,logname.size()-4)+"_g_all.csv."+std::to_string(static_cast<long long>(gen));
	 gout_all.open(filename0);
	 gout_all << "x,y,fit,age,g,e\n";
	// on genes
	 if (p.eHC_on){

		 string filename = logname.substr(0,logname.size()-4)+"_g_on.csv."+std::to_string(static_cast<long long>(gen));
		 gout_on.open(filename);
		 gout_on << "x,y,fit,age,g,e\n";
		 // off genes

		 string filename2 = logname.substr(0,logname.size()-4)+"_g_off.csv."+std::to_string(static_cast<long long>(gen));
		 gout_off.open(filename2);
		 gout_off << "x,y,fit,age,g,e\n";
	 }
	 string out;
	 string tmp;
	 int k;
	 float y,y_on, y_off;
	 float max_fit=0;
	 float min_fit=10000;
	 float max_age=0;
	 float min_age=100000;
	 float max_line=0;
	 float min_line = 1000;
	 for (size_t i =0; i<T.pop.size(); ++i){
		if (log(1+T.pop[i].fitness)>max_fit) max_fit=log(1+T.pop[i].fitness);
		if (log(1+T.pop[i].fitness)<min_fit) min_fit=log(1+T.pop[i].fitness);
		if (T.pop[i].age>max_age) max_age=T.pop[i].age;
		if (T.pop[i].age<min_age) min_age=T.pop[i].age;
		for (size_t j = 0; j<T.pop[i].line.size(); ++j){
			if (T.pop[i].line.size()>max_line) max_line=T.pop[i].line.size();
			if (T.pop[i].line.size()<min_line) min_line=T.pop[i].line.size();
		}
	}
	if(p.sel==4) T.sortpop_age();
	else T.sortpop();

	 for (size_t i=0; i<T.pop.size(); ++i){
		 y=0;
		 y_on=0;
		 y_off=0;
		 for (int j=T.pop[i].line.size()-1; j>=0; --j){
			 switch(T.pop[i].line[j].type){
			 case '+':
				 out="1";
				 break;
			 case '-':
				 out="2";
				 break;
			 case '*':
				 out="3";
				 break;
			 case '/':
				 out="4";
				 break;
			 case 's':
				 out="5";
				 break;
			 case 'c':
				 out="6";
				 break;
			 case 'e':
				 out="7";
				 break;
			 case 'l':
				 out="8";
				 break;
			 case 'n':
				 out="9";
				 break;
			 case 'v':
				 k=0;
				 while (T.pop[i].line[j].varname.compare(d.label[k])!=0)
					 ++k;
				 tmp=std::to_string(static_cast<long long>(k+10));
				 out = tmp;
				 break;
			 }
			  gout_all << float(float(i)/float(T.pop.size())) << "," << y/p.max_len << "," << (log(1+T.pop[i].fitness)-min_fit)/(max_fit-min_fit) << "," << (float(T.pop[i].age)-min_age)/(max_age-min_age) << "," << out << "," << T.pop[i].line[j].on << "\n";
			 ++y;
			 if (p.eHC_on){
				 if (T.pop[i].line[j].on){
				 gout_on << float(float(i)/float(T.pop.size())) << "," << y_on/p.max_len << "," << (log(1+T.pop[i].fitness)-min_fit)/(max_fit-min_fit) << "," << (float(T.pop[i].age)-min_age)/(max_age-min_age) << "," << out << "," << T.pop[i].line[j].on << "\n";
				 ++y_on;
				 }
				 else{
				  gout_off << float(float(i)/float(T.pop.size())) << "," << y_off/p.max_len << "," << (log(1+T.pop[i].fitness)-min_fit)/(max_fit-min_fit) << "," << (float(T.pop[i].age)-min_age)/(max_age-min_age) << "," << out << "," << T.pop[i].line[j].on << "\n";
				  ++y_off;
				 }
			//++y;
			 }
		 }
	 }
	 gout_all.close();
	 gout_off.close();
		 gout_on.close();
	 /*if (p.eHC_on) {

	 }*/
}
void printstats(tribe& T,int &i,state& s,params& p,paretoarchive& A){
//boost::progress_timer timer;

s.out << "--- Generation " << i << "---------------------------------------------------------------" << "\n";
s.out << "population size: " << T.pop.size() << "\n";
s.out << "Number of evals: " << s.genevals.back() << "\n";
s.out << "Best Fitness: " << T.bestFit() <<"\n";
s.out << "Best Fitness (v): " << T.bestFit_v() <<"\n";
s.out << "Median Fitness: " << T.medFit_v()<<"\n";
s.out << "Median Fitness (v): " << T.medFit()<<"\n";
s.out << "Mean Size: " << T.meanSize() << "\n";
s.out << "Mean Eff Size: " << T.meanEffSize() << "\n";
s.out << "Pareto Front Equations: " << A.optimal_size << "\n";
if (p.pHC_on) {
	s.out << "Parameter updates: " << float(s.setPHCupdates()) / float(p.popsize) * 100 << "%\n";
	s.out << "Epigenetic updates: " << float(s.setEHCupdates()) / float(p.popsize) * 100 << "%\n";
	s.out << "Epigenetic ties: " << float(s.setEHCties()) / float(p.popsize) * 100 << "%\n";
}
s.out << "Beneficial Genetics: " << s.getGoodCrossPct() << "%\n";
s.out << "Neutral Genetics: " << s.getNeutCrossPct() << "%\n";
s.out << "Bad Genetics: " << s.getBadCrossPct() << "%\n";

if (p.classification){
	s.out << "Fitness \t Equation \n";
	vector <sub_ind> besteqns;
	T.topTen(besteqns);
	//for(unsigned int j=0;j<min(10,int(A.pop.size()));++j)
	//	s.out <<A.pop.at(j).abserror_v << "\t" << A.pop.at(j).corr_v << "\t" << A.pop.at(j).eqn <<"\n";
	for(unsigned int j=0;j<besteqns.size();++j)
		s.out << besteqns.at(j).fitness << "\t" << besteqns.at(j).eqn <<"\n";

}
else{
	s.out << "MAE \t R^2 \t Fitness \t Equation \n";
	vector <sub_ind> besteqns;
	T.topTen(besteqns);
	//for(unsigned int j=0;j<min(10,int(A.pop.size()));++j)
	//	s.out <<A.pop.at(j).abserror_v << "\t" << A.pop.at(j).corr_v << "\t" << A.pop.at(j).eqn <<"\n";
	for(unsigned int j=0;j<besteqns.size();++j){
		s.out << setprecision(3) << besteqns.at(j).abserror << "\t" << setprecision(3) << besteqns.at(j).corr << "\t" << setprecision(3) << besteqns.at(j).fitness << "\t" << besteqns.at(j).eqn <<"\n";
		/*if(boost::math::isnan(besteqns.at(j).abserror))
		{
			cout << "equation with NaN error: " + besteqns.at(j).eqn + "\n";
		}*/
	}
}
//if (p.classification && p.class_m4gp){
//	ind best_ind;
//	T.getbestind(best_ind);
//	s.out << "best M:\n";
//	s.out << best_ind.M << "\n";
//	for (unsigned i = 0; i<p.number_of_classes; ++i){
//		s.out << "best C[" << i << "]:\n";
//		s.out << best_ind.C[i] << "\n";
//	}
//}
s.out << "-------------------------------------------------------------------------------" << "\n";
}


int load_pop(vector<ind>& pop,params& p,state& s)
{
	// load population from file filename.
	ifstream fs(p.pop_restart_path);
	if(!fs.is_open())
	{
		std::cerr << "Error: couldn't open population file " + p.pop_restart_path + ".\n";
		exit(1);
	}

	string s0;
	string varname;
	float tmpf;
	//int tmpi;
	string tmps;
	bool tmpb;

	//string trash;
	//s.erase();
    //s.reserve(is.rdbuf()->in_avail());
	int i=0;
	int j=0;
	while(!fs.eof() && i<=p.popsize)
    {
		getline(fs,s0,'\n');
		istringstream ss(s0);
		ss >> varname;

		if(varname.compare(0,6,"gline:") == 0 && i<p.popsize)
		{
			//pop.push_back(ind());

			while (ss>>tmps){
				if (tmps.compare("+")==0 || tmps.compare("-")==0 || tmps.compare("*")==0 || tmps.compare("/")==0 || tmps.compare("s")==0 || tmps.compare("c")==0 || tmps.compare("e")==0 || tmps.compare("l")==0 || tmps.compare("!") == 0 || tmps.compare("=") == 0 || tmps.compare("<") == 0 || tmps.compare(">") == 0 || tmps.compare("{") == 0 || tmps.compare("}")==0 || tmps.compare("i") == 0 || tmps.compare("t") == 0 || tmps.compare("&") == 0 || tmps.compare("|") == 0) // operator node
					pop[i].line.push_back(node(char(tmps[0])));
				else if (isdigit(tmps[0]) || tmps[0]=='-') // constant node
					pop[i].line.push_back(node(std::stof(tmps)));
				else //variable node
					pop[i].line.push_back(node(tmps));

			}
			++i;
		}
		else if(p.eHC_on && varname.compare(0,6,"eline:") == 0)
		{
			while (ss>>tmpb){
				pop[i-1].line[j].on = tmpb;
				++j;
			}
			j=0;
		}
		/*else if(varname.compare(0,4,"age:") == 0)
			ss>>pop[i-1].age;*/
	}
	if (!fs.eof()) s.out << "WARNING: truncating population from file to first " + to_string(static_cast<long long>(p.popsize)) + " individuals...\n";
	return i; // size of loaded population
}

void shuffle_data(Data& d, params& p, vector<Randclass>& r,state& s)
{
	vector<int> shuffler;
	vector<float> newtarget;
	vector<vector<float>> newvals;

	for(int i=0;i<d.vals.size();++i)
		shuffler.push_back(i);

	std::random_shuffle(shuffler.begin(),shuffler.end(),r[omp_get_thread_num()]);


	if (p.EstimateFitness){
		bool tmp = s.out.print_to_file;
		s.out.print_to_file=1; // keep output from going to console
		s.out << "data shuffle index: ";
		for (int i=0; i<shuffler.size(); ++i)
			s.out << shuffler.at(i) << " ";
		s.out << "\n";
		s.out.print_to_file=tmp;
	}
	for(int i=0;i<d.vals.size();++i)
	{
		newtarget.push_back(d.target.at(shuffler.at(i)));
		newvals.push_back(d.vals.at(shuffler.at(i)));

	}
	using std::swap;
	std::swap(d.target,newtarget);
	std::swap(d.vals,newvals);
}
bool stopcondition(tribe& T,params p,Data& d,state& s,FitnessEstimator& FE)
{
	if (!p.stop_condition)
		return false;
	if (!p.EstimateFitness){
		if (T.bestFit() <= p.stop_threshold){
			s.out << "best fitness criterion achieved: " << T.bestFit() << "<" << p.stop_threshold << "\n";
			return true;
		}
		else
			return false;
	}
	else{
		vector<ind> best(1);
		T.getbestind(best[0]);

		p.EstimateFitness=0;
		Fitness(best,p,d,s,FE);
		p.EstimateFitness=1;
		if (best[0].fitness <= p.stop_threshold){
			s.out << "best fitness criterion achieved: " << best[0].fitness << "<" << p.stop_threshold << "\n";
			return true;
		}
		else
			return false;
	}
}
int get_next_task(int& index,vector<int>& task_assignments)
{
	// 0: evolve solution
	// 1: evolve fitness estimation
	// 2: print output (after all 0 tasks finish)
	// 3: update pareto archive (after all 0 tasks finish)
	if (index == task_assignments.size()-1)
		return -1;
	else{
		return task_assignments.at(++index);
	}
}
// template <size_t samples, size_t features>
// void runEllenGP(dict& param_dict, float (&X)[samples][features], float (&Y)[samples],string pname,string dname)
// numeric::array& features, numeric::array& target

void reference_contiguous_array(PyObject* in, float* &ptr, vector<int>& dims)
{ // returns pointer to underlying c array (ptr) from PyObject in, checking for contiguos memory storage
	  PyArrayObject* in_con;
		PyArray_Descr* x = PyArray_DESCR(in);
		// std::cout << "array dtype: " << (*x).type << "\n";
		// std::cout << "array kind: " << (*x).kind << "\n";
		// std::cout << "array byteorder: " << (*x).byteorder << "\n";
		// std::cout << "array flags: " << (*x).flags << "\n";
		// make sure data is contigous
    in_con = PyArray_GETCONTIGUOUS((PyArrayObject*)in);

		// get pointer to c array
    ptr = (float*)PyArray_DATA(in_con);


		// get dimensions
		int num_dim = PyArray_NDIM(in_con);
    npy_intp* pdim = PyArray_DIMS(in_con);
		// std::cout << "PyArray_NDIM:" << num_dim << "\n";

    for (int i = 0; i < num_dim; i++){
        dims.push_back(pdim[i]);
				// std::cout << "dim" << i << ": " << dims[i] << "\n";
			}
		// for (int i = 0; i < dims[0]; ++i){
		// 	std::cout << "*(ptr + " << i << "): " << *(ptr+i) << "\n";
		// }
}
void dereference(PyObject* o)
{
    Py_DECREF(o);
}

void line_to_py(vector<node>& line,bp::list& prog){
	// converts program to tuple for export to python.
	// cout << "in line to py\n";
	for (auto n: line){
		if (n.on){
			switch(n.type)
			{
			case 'v':
				if(n.varname.compare(0,6,"target")==0){
					// cout << "autoregressing " << n.varname << "\n";
					prog.append(bp::make_tuple("y", n.arity_float + n.arity_bool, stoi(string(n.varname.begin()+7,n.varname.end()))));
				}
				else{
					if (n.varname.find("_d") != std::string::npos) {
						size_t pos = n.varname.find("_d");
						// this is an auto-regressive variable
						// cout << "auto-regressive variable index: " << string(n.varname.begin()+2,n.varname.begin()+pos) << "\n";
						// cout << "auto-regressive variable delay: " << string(n.varname.begin()+pos+2,n.varname.end()) << "\n";
						prog.append(bp::make_tuple("xd", n.arity_float + n.arity_bool, stoi(string(n.varname.begin()+2,n.varname.begin()+pos)),stoi(string(n.varname.begin()+pos+2,n.varname.end()))));
					}
					else{
						// cout << n.varname << " regular\n";
						prog.append(bp::make_tuple("x", n.arity_float + n.arity_bool, stoi(string(n.varname.begin()+2,n.varname.end()))));
					}
				}
				break;
			case 'n':
				prog.append(bp::make_tuple("k", n.arity_float + n.arity_bool, n.value));
				break;
			default:
				prog.append(bp::make_tuple(n.type, n.arity_float + n.arity_bool));
			}
		}
	}
}
void pop_to_py(vector<ind>& archive,bp::list& arch_list){
	// converts program to tuple for export to python.
	// cout << "pop_to_py\n";
	// cout << "size of vector<ind> archive: " << archive.size() << "\n";
	for (auto i: archive){
	// for (unsigned int i = 0; i < archive.size(); ++i){
		// cout << i << "\n";
		// cout << i.eqn << "\n";
		bp::list prog;
		line_to_py(i.line,prog);
		arch_list.append(bp::make_tuple(prog,i.fitness,i.fitness_v));
	}
	// cout << "size of arch_list: " << bp::len(arch_list) << "\n";
	// cout <<"beepbopboop";
}

void runEllenGP(bp::dict& param_dict, PyObject* features, PyObject* target, bp::list& best_prog) //string pname,string dname
{
	// MARK_FUNCTION
	try{
		bool trials = 0;
		int trialnum = 0;
		string pname = "ellenGP";
		string dname = "d";
		// string paramfile = "ellyn";
		// string datafile = "d";
		//string paramfile(param_in);
		//string datafile(data_in);
		/* ===================================
		steps:
		Initialize population
			make genotypes
			genotype to phenotype
			calculate fitness
			hill climb
		Next Generation
			Select Parents
			Create children genotypes
			genotype to phenotype
			calculate fitness
			hill climb
		Store Statistics
		Print Update

		INPUTS
		paramfile: parameter file
		datafile: data set: target in first column, dependent variables in second column
		=================================== */

		struct params p;
		struct Data d;
		struct state s;

		vector <Randclass> r;

		// load parameter file
		p.set(param_dict);

		// get the input array
		float* feat_ptr;
		vector<int> dims;
		reference_contiguous_array(features, feat_ptr, dims);

		// set up features
		d.set_train(feat_ptr,dims[0],dims[1]);

		float* target_ptr;
	  dims.resize(0);

		reference_contiguous_array(target, target_ptr, dims);
		d.set_target(target_ptr,dims[0]);
		Py_DECREF(features);
		Py_DECREF(target);

		d.set_dependencies(p);
		// load_data(d,ds,p);

		std::time_t t =  std::time(NULL);


	#if defined(_WIN32)
		std::tm tm;
		localtime_s(&tm,&t);
		char tmplog[100];
		strftime(tmplog,100,"%Y-%m-%d_%H-%M-%S",&tm);
	#else
		std::tm * tm = localtime(&t);
		char tmplog[100];
		strftime(tmplog,100,"%F_%H-%M-%S",tm);
	#endif


	   // string tmplog = "777";
		const char * c = p.resultspath.c_str();
		#if defined(_WIN32)
			_mkdir(c);
		#else
			mkdir(c, 0777); // notice that 777 is different than 0777
		#endif
		 int thrd = omp_get_thread_num();
		 string thread;
		 if (trials) thread = std::to_string(static_cast<long long>(trialnum));
		 else thread = std::to_string(static_cast<long long>(thrd));
	// define save file name
	string logname;
	#if defined(_WIN32)
		if (!p.savename.empty())
			logname = p.resultspath + '/' + p.savename + ".log";
		else
	  	logname = p.resultspath + '\\' + "ellyn_" + tmplog + "_" + pname + "_" + dname + "_" + thread + ".log";
	#else
		if (!p.savename.empty())
			logname = p.resultspath + '/' + p.savename + ".log";
		else
			logname = p.resultspath + '/' + "ellyn_" + tmplog + "_" + pname + "_" + dname + "_" + thread + ".log";
	#endif


		 s.out.set_ptf(trials && p.print_log); // only print to file if print_log

		 s.out.set_v(p.verbosity>1); // only print a log file to screen if verbosity > 1
		 // print log setup
		 if (p.print_log){
			 s.out.open(logname);
			 if (!s.out.is_open()){
				 cerr << "Write-to File " << p.resultspath + '/' + logname << " did not open correctly.\n";
				 exit(1);
			 }
		 	}
		 // initialize data file
		 std::ofstream dfout;
		 if (p.print_data)	initdatafile(dfout,logname,p);



		 if (p.verbosity>0){
			 s.out << "_______________________________________________________________________________ \n";
			 s.out << "                                    ellenGP                                     \n";
			 s.out << "_______________________________________________________________________________ \n";
			 //s.out << "Time right now is " << std::put_time(&tm, "%c %Z") << '\n';
			 s.out<< "Results Path: " << p.resultspath  << "\n";
			 if (!p.savename.empty())
			 	s.out << "Save filename: " << p.savename << "\n";
			 s.out << "parameter name: " << pname << "\n";
			 s.out << "data file: " << dname << "\n";
			 if(trials) {s.out << "Running in Trial Mode\n";
			 s.out << "Trial ID: " << trialnum << "\n";
			 }
			 s.out << "Settings: \n";
			 // get evolutionary method
			 s.out << "Evolutionary Method: ";
			 switch(p.sel){
				 case 1:
					 s.out << "Standard Tournament\n";
					 break;
				 case 2:
					 s.out << "Deterministic Crowding\n";
					 break;
				 case 3:
					 s.out << "Lexicase Selection\n";
					 break;
				 case 4:
					 s.out << "Age-Fitness Pareto\n";
					 break;
			 }
			 if (p.sel==3){
				 s.out << "Lexicase metacases:\n";
				// add metacases if needed
				for (auto i: p.lex_metacases)
					s.out << "\t" << i << "\n";
			 }
			 if (p.ERC) s.out << "ERCs on\n";
			 if (p.pHC_on) s.out << p.pHC_its << " iterations of Parameter Hill Climbing\n";
			 if (p.eHC_on) s.out << p.eHC_its << " iterations of Epigenetic Hill Climbing\n";
			 if(p.train) s.out << "Data split " << p.train_pct << "/" << 1-p.train_pct << " for training and validation.\n";
			 s.out << "Total Population Size: " << p.popsize << "\n";
			 if (p.limit_evals) s.out << "Maximum Point Evals: " << p.max_evals << "\n";
			 else s.out << "Maximum Generations: " << p.g << "\n";
			 s.out << "Number of log points: " << p.num_log_pts << " (0 means log all points)\n";
			 if (trials && p.islands){
				s.out << "WARNING: cannot run island populations in trial mode. This trial will run on one core.\n";
				p.islands = false;
			 }
			 s.out << "fitness type: " << p.fit_type << "\n";
			 s.out << "verbosity: " << p.verbosity << "\n";
		  }
		 // allow threads (release the Python GIL) python gil unlock
		// Py_BEGIN_ALLOW_THREADS{
		int nt=0;
		//int ntt=0;
		// if specified by the user, override the default omp number of threads
		if (p.islands && p.num_islands !=0){
			nt = p.num_islands;
			omp_set_num_threads(nt);
		}
		else{
			#pragma omp parallel
			{
				nt = omp_get_num_threads();
			}
		}
		if (p.verbosity>0) s.out << "Number of threads: " << nt << "\n";
		//s.out << "OMP Number of threads: " << omp_get_num_threads() << "\n";
		p.nt = nt;

		//initialize random number generator
		unsigned int seed1 = int(time(NULL));
		if (p.verbosity>0) s.out << "seeds: \n";
		r.resize(omp_get_max_threads());
		#pragma omp parallel
		{
				//cout << "seeder: " << seeder <<endl;
				//cout << "seed1: " << seed1*seeder <<endl;
				if(!trials){
					if (p.verbosity>0) s.out << to_string(static_cast<long long>(seed1*(omp_get_thread_num()+1))) + "\n";
					//r.at(seeder).SetSeed(seed1*(seeder+1));
					r.at(omp_get_thread_num()).SetSeed(seed1*(omp_get_thread_num()+1));
				}
				else
					r.at(omp_get_thread_num()).SetSeed(seed1*(omp_get_thread_num()+1)*trialnum);
		}
		if(trials)
			if (p.verbosity>0) s.out << (omp_get_thread_num()+1)*seed1*trialnum << "\n";

		//shuffle data for training
		if (p.shuffle_data)
			shuffle_data(d,p,r,s);

		// define class weights for wighted F1 fitness
		if (p.classification && (p.fit_type.compare("2")==0 || p.fit_type.compare("F1W")==0))
			d.define_class_weights(p);

		boost::timer time;
		paretoarchive A;
		if (p.prto_arch_on){
			A.resize(p.prto_arch_size);
			// tribe FinalArchive(p.prto_arch_size,p.max_fit,p.min_fit);
			// FinalArchive.pop = A.pop;
		}

		vector<FitnessEstimator> FE(1);
		vector<ind> trainers;
	  //////////////////////////////////////////////////////////////////////////////
		// islands
		//////////////////////////////////////////////////////////////////////////////
		if (p.islands)
		{
			//p.parallel=false;
			// determine number of threads

			//int num_islands=omp_get_max_threads(); //
			int num_islands=nt;
			int subpops = p.popsize/num_islands;
			p.popsize = subpops*num_islands;
			vector<tribe> T;
			tribe World(subpops*num_islands,p.max_fit,p.min_fit); //total population of tribes
			if (p.verbosity>0) s.out << num_islands << " islands of " << subpops << " individuals, total pop " << p.popsize <<"\n";


			for(int i=0;i<num_islands;++i)
				T.push_back(tribe(subpops,p.max_fit,p.min_fit));
			// run separate islands
			if (p.pop_restart) // initialize population from file
			{
				if (p.verbosity>0) s.out << "loading pop from " + p.pop_restart_path + "...\n";
				int tmp = load_pop(World.pop,p,s);
				if (tmp < p.popsize){
					if (p.verbosity>0) s.out << "WARNING: population size loaded from file (" << tmp << ") is smaller than set pop size (" << p.popsize << "). " << (p.popsize-tmp) << " randomly initiated individuals will be added...\n";
					vector<ind> tmppop(p.popsize-tmp);
					InitPop(tmppop,p,r);
					swap_ranges(World.pop.end()-(p.popsize-tmp),World.pop.end(),tmppop.begin());
				}
				// initialize fitness estimation pop
				if (p.EstimateFitness)
					InitPopFE(FE,World.pop,trainers,p,r,d,s);

				Fitness(World.pop,p,d,s,FE[0]);

				std::random_shuffle(World.pop.begin(),World.pop.end(),r[0]);
				//assign population to islands
				#pragma omp parallel for
				for (int q = 0; q<num_islands; ++q)
					T[q].pop.assign(World.pop.begin()+q*subpops,World.pop.begin()+(q+1)*subpops);
			}
			else // initialize population from random
			{
				if (p.init_validate_on)
				{
					if (p.verbosity>0) s.out << "Initial validation...";


					#pragma omp parallel for
					for(int i=0;i<num_islands;++i)
						InitPop(T.at(i).pop,p,r);

					if (p.EstimateFitness)
					{
						// construct world population
						for(int j=0;j<T.size();++j){
							for(int k=0;k<T[0].pop.size();++k){
								World.pop.at(j*T[0].pop.size()+k)=T.at(j).pop.at(k);
								//makenew(World.pop[j*T[0].pop.size()+k]);
							}
						}
						// initialize fitness estimation pop
						InitPopFE(FE,World.pop,trainers,p,r,d,s);
					}
					//discard invalid individuals
					#pragma omp parallel for
					for(int i=0;i<num_islands;++i){
						float worstfit;
						float bestfit;
						vector<ind> tmppop;

						Fitness(T.at(i).pop,p,d,s,FE[0]);
						worstfit = T.at(i).worstFit();
						bestfit = T.at(i).bestFit();


						int counter=0;
						while(worstfit == p.max_fit && counter<100)
						{
							for (vector<ind>::iterator j=T.at(i).pop.begin();j!=T.at(i).pop.end();)
							{
								if ( (*j).fitness == p.max_fit)
								{
									j=T.at(i).pop.erase(j);
									tmppop.push_back(ind());
								}
								else
									++j;
							}

							InitPop(tmppop,p,r);
							Fitness(tmppop,p,d,s,FE[0]);
							T.at(i).pop.insert(T.at(i).pop.end(),tmppop.begin(),tmppop.end());
							tmppop.clear();
							worstfit = T.at(i).worstFit();
							counter++;
							if(counter==100)
								if (p.verbosity>0) s.out << "initial population count exceeded. Starting evolution...\n";
						}
					}
					s.setgenevals();
					if (p.verbosity>0) s.out << " number of evals: " << s.getgenevals() << "\n";

				}
				else // normal population initialization
				{
					/*bool tmp = p.EstimateFitness;
					p.EstimateFitness=0;*/
					if (p.verbosity>1) s.out << "InitPop...\n";
                    #pragma omp parallel for
					for(int i=0;i<num_islands;++i)
						InitPop(T.at(i).pop,p,r);

					if (p.EstimateFitness){
						// construct world population
						for(int j=0;j<T.size();++j){
							for(int k=0;k<T[0].pop.size();++k){
								World.pop.at(j*T[0].pop.size()+k)=T.at(j).pop.at(k);
								//makenew(World.pop[j*T[0].pop.size()+k]);
							}
						}
						InitPopFE(FE,World.pop,trainers,p,r,d,s);
					}

					#pragma omp parallel for
					for(int i=0;i<num_islands;++i)
						Fitness(T.at(i).pop,p,d,s,FE[0]);

					// construct world population with assigned fitness values
					for(int j=0;j<T.size();++j){
						for(int k=0;k<T[0].pop.size();++k){
							World.pop.at(j*T[0].pop.size()+k)=T.at(j).pop.at(k);
							//makenew(World.pop[j*T[0].pop.size()+k]);
						}
					}

				}
			}
			// use tmpFE for updating in parallel
			vector<FitnessEstimator> tmpFE = FE;

			int gen=0;
			long long termits,term, print_trigger;
			if (p.limit_evals){
				termits = s.totalptevals();
				term = p.max_evals;
				if (p.num_log_pts ==0) print_trigger=0;
				else print_trigger = p.max_evals/p.num_log_pts;

			}
			else {
				print_trigger=0;
				termits=1;
				term = p.g;
			}
			bool pass=1;
			int mixtrigger=p.island_gens*(p.popsize+p.popsize*p.eHC_on+p.popsize*p.pHC_on);
			//int trainer_trigger=0;
			int trainer_trigger=p.FE_train_gens;//p.FE_train_gens*(p.popsize+p.popsize*p.eHC_on+p.popsize*p.pHC_on);

			bool migrate=false;
			int q;

			// construct world population
			if (!p.EstimateFitness || !p.pop_restart){
				int cntr=0;
				for (int q0 = 0; q0<nt;++q0){
					for(int k=q0*subpops;k<(q0+1)*subpops;++k){
						World.pop.at(k)=T.at(q0).pop.at(cntr);
						//makenew(World.pop.at(k));
						++cntr;
					}
					cntr=0;
				}
			}
			if (p.print_init_pop) printpop(World.pop,p,s,logname,3);
			if (p.print_genome) printGenome(World,0,logname,d,p);
			if (p.print_db) printDB(World.pop,logname,d,p);
			#pragma omp parallel private(q) shared(pass)
			{

				q = omp_get_thread_num();

				while(termits<=term)
				{


					if(pass){
						// cout << "thread " + std::to_string(static_cast<long long>(q)) + " Generation\n";
						Generation(T[q].pop,p,r,d,s,FE[0]);

						if (stopcondition(T[q],p,d,s,FE[0])){
							pass=0;
							// cout << "thread " + std::to_string(static_cast<long long>(q)) + " solution\n";
						}
					}

					if (pass) {
						if (p.pHC_on && p.ERC)
						{
								for(int k=0; k<T[q].pop.size(); ++k)
									HillClimb(T[q].pop.at(k),p,r,d,s,FE[0]);
						}
						if (p.eHC_on && !p.eHC_mut)
						{
								for(int m=0; m<T[q].pop.size(); m++)
									EpiHC(T[q].pop.at(m),p,r,d,s,FE[0]);
						}
                        if (p.SGD && p.ERC)
                        {
                            for(int k=0; k<T[q].pop.size(); ++k)
									StochasticGradient(T[q].pop.at(k),p,r,d,s,FE[0],gen);

                        }
						if (stopcondition(T[q],p,d,s,FE[0])){
							pass=0;
							// cout << "thread " + std::to_string(static_cast<long long>(q)) + " solution\n";
						}
					}
					// cout << "thread " + std::to_string(static_cast<long long>(q)) + " pass val: " + std::to_string(static_cast<long long>(pass)) + "\n";


						// construct world population
						int cntr=0;
						//std::vector<ind>::iterator it = T[q].begin();
						//World.pop.assign(
						for(int k=q*subpops;k<(q+1)*subpops;++k){
							World.pop.at(k)=T[q].pop.at(cntr);
							//makenew(World.pop.at(k));
							cntr++;
						}
						// cout << "thread " + std::to_string(static_cast<long long>(q)) + " World pop constructed\n";

						#pragma omp barrier

						if (!pass)
							break;
						// cout << "thread " + std::to_string(static_cast<long long>(q)) + " past the barrier\n";
						#pragma omp single  nowait //coevolve fitness estimators
						{

							if (p.EstimateFitness){


								float aveFEfit=0;
								for (int u=0;u<FE.size();u++)
									aveFEfit+=FE[u].fitness;
								aveFEfit /= FE.size();
								if (aveFEfit==0)
									std::random_shuffle(tmpFE.begin(),tmpFE.end(),r[omp_get_thread_num()]);
								else
									EvolveFE(World.pop,tmpFE,trainers,p,d,s,r);
								if (!p.limit_evals || s.totalptevals() >= print_trigger){
									if (p.verbosity>0) s.out << "Evolving fitness estimators...\n";
									if (p.verbosity>0) s.out << "Best FE fit: " << FE[0].fitness <<"\n";
									if (p.estimate_generality) s.out << "Best FE genty: " << FE[0].genty <<"\n";
									if (p.verbosity>0) s.out << "Ave FE fit: " << aveFEfit << "\n";
									if (p.verbosity>0) s.out << "Current Fitness Estimator:\n";

									if (p.verbosity>0){
										for (int b=0;b<FE[0].FEpts.size();b++)
											 s.out << FE[0].FEpts[b] << " ";
										s.out << "\n";
									}
								}

							}
						}
						#pragma omp single
						{
							s.setgenevals();
							if(s.totalevals()>mixtrigger)
							{
								//shuffle population
								std::random_shuffle(World.pop.begin(),World.pop.end(),r[q]);
								//redistribute populations to islands
								if (p.verbosity>0) s.out << "Shuffling island populations...\n";
								migrate = true;
								mixtrigger+=p.island_gens*(p.popsize+p.popsize*p.eHC_on+p.popsize*p.pHC_on);
							}
							else
								migrate=false;
						}
						#pragma omp single
						{
							if(p.prto_arch_on){
								A.update(World.pop);
								if (p.print_archive) printpop(A.pop,p,s,logname,1);
							}
							if (!p.limit_evals || s.totalptevals() >= print_trigger){
								if (p.print_data) printdatafile(World,s,p,r,dfout,gen,time.elapsed());
								if (p.print_every_pop) printpop(World.pop,p,s,logname,2);
								if (p.verbosity > 1) {
									printstats(World,gen,s,p,A);
									s.out << "Total Time: " << (int)floor(time.elapsed()/num_islands/3600) << " hr " << ((int)(time.elapsed()/num_islands) % 3600)/60 << " min " << (int)(time.elapsed()/num_islands) % 60 << " s\n";
									s.out << "Total Evals: " << s.totalevals() << "\n";
									s.out << "Point Evals: " << s.totalptevals() << "\n";
									s.out << "Average evals per second: " << (float)s.totalevals()/time.elapsed() << "\n";
									s.out << "Average point evals per second: " << (float)s.totalptevals()/(time.elapsed()/num_islands) << "\n";
								}
								if (print_trigger!=0) print_trigger += p.max_evals/p.num_log_pts;
								if (p.print_genome) printGenome(World,gen,logname,d,p);
								if (p.print_db) printDB(World.pop,logname,d,p);
							}
							++gen;
							if (p.limit_evals) termits = s.totalptevals();
							else ++termits;

							if (!p.EstimateFitness && p.estimate_generality && p.G_shuffle)
								shuffle_data(d,p,r,s);

						}

						#pragma omp single
						{
							if (p.EstimateFitness){
								// assign tmpFE to FE
								FE.assign(tmpFE.begin(),tmpFE.end());
								/*float avefit=0;
								for (int u=0;u<FE.size();u++)
									avefit+=FE[u].fitness;
								avefit /= FE.size();
								if (avefit>1)
									cout <<"avefit error\n";*/
								if(gen>trainer_trigger) { //pick new trainers when FE pop has converged or when it has been enough generations
									if (p.verbosity>0) s.out << "Picking trainers...\n";
									PickTrainers(World.pop,FE,trainers,p,d,s);
									trainer_trigger = gen + p.FE_train_gens; //p.island_gens*(p.popsize+p.popsize*p.eHC_on+p.popsize*p.pHC_on);
								}
							}
						}

						if (migrate)
							T[q].pop.assign(World.pop.begin()+q*subpops,World.pop.begin()+(q+1)*subpops);

						//if (gen>p.g) pass=0;


				} if (p.verbosity>1) s.out << "thread " + std::to_string(static_cast<long long>(q)) + " exited while loop...\n";
				if (!pass){ //make sure archive was updated and generation prints out (may be redundant)
					#pragma omp single
					{
						if(p.prto_arch_on){
							A.update(World.pop);
							if (p.print_archive) printpop(A.pop,p,s,logname,1);
						}
						if (!p.limit_evals || s.totalptevals() >= print_trigger){
							if (p.print_data) printdatafile(World,s,p,r,dfout,gen,time.elapsed());
							if (p.print_every_pop) printpop(World.pop,p,s,logname,2);
							if (p.verbosity > 1) {
								printstats(World,gen,s,p,A);
								s.out << "Total Time: " << (int)floor(time.elapsed()/num_islands/3600) << " hr " << ((int)(time.elapsed()/num_islands) % 3600)/60 << " min " << (int)(time.elapsed()/num_islands) % 60 << " s\n";
								s.out << "Total Evals: " << s.totalevals() << "\n";
								s.out << "Point Evals: " << s.totalptevals() << "\n";
								s.out << "Average evals per second: " << (float)s.totalevals()/time.elapsed() << "\n";
								s.out << "Average point evals per second: " << (float)s.totalptevals()/time.elapsed() << "\n";
							}
							if (print_trigger!=0) print_trigger += p.max_evals/p.num_log_pts;
							if (p.print_genome) printGenome(World,gen,logname,d,p);
							if (p.print_db) printDB(World.pop,logname,d,p);
						}
						++gen;
						if (p.limit_evals) termits = s.totalptevals();
						else ++termits;

						if (!p.EstimateFitness && p.estimate_generality && p.G_shuffle)
							shuffle_data(d,p,r,s);

					}
				}
			} if (p.verbosity>1) s.out << "exited parallel region ...\n";


			if (p.EstimateFitness || p.test_at_end){// assign real fitness values to final population and archive
				p.EstimateFitness=0;
				p.test_at_end = 0;
				Fitness(World.pop,p,d,s,FE[0]);
				if (p.prto_arch_on) Fitness(A.pop,p,d,s,FE[0]);
				p.EstimateFitness=1;
			}

			if (p.print_data) printdatafile(World,s,p,r,dfout,gen,time.elapsed());
			if (p.print_best_ind) printbestind(World,p,s,logname);
			if (p.print_last_pop) printpop(World.pop,p,s,logname,0);

			if (p.prto_arch_on){
				// if (A.pop.empty()) A.update(World.pop);

				if (p.print_archive) printpop(A.pop,p,s,logname,1);
				// save archive to best_prog for python
				if (p.return_pop){
					World.sortpop();
					pop_to_py(World.pop,best_prog);
				}
				else
					pop_to_py(A.pop,best_prog);
			}
			else{
				// save best individual to best_prog for python
				if (p.return_pop){
					World.sortpop();
					pop_to_py(World.pop,best_prog);
				}
				else{
					vector<ind> best(1);
					World.getbestind(best[0]);
					line_to_py(best[0].line,best_prog);
				}
			}
		}
		/////////////////////////////////////////////////////////////////////////////
		// no islands
		//////////////////////////////////////////////////////////////////////////////
		else //no islands
		{
	 		tribe T(p.popsize,p.max_fit,p.min_fit);
			if (p.pop_restart) // initialize population from file
			{
				if (p.verbosity>0) s.out << "loading pop from " + p.pop_restart_path + "...\n";
				int tmp = load_pop(T.pop,p,s);
				if (tmp < p.popsize){
					if (p.verbosity>0) s.out << "WARNING: population size loaded from file (" << tmp << ") is smaller than set pop size (" << p.popsize << "). " << (p.popsize-tmp) << " randomly initiated individuals will be added...\n";
					vector<ind> tmppop(p.popsize-tmp);
					InitPop(tmppop,p,r);
					swap_ranges(T.pop.end()-(p.popsize-tmp),T.pop.end(),tmppop.begin());
				}
				if (p.EstimateFitness)
					InitPopFE(FE,T.pop,trainers,p,r,d,s);
				Fitness(T.pop,p,d,s,FE[0]);
			}
			else{
				if (p.init_validate_on)
				{
					if (p.verbosity>0) s.out << "Initial validation...";
					//bool tmp = p.EstimateFitness;
					//p.EstimateFitness=0;


					float worstfit;
					int cnt=0;
					//float bestfit;
					vector<ind> tmppop;
					// s.out << "Initialize Population..." << "\n";
					InitPop(T.pop,p,r);
					assert (T.pop.size()== p.popsize) ;
					// s.out << "Gen 2 Phen..." << "\n";
					// s.out << "Fitness..." << "\n";
					if (p.EstimateFitness)
						InitPopFE(FE,T.pop,trainers,p,r,d,s);

					Fitness(T.pop,p,d,s,FE[0]);
					assert (T.pop.size()== p.popsize) ;

					worstfit = T.worstFit();
					while(worstfit == p.max_fit && cnt<100)
					{
						for (vector<ind>::iterator j=T.pop.begin();j!=T.pop.end();)
						{
							if ( (*j).fitness == p.max_fit)
							{
								j=T.pop.erase(j);
								tmppop.push_back(ind());
							}
							else
								++j;
						}

						if (p.verbosity>0) s.out << "\ntmppop size: " << tmppop.size();
						InitPop(tmppop,p,r);
						Fitness(tmppop,p,d,s,FE[0]);

						T.pop.insert(T.pop.end(),tmppop.begin(),tmppop.end());
						tmppop.clear();
						worstfit = T.worstFit();
						cnt++;
						if(cnt==100)
							if (p.verbosity>0) s.out << "initial population count exceeded. Starting evolution...\n";
					}
					//p.EstimateFitness=tmp;
				}
				else // normal population initialization
				{
					InitPop(T.pop,p,r);
					if (p.EstimateFitness)
						InitPopFE(FE,T.pop,trainers,p,r,d,s);
					//bool tmp = p.EstimateFitness;
					//p.EstimateFitness=0;
					Fitness(T.pop,p,d,s,FE[0]);
					//p.EstimateFitness=tmp;
				}
			}
			s.setgenevals();
			if (p.verbosity>0) s.out << " number of evals: " << s.getgenevals() << "\n";
			int its = 1;
			long long termits;
			int gits=1;

			if (p.limit_evals) termits = s.totalptevals();
			else termits=1;
			int trigger=0;
			int trainer_trigger=p.FE_train_gens;//*(p.popsize+p.popsize*p.eHC_on+p.popsize*p.pHC_on);

			int gen=0;
			int counter=0;
			//if(p.sel==2) // if using deterministic crowding, increase gen size
			//{
			//	gen = p.g*(p.popsize*(p.rt_mut+p.rt_rep) + p.popsize*p.rt_cross/2);
			//	trigger = p.popsize*(p.rt_mut+p.rt_rep)+p.popsize*p.rt_cross/2;
			//}
			//else
				gen=p.g;
			long long term, print_trigger;
			bool printed=false;

			if (p.limit_evals) {
				term = p.max_evals;
				if (p.num_log_pts ==0) print_trigger=0;
				else print_trigger = p.max_evals/p.num_log_pts;
			}
			else{
				print_trigger=0;
				term = gen;
			}


			float etmp;
			//print initial population
			if (p.print_init_pop) printpop(T.pop,p,s,logname,3);
			if (p.print_genome) printGenome(T,0,logname,d,p);
			if (p.print_db) printDB(T.pop,logname,d,p);

			while (termits<=term && !stopcondition(T,p,d,s,FE[0]))
			{

				 etmp = s.numevals[omp_get_thread_num()];

				 if (!p.EstimateFitness && p.estimate_generality && p.G_shuffle)
					shuffle_data(d,p,r,s);

				 assert (T.pop.size() == p.popsize);

				 try{
				 	Generation(T.pop,p,r,d,s,FE[0]);
				}
				catch(std::exception& e){
					std::cerr << e.what() << std::endl;
				}
				catch(...){
					std::cerr << "not a standard error\n";
				}
				 assert (T.pop.size()== p.popsize);
				 //s.out << "Generation evals = " + to_string(static_cast<long long>(s.numevals[omp_get_thread_num()]-etmp)) + "\n";


				 if (its>trigger)
				 {
					 if (p.pHC_on && p.ERC)
					 {
						 etmp = s.numevals[omp_get_thread_num()];
						//#pragma omp parallel for
			 			for(int k=0; k<T.pop.size(); ++k)
			 				HillClimb(T.pop.at(k),p,r,d,s,FE[0]);
			 			 //s.out << "Hill climb evals = " + to_string(static_cast<long long>(s.numevals[omp_get_thread_num()]-etmp)) + "\n";


			 		 }

					 if (p.eHC_on&& !p.eHC_mut)
					 {
						 etmp = s.numevals[omp_get_thread_num()];
						// boost::progress_timer tm1;
						//#pragma omp parallel for
						for(int m=0; m<T.pop.size(); m++)
							EpiHC(T.pop.at(m),p,r,d,s,FE[0]);
						 //s.out << "EHC evals = " + to_string(static_cast<long long>(s.numevals[omp_get_thread_num()]-etmp)) + "\n";
					 }
                     if (p.SGD && p.ERC)
                    {
                        for(int k=0; k<T.pop.size(); ++k)
                                StochasticGradient(T.pop.at(k),p,r,d,s,FE[0],gen);

                    }
					s.setgenevals();
					//s.out << "Elapsed time: \n";
					if (p.prto_arch_on){
						A.update(T.pop);
						if (p.print_archive) printpop(A.pop,p,s,logname,1);
					}
					if (!p.limit_evals || s.totalptevals() >= print_trigger){
						if (p.print_data) printdatafile(T,s,p,r,dfout,counter,time.elapsed());
						if (p.print_every_pop) printpop(T.pop,p,s,logname,2);
						if (p.verbosity > 1){
							printstats(T,counter,s,p,A);
							s.out << "Total Time: " << (int)floor(time.elapsed()/3600) << " hr " << ((int)time.elapsed() % 3600)/60 << " min " << (int)time.elapsed() % 60 << " s\n";
							s.out << "Total Evals: " << s.totalevals() << "\n";
							s.out << "Point Evals: " << s.totalptevals() << "\n";
							s.out << "Average evals per second: " << (float)s.totalevals()/time.elapsed() << "\n";
							s.out << "Average point evals per second: " << (float)s.totalptevals()/time.elapsed() << "\n";
						}
						if (print_trigger!=0) print_trigger += p.max_evals/p.num_log_pts;
						if (p.print_genome) printGenome(T,counter,logname,d,p);
						if (p.print_db) printDB(T.pop,logname,d,p);

						printed=true;
					}

					//if (p.sel==2)
						//trigger+=p.popsize*(p.rt_mut+p.rt_rep)+p.popsize*p.rt_cross/2;
					++counter;

				 }

				if (p.EstimateFitness){
					EvolveFE(T.pop,FE,trainers,p,d,s,r);
					if (!p.limit_evals || printed){
						if (p.verbosity>0) s.out << "Evolving fitness estimators...\n";
						if (p.verbosity>0) s.out << "Best FE fit: " << FE[0].fitness <<"\n";
						if (p.estimate_generality) s.out << "Best FE genty: " << FE[0].genty <<"\n";
						if (p.verbosity>0) s.out << "Current Fitness Estimator:\n";

						if (p.verbosity>0) {
							for (int b=0;b<FE[0].FEpts.size();b++)
								s.out << FE[0].FEpts[b] << " ";
							s.out << "\n";
						}
					}
				}


				if (p.EstimateFitness){
					if(counter>trainer_trigger) {
						if (p.verbosity>0) s.out << "Picking trainers...\n";
						PickTrainers(T.pop,FE,trainers,p,d,s);
						trainer_trigger= counter + p.FE_train_gens; //*(p.popsize+p.popsize*p.eHC_on+p.popsize*p.pHC_on);
						//trainer_trigger=0;
					}
				}


				if (p.limit_evals) termits = s.totalptevals();
				else termits++;
				its++;
				printed=false;
			}


			if (p.EstimateFitness || p.test_at_end){// assign real fitness values to final population and archive
				p.EstimateFitness=0;
				p.test_at_end = 0;
				Fitness(T.pop,p,d,s,FE[0]);
				if (p.prto_arch_on) Fitness(A.pop,p,d,s,FE[0]);
				p.EstimateFitness=1;
			}

				if (p.print_data) printdatafile(T,s,p,r,dfout,counter,time.elapsed());
				if (p.print_best_ind) printbestind(T,p,s,logname);
				if (p.print_last_pop) printpop(T.pop,p,s,logname,0);


			if (p.prto_arch_on){
				if (A.pop.empty()) A.update(T.pop);
				if (p.print_archive) printpop(A.pop,p,s,logname,1);
				// save archive to best_prog for python
				if (p.return_pop){
					T.sortpop();
					pop_to_py(T.pop,best_prog);
				}
				else
					pop_to_py(A.pop,best_prog);
			}
			else{
				// save best individual to best_prog for python

				if (p.return_pop){
					T.sortpop();
					pop_to_py(T.pop,best_prog);
				}
				else{
					vector<ind> best(1);
					T.getbestind(best[0]);
					line_to_py(best[0].line,best_prog);
				}
			}
		}


		if (p.verbosity>0) s.out << "\n Program finished sucessfully.\n";

// Py_END_ALLOW_THREADS	} //end python gil unlock
	}
	catch(std::exception& e){
		std::cerr << e.what();

	}

}

BOOST_PYTHON_MODULE(elgp)
{
    def("runEllenGP", runEllenGP);
		// import_array();
}
