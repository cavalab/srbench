// header file for ind struct
#pragma once
#ifndef PARAMS_H
#define PARAMS_H
//#include <iostream>
//#include <string>
//#include <vector>
//#include <random>
//#include <array>
#include "op_node.h"
//#include "data.h"
#include "Eqn2Line.h"
using namespace std;
#include <boost/python.hpp>
using namespace boost::python;
#if defined(_WIN32)
	#include <direct.h>
	#define GetCurrentDir _getcwd
#else
	#include <unistd.h>
	#include <iomanip>
	#define GetCurrentDir getcwd
#endif
// using std::vector;
// using std::begin;
// using std::string;
// using std::cout;
// BOOST_PYTHON_MODULE(params)
// {
//     class_<params>("params")
//         .def("set", &World::set)
//     ;
// }

struct params {

	int g; // number of generations (limited by default)
	int popsize; //population size
	bool limit_evals; // limit evals instead of generations
	long long max_evals; // maximum number of evals before termination (only active if limit_evals is true)

	// Generation Settings
	int sel; // 1: tournament 2: deterministic crowding 3: lexicase selection 4: age-fitness pareto algorithm
	int tourn_size;
	float rt_rep; //probability of reproduction
	float rt_cross;
	float rt_mut;
	vector<float> rep_wheel;
	float cross_ar; //crossover alternation rate
	float mut_ar;
	int cross; // 1: ultra; 2: one point
	int mutate; // 1: uniform point mutation; 2: sub-tree mutation
	bool align_dev; // on or off - adds alignment deviation via gaussian random variable
	bool elitism; // on or off - if on, saves best individual each generation
	//float stoperror; // stop condition / convergence condition

	// Data settings
	bool init_validate_on; // initial fitness validation of individuals
	bool train; // choice to turn on training for splitting up the data set
	float train_pct; // percent of data to use for training (validation pct = 1-train_pct)
	bool shuffle_data; // shuffle the data
	bool test_at_end; // only run the test fitness on the population at the end of a run

	bool pop_restart; // restart from previous population
	string pop_restart_path; // restart population file path
	// Results
	string resultspath;
	string savename; // savename for files
	//bool loud;

	// use fitness estimator coevolution
	bool EstimateFitness;
	int FE_pop_size;
	int FE_ind_size;
	int FE_train_size;
	int FE_train_gens;
	bool FE_rank;

	bool estimate_generality;
	int G_sel;
	bool G_shuffle;

	// Problem information
	vector <string> intvars; // internal variables
	//vector <string> extvars; // external variables (external forces)
	vector <string> cons;
	vector <float> cvals;
	vector <string> seeds;
	vector <vector <node> > seedstacks;
	bool AR; //auto-regressive output
	int AR_na; //order of auto-regressive output
	int AR_nka; // state delay
	int AR_nb; //order of inputs
	int AR_nkb; // input delay
	bool AR_lookahead;
	vector <string> allvars;// = intvars.insert(intvars.end(), extvars.begin(), extvars.end());
	vector <string> allblocks;// = allvars.insert(allvars.end(),consvals.begin(),convals.end());
	//allblocks = allblocks.insert(allblocks.end(),seeds.begin(),seeds.end());

	bool init_trees; // do tree-like recursive initialization of genotypes

	bool ERC; // ephemeral random constants
	bool ERCints;
	int maxERC;
	int minERC;
	int numERC;

	//vector <float> target;
	// Fitness Settings

	string fit_type; // MAE(1), R2(2), MAER2(3), VAF(4), MSE
	bool norm_error; // normalize fitness by the standard deviation of the target output
	float max_fit;
	float min_fit;
	bool weight_error; // weight error vector by predefined weights from data file
	vector<float> error_weight; // vector of error weights
	vector<float> class_w; // vector of class weights (proportion of training samples with class n)
	vector<float> class_w_v; // vector of class weights (proportion of test samples with class n)
	// Operator Settings
	vector <string> op_list;
	vector <int> op_choice; // map op list to pointer location in makeline() pointer function
	vector <float> op_weight;
	vector <int> op_arity; // arity list corresponding op_choice for recursive makeline
	vector <char> return_type; // return type of operators for tree construction
	bool weight_ops_on;

	int min_len;
	int max_len;
	int max_len_init; // initial max len

	int complex_measure; // currently not used


	// Hill Climbing Settings

		// generic line hill climber (Bongard)
	bool lineHC_on;
	int lineHC_its;

		// parameter Hill Climber
	bool pHC_on;
	bool pHC_delay_on;
	float pHC_delay;
	int pHC_size;
	int pHC_its;
	float pHC_gauss;

		// epigenetic Hill Climber
	bool eHC_on;
	int eHC_its;
	float eHC_prob;
	float eHC_init;
	bool eHC_mut; // epigenetic mutation rather than hill climbing
	bool eHC_slim; // use SlimFitness
        // stochastic gradient descent
    bool SGD;
    float learning_rate;
     
    // Pareto settings

	bool prto_arch_on;
	int prto_arch_size;
	bool prto_sel_on;

	//island model
	bool islands;
	int num_islands;
	int island_gens;
	int nt; // number of threads

	//int seed;

	// lexicase selection
	float lexpool; // percent of population to use for lexicase selection events
	bool lexage;// use afp survival after lexicase selection
	bool lex_class; // use class-based fitness objectives instead of raw error
	vector<string> lex_metacases; // extra cases to be used for lexicase selection
	bool lex_eps_error; // errors within episilon of the best error are pass, otherwise fail
	bool lex_eps_target; // errors within epsilon of the target are pass, otherwise fail
	bool lex_eps_std; // errors in a standard dev of the best are pass, otherwise fail
	bool lex_eps_target_mad; // errors in a standard dev of the best are pass, otherwise fail
	bool lex_eps_error_mad; // errors in a standard dev of the best are pass, otherwise fail
	bool lex_eps_global; // pass condition defined relative to whole population (rather than selection pool)
	bool lex_eps_dynamic; // epsilon is defined relative to the pool instead of globally
	bool lex_eps_dynamic_rand; /* epsilon is defined as a random threshold
	corresponding to an error in the pool minus min error in pool*/
	bool lex_eps_dynamic_madcap; // with prob of 0.5, epsilon is replaced with 0
	float lex_epsilon;

	// ==== Printing Options === //
	//print initial population
	bool print_init_pop;
	bool print_last_pop;
	bool print_homology; // print homology
	bool print_log; //print log
	bool print_data;
	bool print_best_ind;
	bool print_archive;
	bool print_every_pop; // print pop every generation
	bool print_genome; // print genome every generation
	bool print_epigenome; // print epigenome every generation
	bool print_novelty; // print novelty in data file
	bool print_db; // print individuals for graph database analysis
	int verbosity; // setting for what gets printed 0: nothing 1: some stuff 2: all stuff

	// number of log points to print (with eval limitation)
	int num_log_pts;
	//pareto survival setting
	int PS_sel;

	//classification
	bool classification;
	bool class_bool; // use binary or multiclass
	bool class_m4gp; // use m3gp fitness
	bool class_prune; // prune dimensions of best individual each generation
	int number_of_classes; // number of unique classes

	// stop condition
	bool stop_condition;
	float stop_threshold;
	//print protected operators
	bool print_protected_operators;
	// return population to python
	bool return_pop;

	params(){ //default values
		g=100; // number of generations (limited by default)
		popsize=500; //population size
		limit_evals=false; // limit evals instead of generations
		max_evals=0; // maximum number of evals before termination (only active if limit_evals is true)
		init_trees=1;
		// Generation Settings
		sel=1; // 1: tournament 2: deterministic crowding 3: lexicase selection 4: age-fitness pareto algorithm
		tourn_size=2;
		rt_rep=0; //probability of reproduction
		rt_cross=0.6;
		rt_mut=0.4;
		cross_ar=0.025; //crossover alternation rate
		mut_ar=0.025;
		cross=3; // 1: ultra; 2: one point; 3: subtree
		mutate=1; // 1: one point; 2: subtree
		align_dev = 0;
		elitism = 0;

		// ===============   Data settings
		init_validate_on=0; // initial fitness validation of individuals
		train=0; // choice to turn on training for splitting up the data set
		train_pct=0.5; // default split of data is 50/50
		shuffle_data=0; // shuffle the data
		test_at_end = 0; // only run the test fitness on the population at the end of a run
		pop_restart = 0; // restart from previous population
		pop_restart_path=""; // restart population file path
		AR = 0;
		AR_nb = 1;
		AR_nkb = 0;
		AR_na = 1;
		AR_nka = 1;
		AR_lookahead = 0;
		// ================ Results and printing
		char cCurrentPath[FILENAME_MAX];
		bool tmp = GetCurrentDir(cCurrentPath, sizeof(cCurrentPath));
		resultspath= (std::string) cCurrentPath;
		//savename
		savename="";
		//print every population
		print_every_pop=0;
		//print initial population
		print_init_pop = 0;
		//print last population
		print_last_pop = 0;
		// print homology
		print_homology = 0;
		//print log
		print_log = 0;
		//print data
		print_data = 0;
		//print best individual at end
		print_best_ind = 0;
		//print archive
		print_archive = 0;
		// number of log points to print (with eval limitation)
		num_log_pts = 0;
		// print csv files of genome each print cycle
		print_genome = 0;
		// print csv files of epigenome each print cycle
		print_epigenome = 0;
		// print number of unique output vectors
		print_novelty = 0;
		// print individuals for graph database analysis
		print_db = 0;
		// verbosity
		verbosity = 0;

		// ============ Fitness settings
		fit_type = "MSE"; // 1: error, 2: corr, 3: combo
		norm_error = 0 ; // normalize fitness by the standard deviation of the target output
		weight_error = 0; // weight error vector by predefined weights from data file
		max_fit = 1.0E20;
		min_fit = 0.00000000000000000001;

		// Fitness estimators
		EstimateFitness=0;
		FE_pop_size=0;
		FE_ind_size=0;
		FE_train_size=0;
		FE_train_gens=0;
		FE_rank=0;
		estimate_generality=0;
		G_sel=1;
		G_shuffle=0;
		// Computer Settings
		//bool parallel;
		//int numcores;


		// =========== Program Settings
		ERC = 1; // ephemeral random constants
		ERCints =0 ;
		maxERC = 1;
		minERC = -1;
		numERC = 1;

		min_len = 3;
		max_len = 20;
		max_len_init = 0;

		complex_measure=1; // 1: genotype size; 2: symbolic size; 3: effective genotype size

		weight_ops_on=0;

		// Hill Climbing Settings

			// generic line hill climber (Bongard)
		lineHC_on = 0;
		lineHC_its = 0;

			// parameter Hill Climber
		pHC_on = 0;
		//pHC_size;
		pHC_its = 1;
		pHC_gauss = 0;

			// epigenetic Hill Climber
		eHC_on = 0;
		eHC_its = 1;
		eHC_prob = 0.1;
		eHC_init = 0.5;
		eHC_mut = 0; // epigenetic mutation rather than hill climbing
		eHC_slim = 0; // use SlimFitness

            // stochastic gradient descent
        SGD = false;
        learning_rate = 1.0;
		// Pareto settings

		prto_arch_on = 0;
		prto_arch_size = 1;
		prto_sel_on = 0;

		//island model
		islands = 0;
		num_islands=0;
		island_gens = 100;
		nt = 1;

		//seed = 0;

		// lexicase selection
		lexpool = 1;
		lexage = 0;
		lex_class = 0;
		lex_eps_error = false; // errors within episilon of the best error are pass, otherwise fail
		lex_eps_target = false; // errors within epsilon of the target are pass, otherwise fail
		lex_eps_std = false; // errors in a standard dev of the best are pass, otherwise fail
		lex_eps_target_mad=false; // errors in a standard dev of the best are pass, otherwise fail
		lex_eps_error_mad=false; // errors in a standard dev of the best are pass, otherwise fail
		lex_epsilon = 0.1;
		lex_eps_global = false; //pass conditions in lex eps defined relative to whole population (rather than selection pool).
		lex_eps_dynamic = false;
		lex_eps_dynamic_rand = false;
		lex_eps_dynamic_madcap = false;
		                       //should be true for regular lexicase (forced in load_params)
		//pareto survival setting
		PS_sel=1;

		// classification
		classification = 0;
		class_bool = 0;
		class_m4gp = 0;
		class_prune = 0;
		//class_multiclass=0; // use multiclass
		number_of_classes=1; //for use with multiclass

		stop_condition=1;
		stop_threshold = 0.000001;
		print_protected_operators = 0;

		// return population to python
		return_pop = false;
	}
	~params(){}




	//void clear()
	//{
	//
	//	rep_wheel.clear();
	//
	//	// Problem information
	//	intvars.clear(); // internal variables
	//	extvars.clear(); // external variables (external forces)
	//	cons.clear();
	//	cvals.clear();
	//	seeds.clear();
	//
	//	allvars.clear();// = intvars.insert(intvars.end(), extvars.begin(), extvars.end());
	//	allblocks.clear();// = allvars.insert(allvars.end(),consvals.begin(),convals.end());
	//	//allblocks = allblocks.insert(allblocks.end(),seeds.begin(),seeds.end());

	//	op_list.clear();
	//	op_choice.clear(); // map op list to pointer location in makeline() pointer function
	//	op_weight.clear();
	//
	//}
	void set(dict& d){
		/* function called from python to set parameter values. equivalent behavior to load_params,
		but for setting params in python.*/
		// std::cout << "in set params\n";
		if (d.has_key("g"))
			g = extract<int>(d["g"]);
		if (d.has_key("popsize"))
			popsize = extract<int>(d["popsize"]);
		if (d.has_key("sel"))
			sel = extract<int>(d["sel"]);
		if (d.has_key("tourn_size"))
			tourn_size = extract<int>(d["tourn_size"]);
		if (d.has_key("rt_rep"))
			rt_rep = extract<float>(d["rt_rep"]);
		if (d.has_key("rt_cross"))
			rt_cross = extract<float>(d["rt_cross"]);
		if (d.has_key("rt_mut"))
			rt_mut = extract<float>(d["rt_mut"]);
		if (d.has_key("cross"))
			cross = extract<int>(d["cross"]);
		if (d.has_key("cross_ar"))
			cross_ar = extract<float>(d["cross_ar"]);
		if (d.has_key("mut_ar"))
			mut_ar = extract<float>(d["mut_ar"]);
		if (d.has_key("init_validate_on"))
			init_validate_on = extract<bool>(d["init_validate_on"]);
		if (d.has_key("resultspath"))
		{
			resultspath = extract<string>(d["resultspath"]);
			// cout << "resultspath:" << resultspath << "\n";

		}
		if (d.has_key("savename"))
		{
			savename = extract<string>(d["savename"]);

		}
		if (d.has_key("intvars"))
		{
			for(unsigned int i = 0; i<len(d["intvars"]); ++i)
				intvars.push_back(extract<string>(d["intvars"][i]));
		}

		if (d.has_key("cvals"))
		{
			for(unsigned int i = 0; i<len(d["cvals"]); ++i) {
				cvals.push_back(extract<float>(d["cvals"][i]));
				cons.push_back(std::to_string(static_cast<long double>(extract<float>(d["cvals"][i]))));
			}

		}
		if (d.has_key("seeds"))
		{
			for(unsigned int i = 0; i<len(d["seeds"]); ++i)
				seeds.push_back(extract<string>(d["seeds"][i]));
		}
		if (d.has_key("ERC"))
			ERC = extract<bool>(d["ERC"]);
		if (d.has_key("ERCints"))
			ERCints = extract<bool>(d["ERCints"]);
		if (d.has_key("maxERC"))
			maxERC = extract<int>(d["maxERC"]);
		if (d.has_key("minERC"))
			minERC = extract<int>(d["minERC"]);
		if (d.has_key("numERC"))
			numERC = extract<int>(d["numERC"]);
		if (d.has_key("fit_type"))
			fit_type = extract<string>(d["fit_type"]);
		if (d.has_key("max_fit"))
			max_fit = extract<float>(d["max_fit"]);
		if (d.has_key("min_fit"))
			min_fit = extract<float>(d["min_fit"]);
		if (d.has_key("op_list"))
		{
			// cout << "op_list: ";
			for(unsigned int i = 0; i<len(d["op_list"]); ++i){
				// string tmp = extract<string>(d["op_list"][i]);
				// cout << tmp << ",";
		   		op_list.push_back(extract<string>(d["op_list"][i]));
			}
			// cout << "\n";
		}
		if (d.has_key("op_weight"))
		{
			// cout << "op_weight: ";
			for(unsigned int i = 0; i<len(d["op_weight"]); ++i){
				// float tmp = extract<float>(d["op_weight"][i]);
				// cout << tmp << ",";
		   		op_weight.push_back(extract<float>(d["op_weight"][i]));
			}
			// cout << "\n";
		}
		if (d.has_key("weight_ops_on"))
			weight_ops_on = extract<bool>(d["weight_ops_on"]);
		if (d.has_key("min_len"))
			min_len = extract<int>(d["min_len"]);
		if (d.has_key("max_len"))
			max_len = extract<int>(d["max_len"]);
		if (d.has_key("max_len_init"))
			max_len_init = extract<int>(d["max_len_init"]);
		if (d.has_key("complex_measure"))
			complex_measure = extract<int>(d["complex_measure"]);
		if (d.has_key("lineHC_on"))
			lineHC_on = extract<bool>(d["lineHC_on"]);
		if (d.has_key("lineHC_its"))
			lineHC_its = extract<int>(d["lineHC_its"]);
		if (d.has_key("pHC_on"))
			pHC_on = extract<bool>(d["pHC_on"]);
		if (d.has_key("pHC_delay_on"))
			pHC_delay_on = extract<bool>(d["pHC_delay_on"]);
		if (d.has_key("pHC_size"))
			pHC_size = extract<int>(d["pHC_size"]);
		if (d.has_key("pHC_its"))
			pHC_its = extract<int>(d["pHC_its"]);
		if (d.has_key("pHC_gauss"))
			pHC_gauss = extract<float>(d["pHC_gauss"]);
		if (d.has_key("eHC_on"))
			eHC_on = extract<bool>(d["eHC_on"]);
		if (d.has_key("eHC_its"))
			eHC_its = extract<int>(d["eHC_its"]);
		if (d.has_key("eHC_prob"))
			eHC_prob = extract<float>(d["eHC_prob"]);
		if (d.has_key("eHC_init"))
			eHC_init = extract<float>(d["eHC_init"]);
		if (d.has_key("eHC_mut"))
			eHC_mut = extract<bool>(d["eHC_mut"]);
		if (d.has_key("eHC_slim"))
			eHC_slim = extract<bool>(d["eHC_slim"]);
		if (d.has_key("lexpool"))
			lexpool = extract<float>(d["lexpool"]);
		if (d.has_key("prto_arch_on"))
			prto_arch_on = extract<bool>(d["prto_arch_on"]);
		if (d.has_key("prto_arch_size"))
			prto_arch_size = extract<int>(d["prto_arch_size"]);
		if (d.has_key("prto_sel_on"))
			prto_sel_on = extract<bool>(d["prto_sel_on"]);
		if (d.has_key("islands"))
			islands = extract<bool>(d["islands"]);
		if (d.has_key("num_islands"))
			num_islands = extract<int>(d["num_islands"]);
		if (d.has_key("island_gens"))
			island_gens = extract<int>(d["island_gens"]);
		if (d.has_key("train"))
			train = extract<bool>(d["train"]);
		if (d.has_key("train_pct"))
			train_pct = extract<float>(d["train_pct"]);
		if (d.has_key("print_every_pop"))
			print_every_pop = extract<bool>(d["print_every_pop"]);
		if (d.has_key("estimate_fitness"))
			EstimateFitness = extract<bool>(d["estimate_fitness"]);
		if (d.has_key("FE_pop_size"))
			FE_pop_size = extract<int>(d["FE_pop_size"]);
		if (d.has_key("FE_ind_size"))
			FE_ind_size = extract<int>(d["FE_ind_size"]);
		if (d.has_key("FE_train_size"))
			FE_train_size = extract<int>(d["FE_train_size"]);
		if (d.has_key("FE_train_gens"))
			FE_train_gens = extract<int>(d["FE_train_gens"]);
		if (d.has_key("FE_rank"))
			FE_rank = extract<bool>(d["FE_rank"]);
		if (d.has_key("estimate_generality"))
			estimate_generality = extract<bool>(d["estimate_generality"]);
		if (d.has_key("G_sel"))
			G_sel = extract<int>(d["G_sel"]);
		if (d.has_key("G_shuffle"))
			G_shuffle = extract<bool>(d["G_shuffle"]);
		if (d.has_key("norm_error"))
			norm_error = extract<bool>(d["norm_error"]);
		if (d.has_key("shuffle_data"))
			shuffle_data = extract<bool>(d["shuffle_data"]);
		if (d.has_key("init_trees"))
			init_trees = extract<bool>(d["init_trees"]);
		if (d.has_key("limit_evals"))
			limit_evals = extract<bool>(d["limit_evals"]);
		if (d.has_key("max_evals"))
			max_evals = extract<long long>(d["max_evals"]);
		if (d.has_key("print_homology"))
			print_homology = extract<bool>(d["print_homology"]);
		if (d.has_key("print_log"))
			print_log = extract<bool>(d["print_log"]);
		if (d.has_key("print_data"))
			print_data = extract<bool>(d["print_data"]);
		if (d.has_key("print_best_ind"))
			print_data = extract<bool>(d["print_best_ind"]);
		if (d.has_key("print_archive"))
			print_archive = extract<bool>(d["print_archive"]);
		if (d.has_key("print_init_pop"))
			print_init_pop = extract<bool>(d["print_init_pop"]);
		if (d.has_key("print_genome"))
			print_genome = extract<bool>(d["print_genome"]);
		if (d.has_key("print_db"))
			print_db = extract<bool>(d["print_db"]);
		if (d.has_key("print_epigenome"))
			print_epigenome = extract<bool>(d["print_epigenome"]);
		if (d.has_key("num_log_pts"))
			num_log_pts = extract<int>(d["num_log_pts"]);
		if (d.has_key("PS_sel"))
			PS_sel = extract<bool>(d["PS_sel"]);
		if (d.has_key("pop_restart"))
			pop_restart = extract<bool>(d["pop_restart"]);
		if (d.has_key("pop_restart_path"))
			pop_restart_path = extract<string>(d["pop_restart_path"]);
		if (d.has_key("AR"))
			AR = extract<bool>(d["AR"]);
		if (d.has_key("AR_na"))
			AR_na = extract<int>(d["AR_na"]);
		if (d.has_key("AR_nka")) {
			AR_nka = extract<int>(d["AR_nka"]);
			if (AR_nka < 1) {
				cout << "WARNING: AR_nka set to min value of 1\n";
				AR_nka = 1;
			}
		}
		if(d.has_key("AR_nb"))
			AR_nb = extract<int>(d["AR_nb"]);
		if (d.has_key("AR_nkb"))
			AR_nkb;
		if(d.has_key("AR_lookahead"))
			AR_lookahead = extract<bool>(d["AR_lookahead"]);
		if(d.has_key("align_dev"))
			align_dev = extract<bool>(d["align_dev"]);
		if(d.has_key("classification"))
			classification = extract<bool>(d["classification"]);
		if(d.has_key("class_bool"))
			class_bool = extract<bool>(d["class_bool"]);
		if(d.has_key("class_m4gp"))
			class_m4gp = extract<bool>(d["class_m4gp"]);
		if(d.has_key("class_prune"))
			class_prune = extract<bool>(d["class_prune"]);
		if(d.has_key("number_of_classes"))
			number_of_classes = extract<int>(d["number_of_classes"]);
		if(d.has_key("elitism"))
			elitism = extract<bool>(d["elitism"]);
		if(d.has_key("stop_condition"))
			stop_condition = extract<bool>(d["stop_condition"]);
		if(d.has_key("stop_threshold"))
			stop_threshold = extract<float>(d["stop_threshold"]);
		if(d.has_key("mutate"))
			mutate = extract<int>(d["mutate"]);
		if(d.has_key("print_novelty"))
			print_novelty = extract<bool>(d["print_novelty"]);
		if(d.has_key("lex_metacases"))
		{
			for(unsigned int i = 0; i<len(d["lex_metacases"]); ++i)
		    	lex_metacases.push_back(extract<string>(d["lex_metacases"][i]));
		}
		if(d.has_key("lex_class"))
			lex_class = extract<bool>(d["lex_class"]);
		if(d.has_key("weight_error"))
			weight_error = extract<bool>(d["weight_error"]);
		if(d.has_key("print_protected_operators"))
			print_protected_operators = extract<bool>(d["print_protected_operators"]);
		if (d.has_key("lex_eps_error"))
			lex_eps_error = extract<bool>(d["lex_eps_error"]);
		if (d.has_key("lex_eps_target"))
			lex_eps_target = extract<bool>(d["lex_eps_target"]);
		if (d.has_key("lex_eps_std"))
			lex_eps_std = extract<bool>(d["lex_eps_std"]);
		if (d.has_key("lex_eps_target_mad"))
			lex_eps_target_mad = extract<bool>(d["lex_eps_target_mad"]);
		if (d.has_key("lex_eps_error_mad"))
			lex_eps_error_mad = extract<bool>(d["lex_eps_error_mad"]);
		if (d.has_key("lex_epsilon"))
			lex_epsilon = extract<bool>(d["lex_epsilon"]);
		if (d.has_key("lex_eps_global"))
			lex_eps_global = extract<bool>(d["lex_eps_global"]);
		if (d.has_key("lex_eps_dynamic"))
			lex_eps_dynamic = extract<bool>(d["lex_eps_dynamic"]);
		if (d.has_key("lex_eps_dynamic_rand"))
				lex_eps_dynamic_rand = extract<bool>(d["lex_eps_dynamic_rand"]);
		if (d.has_key("lex_eps_dynamic_madcap"))
				lex_eps_dynamic_madcap = extract<bool>(d["lex_eps_dynamic_madcap"]);
		if (d.has_key("test_at_end"))
			test_at_end = extract<bool>(d["test_at_end"]);
		if (d.has_key("verbosity"))
			verbosity = extract<int>(d["verbosity"]);
		if (d.has_key("return_pop"))
			return_pop = extract<bool>(d["return_pop"]);
		if (d.has_key("lexage"))
			lexage = extract<bool>(d["lexage"]);
        if (d.has_key("SGD"))
            SGD = extract<bool>(d["SGD"]);
        if (d.has_key("learning_rate"))
            learning_rate = extract<float>(d["learning_rate"]);
		// finished reading from dict.

		allvars = intvars;
		//allvars.insert(allvars.end(), extvars.begin(), extvars.end());
		allblocks = allvars;
		allblocks.insert(allblocks.end(),cons.begin(),cons.end());
		allblocks.insert(allblocks.end(),seeds.begin(),seeds.end());

		//seed = time(0);


		if (max_len_init == 0)
			max_len_init = max_len;
		// op_list
		if (op_list.empty()){ // set default operator list
			// cout << "using default operator list\n";
			op_list.push_back("n");
			op_list.push_back("v");
			op_list.push_back("+");
			op_list.push_back("-");
			op_list.push_back("*");
			op_list.push_back("/");
		}
		for (unsigned int i=0; i<op_list.size(); ++i)
		{
			if (op_list.at(i).compare("n")==0 )//&& ( ERC || !cvals.empty() ) )
			{
				op_choice.push_back(0);
				op_arity.push_back(0);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("v")==0)
			{
				op_choice.push_back(1);
				op_arity.push_back(0);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("+")==0)
			{
				op_choice.push_back(2);
				op_arity.push_back(2);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("-")==0)
			{
				op_choice.push_back(3);
				op_arity.push_back(2);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("*")==0)
			{
				op_choice.push_back(4);
				op_arity.push_back(2);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("/")==0)
			{
				op_choice.push_back(5);
				op_arity.push_back(2);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("sin")==0)
			{
				op_choice.push_back(6);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("cos")==0)
			{
				op_choice.push_back(7);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("exp")==0)
			{
				op_choice.push_back(8);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("log")==0)
			{
				op_choice.push_back(9);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("sqrt")==0)
			{
				op_choice.push_back(11);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("2")==0)
			{
				op_choice.push_back(12);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("3")==0)
			{
				op_choice.push_back(13);
				op_arity.push_back(1);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("^") == 0)
			{
				op_choice.push_back(14);
				op_arity.push_back(2);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("=")==0)
			{
				op_choice.push_back(15);
				op_arity.push_back(2);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare("!")==0)
			{
				op_choice.push_back(16);
				op_arity.push_back(2);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare("<")==0)
			{
				op_choice.push_back(17);
				op_arity.push_back(2);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare(">")==0)
			{
				op_choice.push_back(18);
				op_arity.push_back(2);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare("<=")==0)
			{
				op_choice.push_back(19);
				op_arity.push_back(2);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare(">=")==0)
			{
				op_choice.push_back(20);
				op_arity.push_back(2);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare("if-then")==0)
			{
				op_choice.push_back(21);
				op_arity.push_back(2);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("if-then-else")==0)
			{
				op_choice.push_back(22);
				op_arity.push_back(3);
				return_type.push_back('f');
			}
			else if (op_list.at(i).compare("&")==0)
			{
				op_choice.push_back(23);
				op_arity.push_back(3);
				return_type.push_back('b');
			}
			else if (op_list.at(i).compare("|")==0)
			{
				op_choice.push_back(24);
				op_arity.push_back(3);
				return_type.push_back('b');
			}
			else
				cout << "bad command (load params op_choice)" << "\n";
		}


		rep_wheel.push_back(rt_rep);
		rep_wheel.push_back(rt_cross);
		rep_wheel.push_back(rt_mut);

		partial_sum(rep_wheel.begin(), rep_wheel.end(), rep_wheel.begin());

		if(!seeds.empty()) // get seed stacks
		{
			op_choice.push_back(10); // include seeds in operation choices
			op_weight.push_back(1); // include opweight if used
			op_list.push_back("seed");
			op_arity.push_back(0);

			for (int i=0; i<seeds.size();++i)
			{
				seedstacks.push_back(vector<node>());

				Eqn2Line(seeds.at(i),seedstacks.at(i));
			}
		}
		//normalize fn weights
		if (weight_ops_on)
		{
			// cout <<"weight_ops_on\n";

			float sumweight = accumulate(op_weight.begin(),op_weight.end(),0.0);
			for(unsigned int i=0;i<op_weight.size();++i)
	                        op_weight.at(i) = op_weight.at(i)/sumweight;
			vector<int>::iterator io = op_choice.begin();
			vector<int>::iterator ia = op_arity.begin();
			vector<string>::iterator il = op_list.begin();
			for (vector<float>::iterator it = op_weight.begin() ; it != op_weight.end();){
				if (*it == 0){
					it = op_weight.erase(it);
					io = op_choice.erase(io);
					ia = op_arity.erase(ia);
					il = op_list.erase(il);
				}
				else{
					++it;
					++io;
					++ia;
					++il;
				}
			}
		}

		// // debugging
		// cout << "savename: " << savename << "\n";
		// cout << "op_list: ";
		// for (auto i : op_list){
		// 	cout << i << ",";
		// }
		// cout << "\n";
		// cout << "op_choice: ";
		// for (auto i : op_choice){
		// 	cout << i << ",";
		// }
		// cout << "\n";
		// cout << "op_arity: ";
		// for (auto i : op_arity){
		// 	cout << i << ",";
		// }
		// cout << "\n";
		// cout << "return_type: ";
		// for (auto i : return_type){
		// 	cout << i << ",";
		// }
		// cout << "\n";
		//

		// turn off AR_nb if AR is not being used
		if (!AR){
			AR_na = 0;
			AR_nb = 0;
			AR_nka = 0;
			AR_nkb = 0;
			AR_lookahead= 0;
		}

		// set train pct to 1 if train is zero
		if (!train) train_pct=1;

		// turn on lex_eps_global if an epsilon method is not used
		if (!lex_eps_global && !(lex_eps_std || lex_eps_error_mad || lex_eps_target_mad || lex_eps_error || lex_eps_target ))
			lex_eps_global = true;

		// make min_len equal the number of classes if m3gp is used
		if(class_m4gp && min_len < number_of_classes)
			min_len = number_of_classes;

	}
};
#endif
