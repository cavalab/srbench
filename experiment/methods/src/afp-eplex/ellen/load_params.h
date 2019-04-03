#pragma once
#ifndef LOAD_PARAMS_H
#define LOAD_PARAMS_H

using namespace std;


void load_params(params &p, std::ifstream& fs)
{
	if (!fs.good())
	{
			cerr << "BAD PARAMETER FILE LOCATION" << "\n";
			exit(1);
	}

	string s;
	string varname;
	float tmpf;
	//int tmpi;
	string tmps;
	//bool tmpb;

	//string trash;
	//s.erase();
    //s.reserve(is.rdbuf()->in_avail());

	while (!fs.eof())
	{
		getline(fs, s, '\n');
		istringstream ss(s);

		ss >> varname;

		//getline(is,s,'\t');

		if (varname.compare("g") == 0)
		{
			ss >> p.g;
			//p.g = tmp;
		}
		else if (varname.compare("popsize") == 0)
			ss >> p.popsize;
		else if (varname.compare("sel") == 0)
			ss >> p.sel;
		else if (varname.compare("tourn_size") == 0)
			ss >> p.tourn_size;
		else if (varname.compare("rt_rep") == 0)
			ss >> p.rt_rep;
		else if (varname.compare("rt_cross") == 0)
			ss >> p.rt_cross;
		else if (varname.compare("rt_mut") == 0)
			ss >> p.rt_mut;
		else if (varname.compare("cross") == 0)
			ss >> p.cross;
		else if (varname.compare("cross_ar") == 0)
			ss >> p.cross_ar;
		else if (varname.compare("mut_ar") == 0)
			ss >> p.mut_ar;
		//~ else if(varname.compare("stoperror") == 0)
			//~ ss>>p.stoperror;
		else if (varname.compare("init_validate_on") == 0)
			ss >> p.init_validate_on;
		else if (varname.compare(0, 11, "resultspath") == 0)
		{
			int q = 0;
			while (ss >> tmps)
			{
				if (q > 0)
					p.resultspath = p.resultspath + ' ';
				p.resultspath.insert(p.resultspath.end(), tmps.begin(), tmps.end());
				++q;
			}
		}
		//~ else if(varname.compare(0,4,"loud") == 0)
			//~ ss>>p.loud;
		/*else if(varname.compare(0,8,"parallel") == 0)
			ss>>p.parallel;
		else if(varname.compare(0,8,"numcores") == 0)
			ss>>p.numcores;*/
			//~ else if(varname.compare(0,11,"sim_nom_mod") == 0)
				//~ ss>>p.sim_nom_mod;
			//~ else if(varname.compare(0,7,"nstates") == 0)
				//~ ss>>p.nstates;
		else if (varname.compare(0, 7, "intvars") == 0)
		{
			while (ss >> tmps)
				p.intvars.push_back(tmps);
		}
		/*else if(varname.compare(0,7,"extvars") == 0)
		{
			while (ss>>tmps)
				p.extvars.push_back(tmps);
		}*/
		/*else if(varname.compare("cons") == 0)
		{
			while (ss>>tmps)
				p.cons.push_back(tmps);
		}*/
		else if (varname.compare("cvals") == 0)
		{
			while (ss >> tmpf) {
				p.cvals.push_back(tmpf);
				p.cons.push_back(std::to_string(static_cast<long double>(tmpf)));
			}

		}
		else if (varname.compare("seeds") == 0)
		{
			while (ss >> tmps)
				p.seeds.push_back(tmps);
		}
		else if (varname.compare("ERC") == 0)
			ss >> p.ERC;
		else if (varname.compare("ERCints") == 0)
			ss >> p.ERCints;
		else if (varname.compare("maxERC") == 0)
			ss >> p.maxERC;
		else if (varname.compare("minERC") == 0)
			ss >> p.minERC;
		else if (varname.compare("numERC") == 0)
			ss >> p.numERC;
		else if (varname.compare("fit_type") == 0)
			ss >> p.fit_type;
		else if (varname.compare("max_fit") == 0)
			ss >> p.max_fit;
		else if (varname.compare("min_fit") == 0)
			ss >> p.min_fit;
		else if (varname.compare("op_list") == 0)
		{
			while (ss >> tmps)
				p.op_list.push_back(tmps);
		}
		else if (varname.compare("op_weight") == 0)
		{
			while (ss >> tmpf)
				p.op_weight.push_back(tmpf);
		}
		else if (varname.compare("weight_ops_on") == 0)
			ss >> p.weight_ops_on;
		else if (varname.compare("min_len") == 0)
			ss >> p.min_len;
		else if (varname.compare("max_len") == 0)
			ss >> p.max_len;
		else if (varname.compare("max_len_init") == 0)
			ss >> p.max_len_init;
		//~ else if(varname.compare("max_dev_len") == 0)
			//~ ss>>p.max_dev_len;
		else if (varname.compare("complex_measure") == 0)
			ss >> p.complex_measure;
		//~ else if(varname.compare("precision") == 0)
			//~ ss>>p.precision;
		else if (varname.compare("lineHC_on") == 0)
			ss >> p.lineHC_on;
		else if (varname.compare("lineHC_its") == 0)
			ss >> p.lineHC_its;
		else if (varname.compare("pHC_on") == 0)
			ss >> p.pHC_on;
		else if (varname.compare("pHC_delay_on") == 0)
			ss >> p.pHC_delay_on;
		else if (varname.compare("pHC_size") == 0)
			ss >> p.pHC_size;
		else if (varname.compare("pHC_its") == 0)
			ss >> p.pHC_its;
		else if (varname.compare("pHC_gauss") == 0)
			ss >> p.pHC_gauss;
		else if (varname.compare("eHC_on") == 0)
			ss >> p.eHC_on;
		else if (varname.compare("eHC_its") == 0)
			ss >> p.eHC_its;
		else if (varname.compare("eHC_prob") == 0)
			ss >> p.eHC_prob;
		//~ else if(varname.compare("eHC_size") == 0)
			//~ ss>>p.eHC_size;
		//~ else if(varname.compare("eHC_cluster") == 0)
			//~ ss>>p.eHC_cluster;
		//~ else if(varname.compare("eHC_dev") == 0)
			//~ ss>>p.eHC_dev;
		//~ else if(varname.compare("eHC_best") == 0)
			//~ ss>>p.eHC_best;
		else if (varname.compare("eHC_init") == 0)
			ss >> p.eHC_init;
		//~ else if(varname.compare("eHC_prob_scale") == 0)
			//~ ss>>p.eHC_prob_scale;
		//~ else if(varname.compare("eHC_max_prob") == 0)
			//~ ss>>p.eHC_max_prob;
		//~ else if(varname.compare("eHC_min_prob") == 0)
			//~ ss>>p.eHC_min_prob;
		else if (varname.compare("eHC_mut") == 0)
			ss >> p.eHC_mut;
		else if (varname.compare("eHC_slim") == 0)
			ss >> p.eHC_slim;
		else if (varname.compare("lexpool") == 0)
			ss >> p.lexpool;
		/*else if(varname.compare("lexage") == 0)
			ss>>p.lexage;*/ // removed lexage from input. it is now used internally for age as a metacase.
		else if (varname.compare("prto_arch_on") == 0)
			ss >> p.prto_arch_on;
		else if (varname.compare("prto_arch_size") == 0)
			ss >> p.prto_arch_size;
		else if (varname.compare("prto_sel_on") == 0)
			ss >> p.prto_sel_on;
		else if (varname.compare("islands") == 0)
			ss >> p.islands;
		else if (varname.compare("island_gens") == 0)
			ss >> p.island_gens;
		else if (varname.compare("train") == 0)
			ss >> p.train;
		else if (varname.compare("train_pct") == 0)
			ss >> p.train_pct;
		else if (varname.compare("print_every_pop") == 0)
			ss >> p.print_every_pop;
		else if (varname.compare("estimate_fitness") == 0)
			ss >> p.EstimateFitness;
		else if (varname.compare("FE_pop_size") == 0)
			ss >> p.FE_pop_size;
		else if (varname.compare("FE_ind_size") == 0)
			ss >> p.FE_ind_size;
		else if (varname.compare("FE_train_size") == 0)
			ss >> p.FE_train_size;
		else if (varname.compare("FE_train_gens") == 0)
			ss >> p.FE_train_gens;
		else if (varname.compare("FE_rank") == 0)
			ss >> p.FE_rank;
		else if (varname.compare("estimate_generality") == 0)
			ss >> p.estimate_generality;
		else if (varname.compare("G_sel") == 0)
			ss >> p.G_sel;
		else if (varname.compare("G_shuffle") == 0)
			ss >> p.G_shuffle;
		else if (varname.compare("norm_error") == 0)
			ss >> p.norm_error;
		else if (varname.compare("shuffle_data") == 0)
			ss >> p.shuffle_data;
		else if (varname.compare("init_trees") == 0)
			ss >> p.init_trees;
		else if (varname.compare("limit_evals") == 0)
			ss >> p.limit_evals;
		else if (varname.compare("max_evals") == 0)
			ss >> p.max_evals;
		else if (varname.compare("print_homology") == 0)
			ss >> p.print_homology;
		else if (varname.compare("print_log") == 0)
			ss >> p.print_log;
		else if (varname.compare("print_init_pop") == 0)
			ss >> p.print_init_pop;
		else if (varname.compare("print_genome") == 0)
			ss >> p.print_genome;
		else if (varname.compare("print_epigenome") == 0)
			ss >> p.print_epigenome;
		else if (varname.compare("num_log_pts") == 0)
			ss >> p.num_log_pts;
		else if (varname.compare("PS_sel") == 0)
			ss >> p.PS_sel;
		else if (varname.compare("pop_restart") == 0)
			ss >> p.pop_restart;
		else if (varname.compare("pop_restart_path") == 0)
			ss >> p.pop_restart_path;
		else if (varname.compare("AR") == 0)
			ss >> p.AR;
		else if (varname.compare("AR_na") == 0)
			ss >> p.AR_na;
		else if (varname.compare("AR_nka") == 0) {
			ss >> p.AR_nka;
			if (p.AR_nka < 1) {
				cout << "WARNING: AR_nka set to min value of 1\n";
				p.AR_nka = 1;
			}
		}
		else if(varname.compare("AR_nb") == 0)
			ss>>p.AR_nb;
		else if (varname.compare("AR_nkb") == 0)
			ss >> p.AR_nkb;
		else if(varname.compare("AR_lookahead") == 0)
			ss>>p.AR_lookahead;
		else if(varname.compare("align_dev") ==0)
			ss>>p.align_dev;
		else if(varname.compare("classification")==0)
			ss>>p.classification;
		else if(varname.compare("class_bool")==0)
			ss>>p.class_bool;
		else if(varname.compare("class_m4gp")==0)
			ss>>p.class_m4gp;
		else if(varname.compare("class_prune")==0)
			ss>>p.class_prune;
		else if(varname.compare("number_of_classes")==0)
			ss>>p.number_of_classes;
		else if(varname.compare("elitism")==0)
			ss>>p.elitism;
		else if(varname.compare("stop_condition")==0)
			ss>>p.stop_condition;
		else if(varname.compare("mutate")==0)
			ss>>p.mutate;
		else if(varname.compare("print_novelty")==0)
			ss>>p.print_novelty;
		else if(varname.compare("lex_metacases")==0)
		{
			while (ss>>tmps)
				p.lex_metacases.push_back(tmps);
		}
		else if(varname.compare("lex_class")==0)
			ss >> p.lex_class;
		else if(varname.compare("weight_error")==0)
			ss >> p.weight_error;
		else if(varname.compare("print_protected_operators")==0)
			ss >> p.print_protected_operators;
		else if (varname.compare("lex_eps_error") == 0)
			ss >> p.lex_eps_error;
		else if (varname.compare("lex_eps_target") == 0)
			ss >> p.lex_eps_target;
		else if (varname.compare("lex_eps_std") == 0)
			ss >> p.lex_eps_std;
		else if (varname.compare("lex_eps_target_mad") == 0)
			ss >> p.lex_eps_target_mad;
		else if (varname.compare("lex_eps_error_mad") == 0)
			ss >> p.lex_eps_error_mad;
		else if (varname.compare("lex_epsilon") == 0)
			ss >> p.lex_epsilon;
		else if (varname.compare("lex_eps_global") == 0)
			ss >> p.lex_eps_global;
		else if (varname.compare("test_at_end") == 0)
			ss >> p.test_at_end;
		else{}
    }
	p.allvars = p.intvars;
	//p.allvars.insert(p.allvars.end(), p.extvars.begin(), p.extvars.end());
	p.allblocks = p.allvars;
	p.allblocks.insert(p.allblocks.end(),p.cons.begin(),p.cons.end());
	p.allblocks.insert(p.allblocks.end(),p.seeds.begin(),p.seeds.end());

	//p.seed = time(0);

	if (p.max_len_init == 0)
		p.max_len_init = p.max_len;
	// op_list
	if (p.op_list.empty()){ // set default operator list
		p.op_list.push_back("n");
		p.op_list.push_back("v");
		p.op_list.push_back("+");
		p.op_list.push_back("-");
		p.op_list.push_back("*");
		p.op_list.push_back("/");
	}
	for (unsigned int i=0; i<p.op_list.size(); ++i)
	{
		if (p.op_list.at(i).compare("n")==0 )//&& ( p.ERC || !p.cvals.empty() ) )
		{
			p.op_choice.push_back(0);
			p.op_arity.push_back(0);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("v")==0)
		{
			p.op_choice.push_back(1);
			p.op_arity.push_back(0);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("+")==0)
		{
			p.op_choice.push_back(2);
			p.op_arity.push_back(2);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("-")==0)
		{
			p.op_choice.push_back(3);
			p.op_arity.push_back(2);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("*")==0)
		{
			p.op_choice.push_back(4);
			p.op_arity.push_back(2);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("/")==0)
		{
			p.op_choice.push_back(5);
			p.op_arity.push_back(2);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("sin")==0)
		{
			p.op_choice.push_back(6);
			p.op_arity.push_back(1);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("cos")==0)
		{
			p.op_choice.push_back(7);
			p.op_arity.push_back(1);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("exp")==0)
		{
			p.op_choice.push_back(8);
			p.op_arity.push_back(1);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("log")==0)
		{
			p.op_choice.push_back(9);
			p.op_arity.push_back(1);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("sqrt")==0)
		{
			p.op_choice.push_back(11);
			p.op_arity.push_back(1);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("^") == 0)
		{
			p.op_choice.push_back(12);
			p.op_arity.push_back(2);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("=")==0)
		{
			p.op_choice.push_back(13);
			p.op_arity.push_back(2);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare("!")==0)
		{
			p.op_choice.push_back(14);
			p.op_arity.push_back(2);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare("<")==0)
		{
			p.op_choice.push_back(15);
			p.op_arity.push_back(2);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare(">")==0)
		{
			p.op_choice.push_back(16);
			p.op_arity.push_back(2);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare("<=")==0)
		{
			p.op_choice.push_back(17);
			p.op_arity.push_back(2);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare(">=")==0)
		{
			p.op_choice.push_back(18);
			p.op_arity.push_back(2);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare("if-then")==0)
		{
			p.op_choice.push_back(19);
			p.op_arity.push_back(2);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("if-then-else")==0)
		{
			p.op_choice.push_back(20);
			p.op_arity.push_back(3);
			p.return_type.push_back('f');
		}
		else if (p.op_list.at(i).compare("&")==0)
		{
			p.op_choice.push_back(21);
			p.op_arity.push_back(3);
			p.return_type.push_back('b');
		}
		else if (p.op_list.at(i).compare("|")==0)
		{
			p.op_choice.push_back(22);
			p.op_arity.push_back(3);
			p.return_type.push_back('b');
		}
		else
			cout << "bad command (load params op_choice)" << "\n";
	}

	p.rep_wheel.push_back(p.rt_rep);
	p.rep_wheel.push_back(p.rt_cross);
	p.rep_wheel.push_back(p.rt_mut);

	partial_sum(p.rep_wheel.begin(), p.rep_wheel.end(), p.rep_wheel.begin());

	if(!p.seeds.empty()) // get seed stacks
	{
		p.op_choice.push_back(10); // include seeds in operation choices
		p.op_weight.push_back(1); // include opweight if used
		p.op_list.push_back("seed");
		p.op_arity.push_back(0);

		for (int i=0; i<p.seeds.size();++i)
		{
			p.seedstacks.push_back(vector<node>());

			Eqn2Line(p.seeds.at(i),p.seedstacks.at(i));
		}
	}
	//normalize fn weights
	if (p.weight_ops_on)
	{
		float sumweight = accumulate(p.op_weight.begin(),p.op_weight.end(),0.0);
		for(unsigned int i=0;i<p.op_weight.size();++i)
                        p.op_weight.at(i) = p.op_weight.at(i)/sumweight;
		vector<int>::iterator io = p.op_choice.begin();
		vector<int>::iterator ia = p.op_arity.begin();
		vector<string>::iterator il = p.op_list.begin();
		for (vector<float>::iterator it = p.op_weight.begin() ; it != p.op_weight.end();){
			if (*it == 0){
				it = p.op_weight.erase(it);
				io = p.op_choice.erase(io);
				ia = p.op_arity.erase(ia);
				il = p.op_list.erase(il);
			}
			else{
				++it;
				++io;
				++ia;
				++il;
			}
		}
	}

	// turn off AR_nb if AR is not being used
	if (!p.AR){
		p.AR_na = 0;
		p.AR_nb = 0;
		p.AR_nka = 0;
		p.AR_nkb = 0;
		p.AR_lookahead= 0;
	}

	// set train pct to 1 if train is zero
	if (!p.train) p.train_pct=1;

	// add lexage flag if age is a metacase
	p.lexage=false;
	for (unsigned i = 0; i<p.lex_metacases.size(); ++i)
	{
		if (p.lex_metacases[i].compare("age")==0)
			p.lexage=true;
	}
	// turn on lex_eps_global if an epsilon method is not used
	if (!p.lex_eps_global && !(p.lex_eps_std || p.lex_eps_error_mad || p.lex_eps_target_mad || p.lex_eps_error || p.lex_eps_target ))
		p.lex_eps_global = true;

	// make p.min_len equal the number of classes if m3gp is used
	if(p.class_m4gp && p.min_len < p.number_of_classes)
		p.min_len = p.number_of_classes;

}

#endif
