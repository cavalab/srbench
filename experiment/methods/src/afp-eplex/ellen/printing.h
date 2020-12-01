#pragma once
#ifndef LOAD_PRINTING_H
#define LOAD_PRINTING_H
#include <boost/uuid/uuid_io.hpp>
using namespace std;


void printbestind(tribe& T,params& p,state& s,string& logname)
{
	if (p.verbosity>0) s.out << "saving best ind... \n";
	ind best;
	T.getbestind(best);
	string bestname = logname.substr(0,logname.size()-4)+".best_ind";
	std::ofstream fout;
	fout.open(bestname);
	//boost::progress_timer timer;
	fout << "--- Best Individual ---------------------------------------------------------------" << "\n";
	fout << "Corresponding Logfile: " + logname + "\n";
	fout << "Total Evaluations: " << s.totalevals() << "\n";
	fout << "f = " + best.eqn + "\n";
	fout << "gline: ";
	for(unsigned int i =0;i<best.line.size();++i)
	{
		if (best.line[i].type=='n')
			fout << best.line[i].value << "\t";
		else if (best.line[i].type=='v')
			fout << best.line[i].varname << "\t";
		else
			fout << best.line[i].type << "\t";
	}
	fout << endl;
	fout << "eline: ";
	for(unsigned int i =0;i<best.line.size();++i){
		if (best.line.at(i).on)
			fout <<"1\t";
		else
			fout <<"0\t";
	}
	fout << endl;
	fout << "size: " << best.line.size() << "\n";
	fout << "eff size: " << best.eff_size << "\n";
	fout << "training abs error: " << best.abserror<< "\n";
	fout << "training square error: " << best.sq_error << "\n";
	fout << "training correlation: " << best.corr<< "\n";
	fout << "training fitness: " << best.fitness<< "\n";
	fout << "training VAF: " << best.VAF <<"\n";
	fout << "validation abs error: " << best.abserror_v<< "\n";
	fout << "validation square error: " << best.sq_error_v << "\n";
	fout << "validation correlation: " << best.corr_v<< "\n";
	fout << "validation fitness: " << best.fitness_v<< "\n";
	fout << "validation VAF: " << best.VAF_v <<"\n";
	fout << "parent fitness: " << best.parentfitness << "\n";
	fout << "origin: " << best.origin << "\n";
	fout << "age: " << best.age << "\n";
	fout << "eqn form: " << best.eqn_form << "\n";
	if (p.classification && p.class_m4gp){
		fout << "M:\n";
		fout << best.M << "\n";
		for (unsigned i = 0; i<p.number_of_classes; ++i){
			fout << "C[" << i << "]:\n";
			fout << best.C[i] << "\n";
		}
	}
	fout << "output:\n";
	for(unsigned int i =0;i<best.output.size();++i)
	{
		fout << best.output.at(i) << "\n";
	}
	//fout<<"\n";
}
void initdatafile(std::ofstream& dfout,string & logname,params& p)
{
	string dataname = logname.substr(0,logname.size()-4)+".data";
	// cout << "dataname: " << dataname << "\n";
	dfout.open(dataname,std::ofstream::out | std::ofstream::app);
	if (!dfout.is_open()){
	   cerr << "data filee " << p.resultspath + '/' + dataname << " did not open correctly.\n";
	   exit(1);
   }
	//dfout.open(dataname,std::ofstream::app);
	//dfout << "pt_evals\tbest_eqn\tbest_fit\tbest_fit_v\tmed_fit\tmed_fit_v\tbest_MAE\tbest_MAE_v\tbest_R2\tbest_R2_v\tbest_VAF\tbest_VAF_v\tsize\teff_size\tpHC_pct\teHC_pct\tgood_g_pct\tneut_g_pct\tbad_g_pct\ttot_hom\ton_hom\toff_hom\n";
	dfout << "gen\tpt_evals\tbest_eqn\tbest_fit\tbest_fit_v\tmed_fit\tmed_fit_v\tbest_MAE\tbest_MAE_v\tbest_MSE\tbest_MSE_v\tbest_R2\tbest_R2_v\tbest_VAF\tbest_VAF_v\tsize\teff_size\tpHC_pct\teHC_pct\teHC_ties\tgood_g_pct\tneut_g_pct\tbad_g_pct\ttime";
	if (p.print_homology)
		dfout << "\ttot_hom\ton_hom\toff_hom";
	if (p.classification && p.class_m4gp)
		dfout << "\tdimension" ;
	if (p.print_novelty)
		dfout << "\tnovelty" ;
	if (p.print_protected_operators)
		dfout << "\tbest_eqn_matlab";
	if (p.sel == 3) {
		dfout << "\tmedian_lex_cases";
		dfout << "\tmedian_lex_pool";
	}
	dfout << "\n";
	//fout.close(dataname);
}
void printdatafile(tribe& T,state& s,params& p, vector<Randclass>& r,std::ofstream& dfout, int gen, double elapsed_time)
{
	//string dataname = logname.substr(0,logname.size()-4)+".data";
	//std::ofstream fout;
	//dfout.open(dataname,std::ofstream::app);

	sub_ind best_ind;
	T.getbestsubind(best_ind);

	/*dfout << s.totalptevals() << "\t" << best_ind.eqn << "\t" << T.bestFit() << "\t" << T.bestFit_v() << "\t" << T.medFit() << "\t" << T.medFit_v() << "\t" << best_ind.abserror << "\t" << best_ind.abserror_v << "\t" << best_ind.corr << "\t" << best_ind.corr_v << "\t" << T.meanSize() << "\t" << T.meanEffSize() << "\t" << s.current_pHC_updates/float(p.popsize)*100.0 << "\t" << s.current_eHC_updates/float(p.popsize)*100.0 << "\t" <<  s.good_cross_pct << "\t" << s.neut_cross_pct << "\t" << s.bad_cross_pct;*/
	dfout << gen << "\t" << s.totalptevals() << "\t" << best_ind.eqn << "\t" << T.bestFit() << "\t" << T.bestFit_v() << "\t" << T.medFit() << "\t" << T.medFit_v() << "\t" << best_ind.abserror << "\t" << best_ind.abserror_v << "\t" << best_ind.sq_error << "\t" << best_ind.sq_error_v << "\t" << best_ind.corr << "\t" << best_ind.corr_v << "\t" << best_ind.VAF << "\t" << best_ind.VAF_v << "\t" << T.meanSize() << "\t" << T.meanEffSize() << "\t" << s.current_pHC_updates/float(p.popsize)*100.0 << "\t" << s.current_eHC_updates/float(p.popsize)*100.0 << "\t" << s.current_eHC_ties / float(p.popsize)*100.0 << "\t" <<  s.good_cross_pct << "\t" << s.neut_cross_pct << "\t" << s.bad_cross_pct << "\t" << elapsed_time;
	if (p.print_homology){
		float tot_hom, on_hom, off_hom;
		T.hom(r,tot_hom,on_hom,off_hom);
		dfout << "\t" << tot_hom << "\t" << on_hom << "\t" << off_hom;
	}
	if (p.classification && p.class_m4gp)
		dfout << "\t" << best_ind.dim ;
	if (p.print_novelty){
		float novelty;
		T.novelty(novelty);
		dfout << "\t" << novelty;
	}
	if (p.print_protected_operators)
		dfout << "\t" + best_ind.eqn_matlab;
	if (p.sel == 3) {
		dfout << "\t" << s.get_median_lex_cases();
		dfout << "\t" << s.get_median_lex_pool();
		// dfout << "\t" << s.get_median_passes_per_case();
	}
	dfout <<"\n";

	//s.clearCross();
}
void printpop(vector<ind>& pop,params& p,state& s,string& logname,int type)
{
	string bestname;
	if (type==1){
		//s.out << "saving pareto archive... \n";
		bestname = logname.substr(0,logname.size()-4)+".archive";
		sort(pop.begin(),pop.end(),SortComplexity());
		stable_sort(pop.begin(),pop.end(),SortRank());
	}
	else if (type == 2){
		string gen = to_string(static_cast<long long>(s.genevals.size()));
		if (p.verbosity>0) s.out << "saving pop... \n";
		bestname = logname.substr(0,logname.size()-4) + "gen" + gen + ".pop";
		sort(pop.begin(),pop.end(),SortFit());
	}
	else if (type == 3){
		string gen = to_string(static_cast<long long>(s.genevals.size()));
		if (p.verbosity>0) s.out << "saving initial pop... \n";
		bestname = logname.substr(0,logname.size()-4) + ".init_pop";
		sort(pop.begin(),pop.end(),SortFit());
	}
	else {
		if (p.verbosity>0) s.out << "saving last pop... \n";
		bestname = logname.substr(0,logname.size()-4)+".last_pop";
		//bestname = "lastpop.last_pop";
		sort(pop.begin(),pop.end(),SortComplexity());
		stable_sort(pop.begin(),pop.end(),SortFit());
	}

	std::ofstream fout;
	fout.open(bestname);
	//boost::progress_timer timer;
	fout << "--- Population ---------------------------------------------------------------" << "\n";
	fout << "Corresponding Logfile: " + logname + "\n";
	fout << "Total Evaluations: " << s.totalevals() << "\n";

	for (int h = 0; h<pop.size(); ++h){
		fout << "--- Individual "<< h << " ------------------------------------------------------------" << "\n";
		fout << "f = " + pop.at(h).eqn + "\n";
		fout << "gline: ";
		for(unsigned int i =0;i<pop.at(h).line.size();++i)
		{
			if (pop.at(h).line[i].type=='n')
				fout <<pop.at(h).line[i].value << "\t";
			else if (pop.at(h).line[i].type=='v')
				fout << pop.at(h).line[i].varname << "\t";
			else
				fout << pop.at(h).line[i].type << "\t";
		}
		fout << endl;
		fout << "eline: ";
		for(unsigned int i =0;i<pop.at(h).line.size();++i){
			if (pop.at(h).line.at(i).on)
				fout <<"1\t";
			else
				fout <<"0\t";
		}
		fout << endl;
		fout << "size: " << pop.at(h).line.size() << "\n";
		fout << "eff size: " << pop.at(h).eff_size << "\n";
		fout << "complexity: " << pop.at(h).complexity << "\n";
		fout << "fitness: " << pop.at(h).fitness<< "\n";
		fout << "fitness_v: " << pop.at(h).fitness_v<< "\n";
		fout << "MAE: " << pop.at(h).abserror<< "\n";;
		fout << "MAE_v: " << pop.at(h).abserror_v<< "\n";
		fout << "MSE: " << pop.at(h).sq_error << "\n";;
		fout << "MSE_v: " << pop.at(h).sq_error_v << "\n";
		fout << "correlation: " << pop.at(h).corr<< "\n";
		fout << "correlation_v: " << pop.at(h).corr_v<< "\n";
		fout << "VAF: " << pop.at(h).VAF<< "\n";
		fout << "VAF_v: " << pop.at(h).VAF_v<< "\n";
		fout <<  "rank: " << pop.at(h).rank << "\n";
		fout << "parent fitness: " << pop.at(h).parentfitness << "\n";
		fout << "origin: " << pop.at(h).origin << "\n";
		fout << "age: " << pop.at(h).age << "\n";
		fout << "eqn form: " << pop.at(h).eqn_form << "\n";
		if (p.classification && p.class_m4gp) fout << "dimensions: " << pop[h].dim << "\n";
		/*fout << "output: ";
		for(unsigned int i =0;i<pop.at(h).output.size();++i)
		{
			fout << T.pop.at(h).output.at(i);
		}
		fout<<"\n";*/
		fout << "------------------------------------------------------------------" << "\n";
		}
}
void printDB(vector<ind>& pop,string& logname,Data& d,params& p)
{
	// print JSON formatted inviduals for graph database use
	// print format:
//		{"individual": {
// 			"id": uuid,
// 			"program": [{nodes}],
//			"eqn":	equation string,
// 			"mse": mean_squared_error,
// 			"origin": crossover, mutation
// 			"parent_id": [parent1 uuid, parent2 uuid]
// 		}

	std::ofstream fout;
	// reopen database file
	fout.open(logname.substr(0,logname.size()-4)+".db", std::ios_base::app);

	for (auto i: pop){
		fout << "{\"individual\": {";
		fout << "\"id\": \"" << i.tag << "\"";
		fout << ", \"program\": [";
		//  print program nodes  << [{nodes}];
		int j = 0;
		for (auto n: i.line){
			if (n.on){
				if (j != 0)
					fout << ", ";
				fout << "{\"type\": \"" << n.type << "\", \"id\":\"" << n.tag << "\"";
				if (n.type=='v')
					fout << ", \"value\": \"" << n.varname << "\"";
				else if (n.type=='n')
					fout << ", \"value\": " << n.value;
				fout << "}";
				++j;
			}
		}
		fout << "]"; // end of program
		fout << ", \"eqn\": \"" << i.eqn << "\"";
		fout << ", \"mse\": " << i.sq_error;
		fout << ", \"origin\": \"" << i.origin << "\"";
		if (i.origin=='c'){
 			fout << ", \"parent_id\": [\"" << i.parent_id[0] << "\", \"" << i.parent_id[1] << "\"]";
		}
		else if (i.origin=='m'){
			// int tmp = i.parent_id.size();
			// fout << ", parent_id size: " << tmp ;
			fout << ", \"parent_id\": [\"" << i.parent_id[0] << "\"]";
		}
		fout << "}}\n";
	}
	fout.close();
}
#endif
