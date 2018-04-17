// header file for ind struct
#pragma once
#ifndef POP_H
#define POP_H

//#include "params.h"
//#include "data.h"

//#include "RPN_class.h"
#include "op_node.h"
#include "rnd.h"
#include "strdist.h"
//#include <Eigen/Dense>
using Eigen::MatrixXf;
using Eigen::VectorXf;
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
//#include "general_fns.h"
//#include "pareto.h"

struct ind {
	/*
	===================================================================
	BE SURE TO ADD ANY NEW VARIABLES TO THE SWAP FUNCTION FOR COPYING!!
	===================================================================
	*/
	//unsigned int id;
	/*vector <std::shared_ptr<node> > line;*/

	vector <node> line;
	vector<float> output;
	vector<float> output_v;
	vector<float> error; // fitnesses for lexicase selection
	vector<float> f; // vector of objectives for multi-objective implementations (PS_sel)
	std::vector<unsigned int> stack_floatlen;
	std::vector<float> stack_float; // linearized stack_float
	std::vector<int> dominated; //for spea2 strength
	std::vector<MatrixXf> C; // covariance matrices for M3GP

	std::string eqn;
	std::string eqn_form; // equation form for string distance comparison to other forms
	std::string eqn_matlab; // equation for matlab (elementwise and protected operators)
	MatrixXf M; // centroids for M3GP
	float abserror;
	float abserror_v;
	float sq_error;
	float sq_error_v;
	float corr;
	float corr_v;
	float VAF;
	float VAF_v;
	float fitness;
	float fitness_v;
	float FEvar; //variance in fitness estimates (for sorting purposes)
	float GEvar; //variance in generality estimates (for sorting purposes)
	float genty; //generality
	float spea_fit;
	float parentfitness;
	int eff_size;
	int age;
	int rank;
	int complexity;
	int dim;
	char origin; // x: crossover, m: mutation, i: initialization
	boost::uuids::uuid tag; // uuid for graph database tracking
	vector<boost::uuids::uuid> parent_id;
	/*
	===================================================================
	BE SURE TO ADD ANY NEW VARIABLES TO THE SWAP FUNCTION FOR COPYING!!
	===================================================================
	*/

	ind()
	:tag(boost::uuids::random_generator()())
	{
		abserror = 0;
		sq_error = 0;
		corr = 0;
		age = 1;
		genty = 1;
	}
	/*ind(const ind& x)
	{
		*this = x;
	}*/
	~ind() {
		//cout << "ind destructor\n";
		//if(!line.empty())
		//{
		//	for(vector<node*>::iterator it= line.begin();it!=line.end();it++)
		//		delete(*it);
		//	line.clear();
		//	//cout << "ind destructor deleted line nodes\n";
		//}
		//
	}
	ind & operator = (ind s) // over-ride copy construction with swap
    {
      s.swap (*this); // Non-throwing swap
      return *this;
    }

    void swap (ind &s)
	{

		line.swap(s.line);						// vectors
		output.swap(s.output);
		output_v.swap(s.output_v);
		error.swap(s.error);
		f.swap(s.f);
		stack_floatlen.swap(s.stack_floatlen);
		stack_float.swap(s.stack_float);
		dominated.swap(dominated);
		C.swap(s.C);
		parent_id.swap(s.parent_id);

		eqn.swap(s.eqn);						// strings
		eqn_form.swap(s.eqn_form);
		eqn_matlab.swap(s.eqn_matlab);

		using std::swap;

		swap(this->M,s.M);
		swap(this->abserror,s.abserror);		// floats
		swap(this->abserror_v,s.abserror_v);
		swap(this->sq_error, s.sq_error);		// floats
		swap(this->sq_error_v, s.sq_error_v);
		swap(this->corr,s.corr);
		swap(this->corr_v,s.corr_v);
		swap(this->VAF,s.VAF);
		swap(this->VAF_v,s.VAF_v);
		swap(this->fitness,s.fitness);
		swap(this->fitness_v,s.fitness_v);
		swap(this->FEvar,s.FEvar);
		swap(this->GEvar,s.GEvar);
		swap(this->genty,s.genty);
		swap(this->spea_fit,s.spea_fit);
		swap(this->parentfitness,s.parentfitness);
		swap(this->eff_size,s.eff_size);		// ints
		swap(this->age,s.age);
		swap(this->rank,s.rank);
		swap(this->complexity,s.complexity);
		swap(this->dim,s.dim);

		swap(this->origin,s.origin);			// chars

		swap(this->tag,s.tag); 					//uuid identifier

	}//throw (); // Also see the non-throwing swap idiom
	////swap optimization
	//void swap(ind&) throw();
	//void init(string& nom_mod)
	//{
	//	eqn = nom_mod;
	//	ptr.push_back(1);
	//	ptr.push_back(nom_mod.size()-2);
	//	//nominal_model=nom_mod;
	//	//expression.register_symbol_table(d.symbol_table);
	//}
	void reset_introns()
	{
		for (size_t i = 0; i < line.size(); ++i) // set all introns to true
			line[i].intron = true;
	}
	void clrPhen()
	{
		abserror = 0;
		abserror_v=0;
		sq_error = 0;
		sq_error_v = 0;
		corr = 0;
		corr_v=0;
		fitness=0;
		fitness_v=0;
		VAF = 0;
		VAF_v = 0;
		eqn = "";
		eqn_form="";
		output.clear();
		output_v.clear();
		error.clear();
		C.clear();
		M.resize(0,0);
		genty = 1;

		//stack_float.clear();
		// nominal model must be encased in set of parenthesis. the pointer points to that which is encased.
		//ptr[0]= 1;
		//ptr[1] = nom_mod.size()-2;
	}
//private:
//	string& nominal_model;
};
struct sub_ind
{
	float fitness;
	float abserror;
	float sq_error;
	float corr;
	float VAF;

	float abserror_v;
	float sq_error_v;
	float corr_v;
	float VAF_v;
	string eqn;
	string eqn_matlab;
	int age;
	int complexity;
	int dim;
	sub_ind(){}
	void init(ind& x){
		fitness = x.fitness;
		abserror = x.abserror;
		abserror_v = x.abserror_v;
		sq_error = x.sq_error;
		sq_error_v = x.sq_error_v;
		corr = x.corr;
		corr_v = x.corr_v;
		VAF = x.VAF;
		VAF_v = x.VAF_v;
		eqn = x.eqn;
		eqn_matlab = x.eqn_matlab;
		age=x.age;
		complexity = x.complexity;
		dim = x.dim;
	}
	~sub_ind(){}
};
//swap optimization
inline void swap(ind& lhs, ind& rhs) { lhs.swap(rhs); }
namespace std { template<> inline void swap<struct ind>(ind& lhs, ind& rhs)	{lhs.swap(rhs);	}}
////using std::swap;
struct SortFit{ bool operator() (const ind& i,const ind& j) { return (i.fitness<j.fitness);} };
struct SortFit2{ bool operator() (const sub_ind& i,const sub_ind& j) { return (i.fitness<j.fitness);} };
struct SortRank{ bool operator() (const ind& i,const ind& j) { return (i.rank<j.rank);} };
struct SortGenty{ bool operator() (const ind& i,const ind& j) { return (i.genty<j.genty);} };
struct revSortRank{	bool operator() (ind& i,ind& j) { return (i.rank>j.rank);} };

struct SortEqnSize{	bool operator() (const ind& i,const ind& j) { return (i.eqn.size()<j.eqn.size());} };
struct SortFEVar{	bool operator() (const ind& i,const ind& j) { return (i.FEvar>j.FEvar);} };
struct SortGEVar{	bool operator() (const ind& i,const ind& j) { return (i.GEvar>j.GEvar);} };
struct SortComplexity{bool operator() (const ind& i,const ind& j) { return (i.complexity<j.complexity);}};
struct SortFit_v{	bool operator() (const ind& i,const ind& j) { return (i.fitness_v<j.fitness_v);}};
struct SortSize{	bool operator() (const ind& i,const ind& j) { return (i.line.size()<j.line.size());}};
struct SortAge{	bool operator() (const ind& i,const ind& j) { return (i.age < j.age );}};

struct sameEqn{	bool operator() (ind& i,ind& j) { return i.eqn==j.eqn;} };

struct sameEqnSize{	bool operator() (ind& i,ind& j) { return i.eqn.size()==j.eqn.size();} };

struct sameSizeFit{	bool operator() (ind& i,ind& j) { return (i.fitness==j.fitness && i.eqn.size()==j.eqn.size());} };

struct sameFit{	bool operator() (ind& i,ind& j) { return (i.fitness==j.fitness);} };
struct sameFit2{	bool operator() (sub_ind& i,sub_ind& j) { return (i.fitness==j.fitness);} };

struct sameOutput{	bool operator() (ind& i, ind& j) {
	if (i.output.size()==j.output.size())
		return std::equal(i.output.begin(),i.output.end(),j.output.begin());
	else
		return 0;
} };

struct sameComplexity{bool operator() (const ind& i,const ind& j) { return (i.complexity==j.complexity);} };

struct sameFitComplexity{bool operator() (const ind& i,const ind& j) { return (i.fitness==j.fitness && i.complexity==j.complexity);} };


struct tribe{

	vector <ind> pop; // population
	float best;
	float worst;

	tribe(int size,float& max_fit,float& min_fit)
	{
		pop.resize(size);
		best=max_fit;
		worst=min_fit;
		/*for(unsigned int i = 1;i<pop.size();++i)
			pop.at(i).init(nom_mod);*/
		maxf=max_fit;
		minf=min_fit;
	}
	~tribe() {}

	float bestFit() // returns best fitness value
	{

		/*#pragma omp parallel
		{
		   float localbest = maxf;

		   #pragma omp for schedule(static)
		   for(int i = 0; i < pop.size(); ++i)
			   localbest = min(localbest, pop.at(i).fitness);

		   #pragma omp critical
		   {
			  best = min(localbest, best);
		   }
		}*/
		best = maxf;
		for(int i = 0; i < pop.size(); ++i)
			   best = min(best, pop.at(i).fitness);
		return best;
	}
	float bestFit_v() // returns best fitness value
	{

		/*#pragma omp parallel
		{
		   float localbest = maxf;

		   #pragma omp for schedule(static)
		   for(int i = 0; i < pop.size(); ++i)
			   localbest = min(localbest, pop.at(i).fitness);

		   #pragma omp critical
		   {
			  best = min(localbest, best);
		   }
		}*/
		best = maxf;
		for(int i = 0; i < pop.size(); ++i)
			   best = min(best, pop.at(i).fitness_v);
		return best;
	}
	float worstFit() //worst fitness
	{
		worst = minf;
		/*#pragma omp parallel
		{
		   float localworst = minf;

		   #pragma omp for schedule(static)
		   for(int i = 0; i < pop.size(); ++i)
			   localworst = max(localworst, pop.at(i).fitness);

		   #pragma omp critical
		   {
			  worst = max(localworst, worst);
		   }
		}*/
		 for(int i = 0; i < pop.size(); ++i)
			 worst = max(worst, pop.at(i).fitness);
		return worst;
	}
	float medFit() //median fitness
	{
		vector<float> fitness(pop.size());
		for(int i =0; i < pop.size(); i++)
			fitness.at(i) = pop.at(i).fitness;
		sort(fitness.begin(),fitness.end());
		if (pop.size() % 2==0) //even
			return fitness.at((int)floor((float)pop.size()/2));
		else
			return (fitness.at(pop.size()/2)+fitness.at(pop.size()/2-1))/2;
	}
	float medFit_v() //median fitness
	{
		vector<float> fitness(pop.size());
		for(int i =0; i < pop.size(); i++)
			fitness.at(i) = pop.at(i).fitness_v;
		sort(fitness.begin(),fitness.end());
		if (pop.size() % 2==0) //even
			return fitness.at((int)floor((float)pop.size()/2));
		else
			return (fitness.at(pop.size()/2)+fitness.at(pop.size()/2-1))/2;

	}
	float meanFit() // mean fitness
	{
		float answer=0;
		//#pragma omp parallel for reduction(+ : answer)
		for(int i=0; i<pop.size(); ++i)
		{
			answer+=pop.at(i).fitness;
		}
		return (float)answer/pop.size();
	}

	float meanSize() // mean line length
	{
		float answer=0;
		//#pragma omp parallel for reduction(+ : answer)
		for(int i=0; i<pop.size(); ++i)
		{
			answer+=pop.at(i).line.size();
		}
		return (float)answer/pop.size();
	}
	float meanEffSize()
	{
		float answer=0;
		//#pragma omp parallel for reduction(+ : answer)
		for(int i=0; i<pop.size(); ++i)
		{
			answer+=pop.at(i).eff_size;
		}
		return (float)answer/pop.size();
	}
	int medSize() // median line length
	{
		//vector<ind> tmppop = pop;
		sort(pop.begin(),pop.end(),SortSize());
		int index = (int)floor((float)pop.size()/2);
		return int(pop.at(index).line.size());
	}

	void topTen(vector <sub_ind>& eqns) //returns address to vector of equation strings
	{
		vector<sub_ind> tmppop(pop.size());
		for (int i = 0;i<pop.size();++i) tmppop[i].init(pop.at(i));
		//vector<ind> tmppop = pop;
		sort(tmppop.begin(),tmppop.end(),SortFit2());
		unique(tmppop.begin(),tmppop.end(),sameFit2());

		for (int i=0;i<10;++i)
			eqns.push_back(tmppop.at(i));

		/*vector <float> fitnesses;
		int i=0;
		bool pass=true;
		while(eqns.size()<10 && i<pop.size())
		{
			fitnesses.push_back(pop.at(i).fitness);
			for(unsigned int j=0;j<fitnesses.size()-1;++j)
			{
				if(fitnesses.at(j)==fitnesses.back())
				{
					fitnesses.pop_back();
					pass=0;
					break;
				}
			}

			if (pass)
				eqns.push_back(pop.at(i));
			else
				pass=1;
			++i;
		}*/
	}
	void getbestsubind(sub_ind& bestind)
	{
		vector<sub_ind> subpop(pop.size());
		for (int i = 0;i<pop.size();++i) subpop[i].init(pop.at(i));
		//vector<ind> tmppop = pop;
		sort(subpop.begin(),subpop.end(),SortFit2());
		bestind = subpop.front();
	}// address of best individual
	void getbestind(ind& bestind)
	{
		//vector<ind> tmppop = pop;
		sort(pop.begin(),pop.end(),SortFit());
		bestind = pop.front();
	}// address of best individual
	void sortpop()
	{
		sort(pop.begin(),pop.end(),SortFit());
	}
	void sortpop_age()
	{
		sort(pop.begin(),pop.end(),SortAge());
	}
	void novelty(float& novelty)
	{ // calculate novelty, where novelty is defined as the percent of unique errors in population
		/*vector<sub_ind> subpop(pop.size());
		for (int i = 0;i<pop.size();++i)
			subpop[i].init(pop.at(i));
		std::sort(subpop.begin(),subpop.end(),SortFit2());
		subpop.erase(unique(subpop.begin(),subpop.end(),sameFit2()),subpop.end());
		novelty = float(subpop.size())/float(pop.size());*/
		// novelty instead defined as the number of unique error vectors
		vector<ind> tmppop = pop;
		std::sort(tmppop.begin(),tmppop.end(),SortFit());
		tmppop.erase(unique(tmppop.begin(),tmppop.end(),sameOutput()),tmppop.end());
		novelty = float(tmppop.size())/float(pop.size());
	}
	void hom(vector<Randclass>& r, float& tot_hom, float& on_hom, float& off_hom)
	{
		tot_hom = 0; on_hom=0; off_hom=0;
		//float sum_strdist=0;
		int c1, c2, s_tot,s_on,s_off;
		float tot_tmp=0,on_tmp=0,off_tmp=0;
		int samplesize=200;
		std::string tmp1, tmp2, tmp1on, tmp2on, tmp1off, tmp2off;
		//std::string tmp2;
		for (int i=0; i<samplesize; ++i)
		{
			//reset temporary strings
			tmp1.resize(0); tmp2.resize(0);
			tmp1on.resize(0); tmp2on.resize(0);
			tmp1off.resize(0); tmp2off.resize(0);
			tot_tmp = 0; on_tmp = 0; off_tmp = 0;
			c1 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			c2 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);

			for (int j=pop.at(c1).line.size()-1; j>=0;--j){
				if (pop.at(c1).line.at(j).type=='v') tmp1 += pop.at(c1).line.at(j).varname;
				else tmp1 += pop.at(c1).line.at(j).type;

				if(pop.at(c1).line.at(j).on){
					if (pop.at(c1).line.at(j).type=='v') tmp1on += pop.at(c1).line.at(j).varname;
					else tmp1on += pop.at(c1).line.at(j).type;
				}
				/*else
					tmp1on += ' ';*/

				if(!pop.at(c1).line.at(j).on){
					if (pop.at(c1).line.at(j).type=='v') tmp1off += pop.at(c1).line.at(j).varname;
					else tmp1off += pop.at(c1).line.at(j).type;
				}
				/*else
					tmp1off += ' ';*/
			}

			for (int j=pop.at(c2).line.size()-1; j>=0;--j){
				if (pop.at(c2).line.at(j).type=='v') tmp2 += pop.at(c2).line.at(j).varname;
				else tmp2 += pop.at(c2).line.at(j).type;

				if(pop.at(c2).line.at(j).on){
					if (pop.at(c2).line.at(j).type=='v') tmp2on += pop.at(c2).line.at(j).varname;
					else tmp2on += pop.at(c2).line.at(j).type;
				}
				/*else
					tmp2on += ' ';*/

				if(!pop.at(c2).line.at(j).on){
					if (pop.at(c2).line.at(j).type=='v') tmp2off += pop.at(c2).line.at(j).varname;
					else tmp2off += pop.at(c2).line.at(j).type;
				}
				/*else
					tmp2off += ' ';*/
			}

			s_tot = strdist(tmp1,tmp2);
			s_on = strdist(tmp1on,tmp2on);
			//s_off = s_tot-s_on;
			if (tmp1off.size()>0 && tmp2off.size()>0) s_off = strdist(tmp1off,tmp2off);
			else s_off = 13785;

			tot_tmp = float(s_tot)/float(std::max(tmp1.size(),tmp2.size()));
			on_tmp  = float(s_on)/float(std::max(tmp1on.size(),tmp2on.size()));
			if (s_off!= 13785) off_tmp = float(s_off)/float(std::max(tmp1off.size(),tmp2off.size()));
			else off_tmp = 1;

			tot_hom += tot_tmp;
			on_hom += on_tmp;
			off_hom += off_tmp;
		}

		tot_hom = 1-tot_hom/samplesize;
		on_hom = 1-on_hom/samplesize;
		off_hom = 1-off_hom/samplesize;

	}
	/*float on_hom(vector<Randclass>& r){
		float sum_strdist=0;
		int c1, c2;
		int samplesize = 100;
		std::string tmp1;
		std::string tmp2;
		for (int i=0; i<samplesize; ++i)
		{
			c1 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			c2 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);

			for(int j=pop.at(c1).line.size()-1; j>=0;--j){
				if(pop.at(c1).line.at(j).on)
					tmp1 += pop.at(c1).line.at(j).type;
				else
					tmp1 += ' ';
			}
			for (int j=pop.at(c2).line.size()-1; j>=0;--j){
				if(pop.at(c2).line.at(j).on)
					tmp2 += pop.at(c2).line.at(j).type;
				else
					tmp2 += ' ';
			}
			sum_strdist += strdist(tmp1,tmp2)/float(std::max(tmp1.size(),tmp2.size()));
		}

		return 1-sum_strdist/samplesize;
	}
	float off_hom(vector<Randclass>& r){
		float sum_strdist=0;
		int c1, c2;
		int samplesize=100;
		std::string tmp1;
		std::string tmp2;
		for (int i=0; i<samplesize; ++i)
		{
			c1 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
			c2 = r[omp_get_thread_num()].rnd_int(0,pop.size()-1);

			for(int j=pop.at(c1).line.size(); j>0;--j){
				if(!pop.at(c1).line.at(j-1).on)
					tmp1 += pop.at(c1).line.at(j-1).type;
				else
					tmp1 += ' ';
			}
			for (int j=pop.at(c2).line.size(); j>0;--j){
				if(!pop.at(c2).line.at(j-1).on)
					tmp2 += pop.at(c2).line.at(j-1).type;
				else
					tmp2 += ' ';
			}
			sum_strdist += strdist(tmp1,tmp2)/float(std::max(tmp1.size(),tmp2.size()));
		}

		return 1-sum_strdist/samplesize;
	}*/
	/*
private:

	bool fitlow (ind& i,ind& j) { return (i.fitness<j.fitness); }
	bool eqncomp(ind& i,ind& j) { return (i.eqn_form.compare(j.eqn_form)==0); }
	bool fitcomp (ind& i,ind& j) { return (i.fitness==j.fitness); }
	*/

private:
		float maxf;
		float minf;


};


#endif
