#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "data.h"
#include "rnd.h"
#include "state.h"
#include "Line2Eqn.h"
#include "EvalEqnStr.h"
#include <unordered_map>
#include <bitset>
#include <math.h>
#include "matrix.h"
#include <Eigen/Dense>
#include "general_fns.h"
using Eigen::MatrixXf;
using Eigen::ArrayXf;
//#include "runEllenGP.h"
#define MAX_FLOAT numeric_limits<float>::max( )

#include "FitnessEstimator.h"
//#include "Fitness.h"

class CException
{
public:
	char* message;
	CException( char* m ) { message = m; };
	void Report(){cout << "error calculating fitness\n";};

};


const int bitlen( int n )
{  //return bitstring length needed to represent n
    // log(n)/log(2) is log2.
    return ceil(log( double(n) ) / log( float(2.0) ));
}
void CalcFitness(ind& me, params& p, vector<vector<float>>& vals, vector<float>& dattovar, vector<float>& target, state& s, bool pass);
bool CalcOutput(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s);
void CalcClassOutput(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s);
void Calc_M3GP_Output(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s);
float getCorr(vector<float>& output,vector<float>& target,float meanout,float meantarget,int off,float& target_std)
{
	float v1,v2;
	float var_target=0;
	float var_ind = 0;
	float q=0;
	float ndata = float(output.size());
	float corr;

	//calculate correlation coefficient
	for (unsigned int c = 0; c<output.size(); ++c)
	{

		v1 = target.at(c+off)-meantarget;
		v2 = output.at(c)-meanout;

		q += v1*v2;
		var_target+=pow(v1,2);
		var_ind+=pow(v2,2);
	}
	q = q/(ndata-1); //unbiased esimator
	var_target=var_target/(ndata-1); //unbiased esimator
	var_ind =var_ind/(ndata-1); //unbiased esimator
	if(abs(var_target)<0.0000001 || abs(var_ind)<0.0000001)
		corr = 0;
	else{
		corr = pow(q,2)/(var_target*var_ind);
		//corr = pow(q,2)/(sqrt(var_target)*var_ind); //for normalizing the error by the standard deviation of the target
	}
	target_std = sqrt(var_target);
	return corr;
}
void getCorr_lex(vector<float>& output, vector<float>& target, float meanout, float meantarget, vector<float>& err_lex)
{
	vector<float> v1, v2;
	float var_target = 0;
	float var_ind = 0;
	float q = 0;
	float ndata = float(output.size());
	float corr,cov=0;
	bool combo;

	if (err_lex.empty())
		combo = false;
	else
		combo = true;

	//calculate correlation coefficient
	for (unsigned int c = 0; c < output.size(); ++c)
	{
		v1.push_back(target[c] - meantarget);
		v2.push_back(output[c] - meanout);
		var_target += pow(v1[c], 2);
		var_ind += pow(v2[c], 2);
		//cov += v1[c]*v2[c];
	}
	//var_target = var_target / (ndata - 1); //unbiased esimator
	//var_ind = var_ind / (ndata - 1); //unbiased esimator
	for (unsigned int c = 0; c < output.size(); ++c)
	{
		if (combo) {
			//err_lex[c] /= std::max(pow(v1[c]*v2[c]/(ndata-1), 2) / (var_target*var_ind), float(0.00000001));
			err_lex[c] /= std::max(float((v1[c] * v2[c]) / (sqrt(var_target)*sqrt(var_ind))), float(0.0000001));
		}
		else
			//err_lex.push_back(1-pow(v1[c] * v2[c] / (ndata - 1), 2) / (var_target*var_ind));
			err_lex.push_back(1- float((v1[c] * v2[c]) / (sqrt(var_target)*sqrt(var_ind))));
		//tmp.push_back(float(pow(v1[c] * v2[c], 2) / (var_target*var_ind)));
	}

	//corr = accumulate(tmp.begin(), tmp.end(), 0.0);
	/*corr = pow(cov, 2) / (var_target*var_ind);
	if (corr > 0.1)
		cout << "pause\n";*/

}
float VAF_loud(vector<float>& output,vector<float>& target,float meantarget,int off,state& s)
{
	float v1,v2;
	float var_target=0;
	float var_diff = 0;
	float q=0;
	float ndata = float(output.size());
	float vaf=0;
//	float diff;
	float diffmean=0;
	s.out << "output.size() = " << output.size() << "\n";
	s.out << "target.size() = " << target.size() << "\n";

	for (unsigned int c = 0; c<output.size(); ++c)
		diffmean += target.at(c+off)-output.at(c);
	diffmean = diffmean/output.size();
	s.out << "diffmean = " << diffmean << "\n";
	//calculate correlation coefficient
	s.out << "var_diff: ";
	for (unsigned int c = 0; c<output.size(); ++c)
	{
		v1 = target.at(c+off)-meantarget;
		v2 = (target.at(c+off)-output.at(c))-diffmean;
		var_target+=pow(v1,2);
		var_diff+=pow(v2,2);
		s.out << var_diff << "\t";
	}
	//q = q/(ndata-1); //unbiased estimator
	var_target=var_target/(ndata-1); //unbiased estimator
	var_diff =var_diff/(ndata-1); //unbiased estimator
	s.out << "\nvar_diff = " << var_diff << "\n";
	s.out << "var_target = " << var_target << "\n";
	s.out << "var_diff / var_target = " << var_diff/var_target << "\n";
	if(var_target<0.0000001)
		return 0;
	else{
		float tmp = (1-var_diff/var_target)*100;
		s.out << "tmp = " << tmp << "\n";
		vaf = std::max(float(0),tmp);
		s.out << "vaf = " << vaf << "\n";
	}
	return vaf;
}
float VAF(vector<float>& output,vector<float>& target,float meantarget,int off)
{
	if (*min_element(output.begin(),output.end())==*max_element(output.begin(),output.end()))
		return 0;

	float v1,v2;
	float var_target=0;
	float var_diff = 0;
	float q=0;
	float ndata = float(output.size());
	float vaf=0;
//	float diff;
	float diffmean=0;

	for (unsigned int c = 0; c<output.size(); ++c)
		diffmean += target.at(c+off)-output.at(c);
	diffmean = diffmean/output.size();
	//calculate correlation coefficient
	for (unsigned int c = 0; c<output.size(); ++c)
	{
		v1 = target.at(c+off)-meantarget;
		v2 = (target.at(c+off)-output.at(c))-diffmean;
		var_target+=pow(v1,2);
		var_diff+=pow(v2,2);
	}
	//q = q/(ndata-1); //unbiased estimator
	var_target=var_target/(ndata-1); //unbiased estimator
	var_diff =var_diff/(ndata-1); //unbiased estimator
	if(var_target<0.0000001)
		return 0;
	else{
		float tmp = (1-var_diff/var_target)*100;
		vaf = std::max(float(0),tmp);
	}
	return vaf;
}
float std_dev(vector<float>& target,float& meantarget)
{
	float s=0;
	//calculate correlation coefficient
	for (unsigned int c = 0; c<target.size(); ++c)
	{
		s+=pow(abs(target[c]-meantarget),2);
	}
	s = s/(target.size()-1);
	return sqrt(s);
}

void eval_complexity(const node& n, vector<int>& c_float, vector<int>& c_bool)
{
	//float n1, n2;
	//bool b1, b2;
	if (c_float.size() >= n.arity_float && c_bool.size() >= n.arity_bool) {
		int c_f = 0;
		int c_b = 0;

		if (n.type == 'n' || n.type == 'v')
			c_float.push_back(n.c);
		else {
			for (size_t i = 0; i < n.arity_float; ++i) {
				c_f += c_float.back();
				c_float.pop_back();
			}
			for (size_t i = 0; i < n.arity_bool; ++i) {
				c_b += c_bool.back();
				c_bool.pop_back();
			}
			if (n.return_type == 'f')
				c_float.push_back(n.c*(c_f + c_b + 1));
			else
				c_bool.push_back(n.c*(c_f + c_b + 1));
		}
	}

}
int getComplexity(ind& me, params& p)
{
	int complexity = 0;

	vector<int> c_float;
	vector<int> c_bool;

	for (size_t i = 0; i < me.line.size(); ++i)
		if (me.line[i].on) eval_complexity(me.line[i], c_float, c_bool);

	if (c_float.empty())
		complexity = p.max_fit;
	else if (p.classification && p.class_m4gp)
		complexity = accumulate(c_float.begin(), c_float.end(), 0);
	else
		complexity = c_float.back();

	return complexity;
}

int getEffSize(vector<node>& line)
{
	int eff_size=0;
	for(int m=0;m<line.size();++m){
		if(line.at(m).on)
			++eff_size;
	}
	return eff_size;
}
void eval(node& n,vector<float>& stack_float,vector<bool>& stack_bool)
{
	float n1,n2;
	bool b1,b2;
	if (stack_float.size()>=n.arity_float && stack_bool.size()>=n.arity_bool){
		n.intron=false;
		switch(n.type)
		{
		case 'n':
			stack_float.push_back(n.value);

			break;
		case 'v':
			stack_float.push_back(*n.valpt);
			if (boost::math::isnan(stack_float.back()))
				cout << "nans in variable computation with input " << n.varname << "\n";
			break;
		case '+':
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_float.push_back(n1+n2);
			break;
		case '-':
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_float.push_back(n1-n2);
			break;
		case '*':
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_float.push_back(n1*n2);
			break;
		case '/':
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			if(abs(n2)<0.000001)
				stack_float.push_back(1);
			else
				stack_float.push_back(n1/n2);
			break;
		case 's':
			n1 = stack_float.back(); stack_float.pop_back();
			stack_float.push_back(sin(n1));
			break;
		case 'c':
			n1 = stack_float.back(); stack_float.pop_back();
			stack_float.push_back(cos(n1));
			break;
		case 'e':
			n1 = stack_float.back(); stack_float.pop_back();
			stack_float.push_back(exp(n1));
			if (boost::math::isnan(stack_float.back()))
				cout << "nans in exp computation with input " << n1 << "\n";
			break;
		case 'l':
			n1 = stack_float.back(); stack_float.pop_back();
			if (abs(n1)<0.000001)
				stack_float.push_back(0);
			else
				// safe log of absolute value of n1
				stack_float.push_back(log(abs(n1)));
				// unsafe log of real value
				//stack_float.push_back(log(n1));
				// check
				if (boost::math::isnan(stack_float.back()))
					cout << "nans in log computation with input " << n1 << "\n";
			break;
		case 'q':
			n1 = stack_float.back(); stack_float.pop_back();
			// safe sqrt of absolute value of n1
			stack_float.push_back(sqrt(abs(n1)));
			break;
		case '2':
			n1 = stack_float.back(); stack_float.pop_back();
			// square n1
			stack_float.push_back(pow(n1,2));
			break;
		case '3':
			n1 = stack_float.back(); stack_float.pop_back();
		    // cube  n1
			stack_float.push_back(pow(n1,3));
			break;
		case '^':
			n1 = stack_float.back(); stack_float.pop_back();
			// safe sqrt of absolute value of n1
			stack_float.push_back(pow(n1,n2));
			break;
		case '=': // equals
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_bool.push_back(n1==n2);
			break;
		case '!': // does not equal
			b1 = stack_bool.back(); stack_bool.pop_back();
			stack_bool.push_back(!b1);
			break;
		case '>': //greater than
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_bool.push_back(n1 > n2);
			break;
		case '<': //less than
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_bool.push_back(n1 < n2);
			break;
		case '}': //greater than or equal
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_bool.push_back(n1 >= n2);
			break;
		case '{': //less than or equal
			n1 = stack_float.back(); stack_float.pop_back();
			n2 = stack_float.back(); stack_float.pop_back();
			stack_bool.push_back(n1 <= n2);
			break;
		case 'i': // if (arity 2). if stack_bool true, leave top element of floating stack. otherwise, pop it.
			b1 = stack_bool.back(); stack_bool.pop_back();
			if (!b1)
				stack_float.pop_back();
			break;
		case 't': // if-then-else (arity 3). if stack_bool true, leave 2nd element of floating stack and pop first. otherwise, pop 2nd and leave first.
			b1 = stack_bool.back(); stack_bool.pop_back();
			//n1 = stack_float.back(); stack_float.pop_back();
			//n2 = stack_float.back(); stack_float.pop_back();
			if (b1)
				stack_float.pop_back();
			else{
				swap(stack_float[stack_float.size()-2],stack_float.back());
				stack_float.pop_back();
			}
			break;
		case '&':
			b1 = stack_bool.back(); stack_bool.pop_back();
			b2 = stack_bool.back(); stack_bool.pop_back();
			stack_bool.push_back(b1 && b2);
			break;
		case '|':
			b1 = stack_bool.back(); stack_bool.pop_back();
			b2 = stack_bool.back(); stack_bool.pop_back();
			stack_bool.push_back(b1 || b2);
			break;
		default:
			cout << "eval error\n";
			break;
		}
	}
	else
		n.intron= n.intron && true; // only set it to intron if it isn't used in any of the execution

	if (!stack_float.empty()) {
		if (boost::math::isinf(abs(stack_float.back())))
			stack_float[stack_float.size() - 1] = MAX_FLOAT;

		if (boost::math::isnan(stack_float.back()))
		 	stack_float[stack_float.size() - 1] = 0;
	}

}

void FitnessEstimate(vector<ind>& pop,params& p,Data& d,state& s,FitnessEstimator& FE);
void StandardFitness(ind& me,params& p,Data& d,state& s,FitnessEstimator& FE, unordered_map<string,float*>& datatable, vector<float>& dattovar);
void LexicaseFitness(ind& me,params& p,Data& d,state& s,FitnessEstimator& FE);

void Fitness(vector<ind>& pop,params& p,Data& d,state& s,FitnessEstimator& FE)
{
	////
	//set up data table for conversion of symbolic variables
	unordered_map <string,float*> datatable;
	vector<float> dattovar(d.label.size());

	for (unsigned int i=0;i<d.label.size(); ++i)
			datatable.insert(pair<string,float*>(d.label[i],&dattovar[i]));

	//#pragma omp parallel for private(e)
	for(int count = 0; count<pop.size(); ++count)
	{
		if (p.print_protected_operators){
			pop.at(count).eqn_matlab = Line2Eqn(pop.at(count).line,pop.at(count).eqn_form,p,true);
			pop.at(count).eqn = Line2Eqn(pop.at(count).line,pop.at(count).eqn_form,p,false);
		}
		else
			pop.at(count).eqn = Line2Eqn(pop.at(count).line,pop.at(count).eqn_form,p,true);
		//getEqnForm(pop.at(count).eqn,pop.at(count).eqn_form);

		pop.at(count).eff_size = getEffSize(pop.at(count).line);

		//if(p.sel!=3){
		StandardFitness(pop.at(count),p,d,s,FE,datatable,dattovar);

		if (p.classification && p.class_m4gp)
			pop[count].dim = pop[count].M.cols();
		//} // if p.sel!=3
		//else //LEXICASE FITNESS
		//{
		//	LexicaseFitness(pop.at(count),p,d,s,FE);
		//}//LEXICASE FITNESS
		if (pop[count].eqn.compare("unwriteable")==0)
			pop[count].complexity=p.max_fit;
		else
			pop.at(count).complexity= getComplexity(pop.at(count),p);
		/*if (p.estimate_generality && pop.at(count).genty != abs(pop[count].fitness-pop[count].fitness_v)/pop[count].fitness && pop.at(count).genty != p.max_fit)
			std::cerr << "genty error, line 300 Fitness.cpp\n";*/
	}//for(int count = 0; count<pop.size(); ++count)
	s.numevals[omp_get_thread_num()]=s.numevals[omp_get_thread_num()]+pop.size();
	//cout << "\nFitness Time: ";
}
void StandardFitness(ind& me,params& p,Data& d,state& s,FitnessEstimator& FE,unordered_map<string,float*>& datatable, vector<float>& dattovar)
{

	me.abserror = 0;
	me.abserror_v = 0;
	me.sq_error = 0;
	me.sq_error_v = 0;
	// set data table and pointers to data in program nodes
	for(int m=0;m<me.line.size();++m){
		if(me.line.at(m).type=='v')
			{// set pointer to dattovar
				//float* set = datatable.at(static_pointer_cast<n_sym>(me.line.at(m)).varname);
				float* set = datatable.at(me.line.at(m).varname);
				if(set==NULL)
					cout<<"hmm";
				/*static_pointer_cast<n_sym>(me.line.at(m)).setpt(set);*/
				me.line.at(m).setpt(set);
				/*if (static_pointer_cast<n_sym>(me.line.at(m)).valpt==NULL)
					cout<<"wth";*/
			}
	}
	//cout << "Equation" << count << ": f=" << me.eqn << "\n";
	bool pass=true;
	if(!me.eqn.compare("unwriteable")==0){

// calculate error
		if(!p.EstimateFitness){
			if(p.classification && p.class_m4gp)
				Calc_M3GP_Output(me,p,d.vals,dattovar,d.target,s);
			else if (p.classification)
				CalcClassOutput(me,p,d.vals,dattovar,d.target,s);
			else {
				pass = CalcOutput(me, p, d.vals, dattovar, d.target, s);
				CalcFitness(me, p, d.vals, dattovar, d.target, s, pass);
			}
		} // if not estimate fitness
		else{
// use fitness estimator subset of d.vals ===========================================================
			vector<vector<float>> FEvals;
			vector<float> FEtarget;
			setFEvals(FEvals,FEtarget,FE,d);
			if(p.classification && p.class_m4gp)  
                Calc_M3GP_Output(me,p,FEvals,dattovar,FEtarget,s);
			else if(p.classification) 
                CalcClassOutput(me,p,FEvals,dattovar,FEtarget,s);
			else {
                pass = CalcOutput(me,p,FEvals,dattovar,FEtarget,s);
                CalcFitness(me,p,FEvals,dattovar,FEtarget,s, pass);
            }
		} // if estimate fitness
		if (p.estimate_generality || p.PS_sel==2){
				if (me.fitness == p.max_fit || me.fitness_v== p.max_fit || boost::math::isnan(me.fitness_v) || boost::math::isinf(me.fitness_v))
					me.genty = p.max_fit;
				else{
					if (p.G_sel==1) // MAE
						me.genty = abs(me.abserror-me.abserror_v)/me.abserror;
					else if (p.G_sel==2) // R2
						me.genty = abs(me.corr-me.corr_v)/me.corr;
					else if (p.G_sel==3) // MAE R2 combo
						me.genty = abs(me.abserror/me.corr-me.abserror_v/me.corr_v)/(me.abserror/me.corr);
					else if (p.G_sel==3) // VAF
						me.genty = abs(me.VAF-me.VAF_v)/me.VAF;
				}
				if ( boost::math::isnan(me.genty) || boost::math::isinf(me.genty))
					me.genty=p.max_fit;
		}
	} // if not unwriteable equation
	else{ // bad equation, assign maximum fitness
		me.abserror=p.max_fit;
		me.sq_error = p.max_fit;
		me.corr = p.min_fit;
		me.VAF = p.min_fit;
		me.fitness = p.max_fit;
		me.fitness_v = p.max_fit;
		me.genty = p.max_fit;
		if (p.train){
			me.abserror_v=p.max_fit;
			me.sq_error_v = p.max_fit;
			me.corr_v = p.min_fit;
			me.VAF_v = p.min_fit;
		}
	}


}
void CalcFitness(ind& me, params& p, vector<vector<float>>& vals, vector<float>& dattovar, vector<float>& target, state& s, bool pass)
{
	float q = 0;
	float var_target = 0;
	float var_ind = 0;
	float meanout = 0;
	float meantarget = 0;
	float meanout_v = 0;
	float meantarget_v = 0;
	float target_std = 1000;
	float target_std_v = 1000;
	int sim_size = vals.size();
	unsigned int ndata_t, ndata_v; // training and validation data sizes
	if (p.train) {
		ndata_t = vals.size()*p.train_pct;
		ndata_v = vals.size() - ndata_t;
		if (p.test_at_end) // don't run the validation data
			sim_size = ndata_t;
	}
	else {
		ndata_t = vals.size();
		ndata_v = 0;
	}

	if (pass) {
		for (unsigned int sim = 0; sim<sim_size; ++sim)
		{

			if ((p.train && sim<ndata_t) || (!p.train)) {

				if (p.weight_error) {
					me.abserror += p.error_weight[sim] * abs(target.at(sim) - me.output.at(sim));
					me.sq_error += p.error_weight[sim] * pow(target.at(sim) - me.output.at(sim),2);
				}
				else {
					me.abserror += abs(target.at(sim) - me.output.at(sim));
					me.sq_error += pow(target.at(sim) - me.output.at(sim), 2);
				}

				// if (p.sel == 3 && ((p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0) || p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0) || p.fit_type.compare("MSE")) // lexicase error vector
				if (p.sel == 3 && !(p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0 || p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0))
					me.error.push_back(abs(target.at(sim) - me.output.at(sim)));

				meantarget += target.at(sim);
				meanout += me.output[sim];
			}
			else //validation set
			{

				if (p.weight_error) {
					me.abserror_v += p.error_weight[sim] * abs(target[sim] - me.output_v[sim - ndata_t]);
					me.sq_error_v += p.error_weight[sim] * pow(target[sim] - me.output_v[sim-ndata_t], 2);
				}
				else{
					me.abserror_v += abs(target[sim] - me.output_v[sim - ndata_t]);
					me.sq_error_v += pow(target[sim] - me.output_v[sim-ndata_t], 2);
				}

				meantarget_v += target.at(sim);
				meanout_v += me.output_v[sim - ndata_t];
			}

		}

		assert(me.output.size() == ndata_t);
		// mean absolute error
		me.abserror = me.abserror / ndata_t;
		me.sq_error = me.sq_error / ndata_t;
		meantarget = meantarget / ndata_t;
		meanout = meanout / ndata_t;
		// lexicase fitness
		if (p.sel == 3 && !(p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)) {
			if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0 || p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0) //correlation
				getCorr_lex(me.output, target, meanout, meantarget, me.error);
			//else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0 )
				//VAF_lex(me.output, target, meanout, meantarget, me.error);

		}
		//calculate correlation coefficient
		me.corr = getCorr(me.output, target, meanout, meantarget, 0, target_std);
		me.VAF = VAF(me.output, target, meantarget, 0);

		if (p.train && !p.test_at_end) // calc validation fitness
		{
			q = 0;
			var_target = 0;
			var_ind = 0;
			// mean absolute error
			me.abserror_v = me.abserror_v / ndata_v;
			me.sq_error_v /= ndata_v;
			meantarget_v = meantarget_v / ndata_v;
			meanout_v = meanout_v / ndata_v;
			//calculate correlation coefficient
			me.corr_v = getCorr(me.output_v, target, meanout_v, meantarget_v, ndata_t, target_std_v);
			me.VAF_v = VAF(me.output_v, target, meantarget_v, ndata_t);
		}
	}
	else {
		me.corr = 0;
		me.corr_v = 0;
		me.VAF = 0;
		me.VAF_v = 0;
		me.abserror = p.max_fit;
		me.sq_error = p.max_fit;
	}


	if (me.corr < p.min_fit)
		me.corr = p.min_fit;
	if (me.VAF < p.min_fit)
		me.VAF = p.min_fit;

	if (me.output.empty())
		me.fitness = p.max_fit;
	else if (boost::math::isnan(me.abserror) || boost::math::isinf(me.abserror) || boost::math::isnan(me.corr) || boost::math::isinf(me.corr))
		me.fitness = p.max_fit;
	else {
		if (p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)
			me.fitness = me.abserror;
		else if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0)
			me.fitness = 1 - me.corr;
		else if (p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0)
			me.fitness = me.abserror / me.corr;
		else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0)
			me.fitness = 1 - me.VAF / 100;
		else if (p.fit_type.compare("MSE")==0)
			me.fitness = me.sq_error;
		if (p.norm_error)
			me.fitness = me.fitness / target_std;
	}


	if (me.fitness>p.max_fit)
		me.fitness = p.max_fit;
	else if (me.fitness<p.min_fit)
		(me.fitness = p.min_fit);

	if (p.train && !p.test_at_end) { //assign validation fitness

		if (me.corr_v < p.min_fit)
			me.corr_v = p.min_fit;
		if (me.VAF_v < p.min_fit)
			me.VAF_v = p.min_fit;

		if (me.output_v.empty())
			me.fitness_v = p.max_fit;
		else if (boost::math::isnan(me.abserror_v) || boost::math::isinf(me.abserror_v) || boost::math::isnan(me.corr_v) || boost::math::isinf(me.corr_v))
			me.fitness_v = p.max_fit;
		else {
			if (p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)
				me.fitness_v = me.abserror_v;
			else if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0)
				me.fitness_v = 1 - me.corr_v;
			else if (p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0)
				me.fitness_v = me.abserror_v / me.corr_v;
			else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0)
				me.fitness_v = 1 - me.VAF_v / 100;
			else if (p.fit_type.compare("MSE")==0)
				me.fitness_v = me.sq_error_v;
			if (p.norm_error)
				me.fitness_v = me.fitness_v / target_std_v;
		}


		if (me.fitness_v>p.max_fit)
			me.fitness_v = p.max_fit;
		else if (me.fitness_v<p.min_fit)
			(me.fitness_v = p.min_fit);
	}
	else { // if not training, assign copy of regular fitness to the validation fitness variables
		me.corr_v = me.corr;
		me.abserror_v = me.abserror;
		me.sq_error_v = me.sq_error;
		me.VAF_v = me.VAF;
		me.fitness_v = me.fitness;
	}

}
bool CalcOutput(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s)
{
	vector<float> stack_float;
	vector<bool> stack_bool;
	me.reset_introns();
	me.output.resize(0);
	me.output_v.resize(0);

	if (p.sel==3)
		me.error.resize(0);

	float q = 0;
	float var_target = 0;
	float var_ind = 0;
	float meanout=0;
	float meantarget=0;
	float meanout_v=0;
	float meantarget_v=0;
	float target_std=1000;
	float target_std_v=1000;
	int ptevals=0;
	int sim_size = vals.size();
	unsigned int ndata_t,ndata_v; // training and validation data sizes
	if (p.train){
		ndata_t = vals.size()*p.train_pct;
		ndata_v = vals.size()-ndata_t;
		if (p.test_at_end) // don't run the validation data
			sim_size = ndata_t;
	}
	else{
		ndata_t = vals.size();
		ndata_v=0;
	}
	//reserve memory for output
	me.output.reserve(ndata_t);
	me.output_v.reserve(ndata_v);

	if (p.eHC_on && p.eHC_slim)
	{
		me.stack_float.resize((me.eff_size)*vals.size());
		me.stack_floatlen.resize(0);
		me.stack_floatlen.reserve(me.eff_size);
	}

	 // loop over data
	for(unsigned int sim=0;sim<sim_size;++sim)
		{
			int k_eff=0;

			for (unsigned int j=0; j<p.allvars.size()-p.AR_na;++j) //wgl: add time delay of output variable here
				dattovar.at(j) = vals[sim][j]; // can we replace this with a pointer to the data so we don't have to copy the whole thing every time?
			if (p.AR){ // auto-regressive output variables
				int ARstart = p.allvars.size()-p.AR_na;
				for (unsigned int h=0; h<p.AR_na; ++h){
					if (sim<ndata_t){
						if (me.output.size() >= h + p.AR_nka) // add distinction for training / validation data
							if (p.AR_lookahead) dattovar[ARstart+h] = target[sim-h-p.AR_nka];
							else dattovar[ARstart+h] = me.output[sim-h - p.AR_nka];
						else dattovar[ARstart+h] = 0;
					}
					else{
						if (me.output_v.size() >= h + p.AR_nka) // add distinction for training / validation data
							if (p.AR_lookahead) dattovar[ARstart+h] = target[sim-h - p.AR_nka];
							else dattovar[ARstart+h] = me.output_v[sim-h-ndata_t - p.AR_nka];
						else dattovar[ARstart+h] = 0;
					}
				}
			}
			// evaluate program
			for(int k=0;k<me.line.size();++k){

				if (me.line.at(k).on){
					//me.line.at(k).eval(stack_float);
					eval(me.line.at(k),stack_float,stack_bool);

					++ptevals;
					if (p.eHC_on && p.eHC_slim) // stack tracing
					{
						if(!stack_float.empty()) me.stack_float[k_eff*vals.size() + sim] = stack_float.back();
						else me.stack_float[k_eff*vals.size() + sim] = 0;
						if (sim==0) me.stack_floatlen.push_back(stack_float.size());

					}
					++k_eff;
				}
			} // program for loop

			if (stack_float.empty()) //stack_float empty check
				return false;
			else{ // push top of stack to output
				if ((p.train && sim < ndata_t) || (!p.train)) {
					me.output.push_back(stack_float.back());
					if (boost::math::isnan(abs(me.output[sim])))
						cout << "nans in output\n";
				}
				else //validation set
					me.output_v.push_back(stack_float.back());


			}
			//reset stacks
			stack_float.resize(0);
			stack_bool.resize(0);
		} // data for loop

		// set point evaluations
		s.ptevals[omp_get_thread_num()]=s.ptevals[omp_get_thread_num()]+ptevals;

		return true;
}
void Calc_M3GP_Output(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s)
{
	vector<float> stack_float;
	vector<bool> stack_bool;

	me.reset_introns();
	me.output.resize(0);
	me.output_v.resize(0);
	if (p.sel==3){
		if (p.lex_class)
			me.error.assign(p.number_of_classes,0);
		else
			me.error.resize(0);
	}
	else if (p.sel==4 && p.PS_sel>=4) // each class is an objective
		me.error.assign(p.number_of_classes,0);

	float var_target = 0;
	float var_ind = 0;
	bool pass = true;
	int ptevals=0;
	int sim_size = vals.size();
	unsigned int ndata_t,ndata_v; // training and validation data sizes
	if (p.train){
		ndata_t = vals.size()*p.train_pct;
		ndata_v = vals.size()-ndata_t;
		if (p.test_at_end) // don't run the validation data
			sim_size = ndata_t;
	}
	else{
		ndata_t = vals.size();
		ndata_v=0;
	}
	me.output.reserve(ndata_t);
	me.output_v.reserve(ndata_v);
	//if (p.eHC_on && p.eHC_slim){ me.stack_float.resize(0); me.stack_float.reserve((me.eff_size)*vals.size());}
	if (p.eHC_on && p.eHC_slim)
	{
		me.stack_float.resize((me.eff_size)*vals.size());
		me.stack_floatlen.resize(0);
		me.stack_floatlen.reserve(me.eff_size);
	}
	//stack_float.reserve(me.eff_size/2);

	//vector<vector<float>> Z; // n_t x d output for m3gp
	MatrixXf Z(ndata_t,1);
	//vector<vector<float>> Z_v; // n_v x d output for m3gp
	MatrixXf Z_v(ndata_v,1);
	vector<MatrixXf> K(p.number_of_classes); // output subsets by class label

	// Calculate Output
	for(unsigned int sim=0;sim<sim_size;++sim)
		{
			//if (p.eHC_slim) me.stack_float.push_back(vector<float>());
			int k_eff=0;

			for (unsigned int j=0; j<p.allvars.size();++j)
				dattovar.at(j)= vals[sim][j];

			for(int k=0;k<me.line.size();++k){
				if (me.line.at(k).on){
					eval(me.line.at(k),stack_float,stack_bool);
					++ptevals;
					if (p.eHC_on && p.eHC_slim) // stack tracing
					{
						if(!stack_float.empty()) me.stack_float[k_eff*vals.size() + sim] = stack_float.back();
						else me.stack_float[k_eff*vals.size() + sim] = 0;
						if (sim==0) me.stack_floatlen.push_back(stack_float.size());
						/*if(!stack_float.empty()) me.stack_float.push_back(stack_float.back());
						else me.stack_float.push_back(0);*/
					}
					++k_eff;
					/*else
						cout << "hm";*/
				}
			}
			/*if (stack_float.size()>1)
				cout << "non-tree\n";*/

			//if(!(!p.classification && stack_float.empty()) && !(p.classification && stack_bool.empty())){
			if(!stack_float.empty()){
				Z.resize(ndata_t,stack_float.size());
				if ((p.train && sim<ndata_t) || (!p.train)){

							//Z.push_back(vector<float>());
							for (int z =0;z<stack_float.size();++z){
								//Z[sim].push_back(stack_float[z]);
								Z(sim,z) = stack_float[z];
								//K[target[sim]].push_back(stack_float[z]);
							}
							//VectorXf Kadd;

							/*cout << K[target[sim]],
								         Z.row(sim);*/
							K[target[sim]].resize(K[target[sim]].rows(),Z.cols());

					//		cout << "K size: " << K[target[sim]].rows() << " x " <<  K[target[sim]].cols() << endl;

					//		cout << "K new size: " << K[target[sim]].rows() << " x " <<  K[target[sim]].cols() << endl;
					//		cout << "Z: " << Z.row(sim) << endl;
							MatrixXf Kadd(K[target[sim]].rows()+1,Z.cols());
							Kadd << K[target[sim]],
								    Z.row(sim);
					//		cout << "Kadd: " << Kadd <<endl;
							//K[target[sim]].bottomRows(1) = Z.row(sim);
							K[target[sim]].swap(Kadd);
							//K[target[sim]].set( (MatrixXd(K[target[sim]].rows()+1,Z) << mat, vec.transpose()).finished() );
					//		cout << "K: " << K[target[sim]] << endl;
							//K[target[sim]].set( (MatrixXf(
							//K[target[sim]] << K[target[sim]],
								              //Z.row(sim);
			// for multiclass classification, error should be based solely on whether or not the right class was assigned

					}
					else //validation set
					{
						Z_v.resize(ndata_v,stack_float.size());
						//Z_v.push_back(vector<float>());
						for (int z =0;z<stack_float.size();++z)
							//Z_v[sim-ndata_t].push_back(stack_float[z]);
							Z_v(sim-ndata_t,z) = stack_float[z];

					}
			}
			else{
				pass=false;
				break;
				}
			//stack_float.clear();
			stack_float.resize(0);
			stack_bool.resize(0);
		}
	// Calculate Covariance and Centroids of model output data by class
	/////////////////////////////////////////////////////////////////////
	VectorXf D(p.number_of_classes);
	VectorXf D_v(p.number_of_classes);

	// True postives, false positives, false negatives
	vector<float> TP(p.number_of_classes, 0);
	vector<float> FP(p.number_of_classes, 0);
	vector<float> FN(p.number_of_classes, 0);

	vector<float> TP_v(p.number_of_classes, 0);
	vector<float> FP_v(p.number_of_classes, 0);
	vector<float> FN_v(p.number_of_classes, 0);

	if (pass) {
		bool pass2 = true; // pass check for invertible covariance matrix
		me.M.resize(p.number_of_classes, Z.cols());

		std::vector<MatrixXf> Cinv(p.number_of_classes, MatrixXf(Z.cols(), Z.cols()));
		for (int i = 0; i < p.number_of_classes; ++i) {
			me.C.push_back(MatrixXf(Z.cols(), Z.cols()));
			//me.M.push_back(vector<float>());
			//cout << K[i].colwise().mean() << endl;
			me.M.row(i) << K[i].colwise().mean();
			//cout << "M centroids for class " << i << ":\n" << me.M.row(i) << endl;
			/*s.out << "/////////////////////////////\n";
			s.out << "K(" << i << ")\n";
			s.out << K[i] << "\n";*/

			cov(K[i], me.C[i]);
			Eigen::FullPivLU<MatrixXf> check(me.C[i]);
			if (check.isInvertible()) {
				//s.out << "C: \n";
				//s.out << me.C[i] << "\n";
				Cinv[i] = me.C[i].inverse();
				//s.out << "C^-1:\n";
				//s.out << Cinv[i] << "\n";
			}
			else {
				//s.out << "C: \n";
				//s.out << me.C[i] << "\n";
				//Cinv[i] = me.C[i].inverse();
				Cinv[i] = MatrixXf::Identity(Cinv[i].rows(), Cinv[i].cols());
				//s.out << "C^-1:\n";
				//s.out << Cinv[i] << "\n";
				//pass=false;
				//pass2=false;
			}



		}
		//vector<float> D; // mahalanobis distance on training set
		//vector<float> D_v; // mahalanobis distance on validation set

		// Calculate Fitness
		////////////////////////////////////////////////////////////////////////


		if (pass2) {


			for (int sim = 0; sim < sim_size; ++sim) {
				if ((p.train && sim < ndata_t) || (!p.train)) {
					//D.resize(0);
					//training set mahalanobis distance

					MahalanobisDistance(Z.row(sim), Cinv, me.M, D, s);


					/*s.out << "Z:\n";
					s.out << Z.row(sim) << "\n";
					s.out << "D:\n";
					s.out << D << "\n";*/
					//assign class based on minimum distance
					//vector<float>::iterator it = min_element(D.begin(),D.end());
					MatrixXf::Index min_i;
					float min = D.minCoeff(&min_i);
					//float ans = min_i;
					me.output.push_back(float(min_i));

					// assign error

					if (target[sim] != me.output[sim]) {
						++me.abserror;
						++FP[me.output[sim]]; // false positives
						++FN[target[sim]]; // false negatives
						if (p.sel == 3) { // lexicase error vector
							if (p.lex_class)
								++me.error[target[sim]];
							else
								me.error.push_back(1);
						}
						else if (p.sel == 4 && p.PS_sel >= 4) // each class is an objective
							++me.error[target[sim]];

					}
					else {
						++TP[target[sim]]; // true positives
						if (p.sel == 3 && !p.lex_class)
							me.error.push_back(0);
					}


				}
				else {
					//D_v.resize(0);
					//validation set mahalanobis distance
					MahalanobisDistance(Z_v.row(sim - ndata_t), Cinv, me.M, D_v, s);

					//assign class based on minimum distance in D_v[sim]
					//vector<float>::iterator it = min_element(D_v.begin(),D_v.end());
					//me.output_v.push_back(float(it - D_v.begin()));
					MatrixXf::Index min_i;
					float min = D_v.minCoeff(&min_i);
					//float ans = (min_i);
					me.output_v.push_back(float(min_i));

					if (target[sim] != me.output_v[sim - ndata_t]) {
						++me.abserror_v;
						++FP_v[me.output_v[sim-ndata_t]]; // false positives
						++FN_v[target[sim]]; // false negatives
					}
					else
						++TP_v[target[sim]]; // true positives


				}
			}
		} // if pass2

		/////////////////////////////////////////////////////////////////////////////////

		assert(me.output.size() == ndata_t);
		// mean absolute error
		me.abserror = me.abserror / ndata_t;

		if (p.train && !p.test_at_end)//mean absolute error
			me.abserror_v = me.abserror_v / ndata_v;


	} // if pass
	else {
		me.abserror = p.max_fit;
	}


	if(me.output.empty())
		me.fitness=p.max_fit;
	else if ( boost::math::isnan(me.abserror) || boost::math::isinf(me.abserror) )
		me.fitness=p.max_fit;
	else{
		if (p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0){
			me.fitness = me.abserror;
		}
		else if (p.fit_type.compare("2")==0 || p.fit_type.compare("F1W")==0) {
			float precision, recall;
			me.fitness = 0;
			for (unsigned int i = 0; i < p.number_of_classes; ++i) {
				if (TP[i] + FP[i] == 0)
					precision = 0;
				else
					precision = TP[i] / (TP[i] + FP[i]);

				if (TP[i] + FN[i] == 0)
					recall = 0;
				else
					recall = TP[i] / (TP[i] + FN[i]);

				if (recall + precision != 0)
					me.fitness += 2 * p.class_w[i] * (precision*recall) / (precision + recall);
			}
			me.fitness = 1 - me.fitness;
		}
		else if (p.fit_type.compare("3") == 0 || p.fit_type.compare("F1")==0) {

			float precision, recall;
			me.fitness = 0;
			for (unsigned int i = 0; i < p.number_of_classes; ++i) {
				if (TP[i] + FP[i] == 0)
					precision = 0;
				else
					precision = TP[i] / (TP[i] + FP[i]);

				if (TP[i] + FN[i] == 0)
					recall = 0;
				else
					recall = TP[i] / (TP[i] + FN[i]);

				if (recall + precision != 0)
					me.fitness += 2 * (precision*recall) / (precision + recall) / p.number_of_classes;
			}

			me.fitness = 1 - me.fitness;
			if (me.fitness<0.01){
				cout << "precision: " << precision << "\n";
				cout << "recall: " << recall << "\n";
				cout << "fitness: " << me.fitness << "\n";
				cout << "TP: ";
				for (auto i : TP)
					cout << i << ",";
				cout << "FP: ";
				for (auto i : FP)
					cout << i << ",";
				cout << "FN: ";
				for (auto i : TP)
					cout << i << ",";
				cout << "\n";

			}
		}
		/*if (p.norm_error)
			me.fitness = me.fitness/target_std;*/
	}

	for (unsigned z=0;z<D.size();++z){
		if(!boost::math::isfinite(D(z)))
			me.fitness = p.max_fit;
	}
	if(me.fitness>p.max_fit)
		me.fitness=p.max_fit;
	else if(me.fitness<p.min_fit)
		(me.fitness=p.min_fit);
	else if (boost::math::isnan(me.fitness))
		me.fitness = p.max_fit;

	if(p.train && !p.test_at_end){ //assign validation fitness

		for (unsigned z=0;z<D_v.size();++z){
			if(!boost::math::isfinite(D_v(z)))
				me.fitness_v = p.max_fit;
		}

		if(me.output_v.empty())
			me.fitness_v=p.max_fit;
		/*else if (*std::max_element(me.output_v.begin(),me.output_v.end())==*std::min_element(me.output_v.begin(),me.output_v.end()))
			me.fitness_v=p.max_fit;*/
		else if ( boost::math::isnan(me.abserror_v) || boost::math::isinf(me.abserror_v))
			me.fitness_v=p.max_fit;
		else{
			if (p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0) {
				me.fitness_v = me.abserror_v;
			}
			else if (p.fit_type.compare("2")==0 || p.fit_type.compare("F1W")==0) {
				float precision, recall;
				me.fitness_v = 0;
				for (unsigned int i = 0; i < p.number_of_classes; ++i) {

					if (TP_v[i] + FP_v[i] == 0)
						precision = 0;
					else
						precision = TP_v[i] / (TP_v[i] + FP_v[i]);

					if (TP_v[i] + FN_v[i] == 0)
						recall = 0;
					else
						recall = TP_v[i] / (TP_v[i] + FN_v[i]);

					if (recall + precision != 0)
						me.fitness_v += 2 * p.class_w_v[i] * (precision*recall) / (precision + recall);
				}
				me.fitness_v = 1 - me.fitness_v;
			}
			else if (p.fit_type.compare("3") == 0 || p.fit_type.compare("F1")==0) {
				float precision, recall;
				me.fitness_v = 0;
				for (unsigned int i = 0; i < p.number_of_classes; ++i) {

					if (TP_v[i] + FP_v[i] == 0)
						precision = 0;
					else
						precision = TP_v[i] / (TP_v[i] + FP_v[i]);

					if (TP_v[i] + FN_v[i] == 0)
						recall = 0;
					else
						recall = TP_v[i] / (TP_v[i] + FN_v[i]);

					if (recall + precision != 0)
						me.fitness_v += 2 * (precision*recall) / (precision + recall) / p.number_of_classes;
				}
				me.fitness_v = 1 - me.fitness_v;
				//debug
				if (me.fitness_v<0.01){
					cout << "precision: " << precision << "\n";
					cout << "recall: " << recall << "\n";
					cout << "fitness_v: " << me.fitness_v << "\n";
					cout << "TP: ";
					for (auto i : TP)
						cout << i << ",";
					cout << "FP: ";
					for (auto i : FP)
						cout << i << ",";
						cout << "FN: ";
					for (auto i : TP)
						cout << i << ",";
					cout << "\n";
				}
			}
		}


		if(me.fitness_v>p.max_fit)
			me.fitness_v=p.max_fit;
		else if(me.fitness_v<p.min_fit)
			(me.fitness_v=p.min_fit);
		else if (boost::math::isnan(me.fitness_v))
			me.fitness_v = p.max_fit;
	}
	else{ // if not training, assign copy of regular fitness to the validation fitness variables
		me.abserror_v=me.abserror;
		me.fitness_v=me.fitness;
	}
	//if (p.estimate_generality || p.PS_sel==2){
	//	if (me.fitness == p.max_fit || me.fitness_v== p.max_fit)
	//		me.genty = p.max_fit;
	//	else{
	//		if (p.G_sel==1) // MAE
	//			me.genty = abs(me.abserror-me.abserror_v)/me.abserror;
	//		else if (p.G_sel==2) // R2
	//			me.genty = abs(me.corr-me.corr_v)/me.corr;
	//		else if (p.G_sel==3) // MAE R2 combo
	//			me.genty = abs(me.abserror/me.corr-me.abserror_v/me.corr_v)/(me.abserror/me.corr);
	//		else if (p.G_sel==3) // VAF
	//			me.genty = abs(me.VAF-me.VAF_v)/me.VAF;
	//	}
	//}
	int tmp = omp_get_thread_num();
s.ptevals[omp_get_thread_num()]=s.ptevals[omp_get_thread_num()]+ptevals;
}
void CalcClassOutput(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s)
{
	vector<float> stack_float;
	vector<bool> stack_bool;
	vector<vector<float>> Z; // n x d output for m3gp
	vector<vector<float>> K; // output subsets by class label
	me.reset_introns();
	me.output.resize(0);
	me.output_v.resize(0);
	me.abserror = 0;
	me.abserror_v = 0;

	if (p.sel==3){
		if (p.lex_class)
			me.error.assign(p.number_of_classes,0);
		else
			me.error.resize(0);
	}
	else if (p.sel==4 && p.PS_sel>=4) // each class is an objective
		me.error.assign(p.number_of_classes,0);

	float SStot=0;
	float SSreg=0;
	float SSres=0;
	float q = 0;
	float var_target = 0;
	float var_ind = 0;
	bool pass = true;
	float meanout=0;
	float meantarget=0;
	float meanout_v=0;
	float meantarget_v=0;
	float target_std=1000;
	float target_std_v=1000;
	int ptevals=0;
	int sim_size = vals.size();
	unsigned int ndata_t,ndata_v; // training and validation data sizes
	if (p.train){
		ndata_t = vals.size()*p.train_pct;
		ndata_v = vals.size()-ndata_t;
		if (p.test_at_end) // don't run the validation data
			sim_size = ndata_t;
	}
	else{
		ndata_t = vals.size();
		ndata_v=0;
	}
	me.output.reserve(ndata_t);
	me.output_v.reserve(ndata_v);
	//if (p.eHC_on && p.eHC_slim){ me.stack_float.resize(0); me.stack_float.reserve((me.eff_size)*vals.size());}
	if (p.eHC_on && p.eHC_slim)
	{
		me.stack_float.resize((me.eff_size)*vals.size());
		me.stack_floatlen.resize(0);
		me.stack_floatlen.reserve(me.eff_size);
	}
	//stack_float.reserve(me.eff_size/2);

	for(unsigned int sim=0;sim<sim_size;++sim)
		{
			//if (p.eHC_slim) me.stack_float.push_back(vector<float>());
			int k_eff=0;

			for (unsigned int j=0; j<p.allvars.size();++j) //wgl: add time delay of output variable here
				dattovar.at(j)= vals[sim][j];

			for(int k=0;k<me.line.size();++k){
				if (me.line.at(k).on){
					eval(me.line.at(k),stack_float,stack_bool);
					++ptevals;
					if (p.eHC_on && p.eHC_slim) // stack tracing
					{
						if(!stack_float.empty()) me.stack_float[k_eff*vals.size() + sim] = stack_float.back();
						else me.stack_float[k_eff*vals.size() + sim] = 0;
						if (sim==0) me.stack_floatlen.push_back(stack_float.size());
						/*if(!stack_float.empty()) me.stack_float.push_back(stack_float.back());
						else me.stack_float.push_back(0);*/
					}
					++k_eff;
					/*else
						cout << "hm";*/
				}
			}
			/*if (stack_float.size()>1)
				cout << "non-tree\n";*/

			//if(!(!p.classification && stack_float.empty()) && !(p.classification && stack_bool.empty())){
			if((p.class_bool && !stack_bool.empty()) || (!p.class_bool && !stack_float.empty())){

				if ((p.train && sim<ndata_t) || (!p.train)){
						 if (p.class_bool){
							// use stack_bool
							std::bitset<4> bitout;//(p.number_of_classes,0);
							for (int i = 0;i<bitlen(p.number_of_classes); ++i){
								if (stack_bool.size()>i) // copy from the back of the stack towards the front
									bitout.set(i,stack_bool[stack_bool.size()-1-i]);
							}
							// interpret output as the integer represented by the bitstring produced by stack_bool
							me.output.push_back(float(bitout.to_ulong()));
						}
						else{
							// interpret output as the index of the largest float in stack_float
							vector<float>::iterator it = max_element(stack_float.begin(),stack_float.end());
							me.output.push_back(float(it - stack_float.begin()));
						}

						// error is based solely on whether or not the right class was assigned
						if (target[sim]!=me.output[sim]){
							++me.abserror;
							if (p.sel==3){ // lexicase error vector
								if (p.lex_class)
									++me.error[target[sim]];
								else
									me.error.push_back(1);
							}
							else if (p.sel==4 && p.PS_sel>=4) // each class is an objective
								++me.error[target[sim]];
						}
						else if (p.sel==3 && !p.lex_class)
							me.error.push_back(0);
					}
					else //validation set
					{
						if (p.class_bool){
							//bool stack
							std::bitset<4> bitout; //(p.number_of_classes,0);
							for (int i = 0;i<bitlen(p.number_of_classes); ++i){
								if (stack_bool.size()>i) // copy from back of stack forward
									bitout.set(i,stack_bool[stack_bool.size()-1-i]);
							}
							me.output_v.push_back(float(bitout.to_ulong()));
						}
						else{
							// class is index of largest element in floating point stack
							vector<float>::iterator it = max_element(stack_float.begin(),stack_float.end());
							me.output_v.push_back(float(it - stack_float.begin()));
						}

					 // error based solely on whether or not the right class was assigned
						if (target[sim]!=me.output_v[sim-ndata_t])
							++me.abserror_v;
					}
			} // if stack empty
			else{
				pass=false;
				break;
				}
			stack_float.resize(0);
			stack_bool.resize(0);
		} // end for

		if (pass){
			assert(me.output.size()==ndata_t);
			// mean absolute error
			me.abserror = me.abserror/ndata_t;

			if (p.train)
			{
				// mean absolute error
				me.abserror_v = me.abserror_v/ndata_v;
				meantarget_v = meantarget_v/ndata_v;
				meanout_v = meanout_v/ndata_v;
			}
		}

		if (!pass){
			me.abserror = p.max_fit;
		}


		if(me.output.empty())
			me.fitness=p.max_fit;
		else if ( boost::math::isnan(me.abserror) || boost::math::isinf(me.abserror) )
			me.fitness=p.max_fit;
		else{
			if (!(p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)){
				if (p.verbosity>0) s.out << "warning: fit_type not set to error. using error anyway (because classification is being output)\n";
				p.fit_type="MAE";
			}
			me.fitness = me.abserror;
		}

		if(me.fitness>p.max_fit)
			me.fitness=p.max_fit;
		else if(me.fitness<p.min_fit)
			(me.fitness=p.min_fit);

		if(p.train && !p.test_at_end){ //assign validation fitness
			if(me.output_v.empty())
				me.fitness_v=p.max_fit;
			else if ( boost::math::isnan(me.abserror_v) || boost::math::isinf(me.abserror_v))
				me.fitness_v=p.max_fit;
			else{
				if (!(p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)){
					if (p.verbosity>0) s.out << "WARNING: fit_type not set to error. using error anyway (because classification is being output)\n";
					p.fit_type="MAE";
				}
				me.fitness_v = me.abserror_v;
			}


			if(me.fitness_v>p.max_fit)
				me.fitness_v=p.max_fit;
			else if(me.fitness_v<p.min_fit)
				(me.fitness_v=p.min_fit);
		}
		else{ // if not training, assign copy of regular fitness to the validation fitness variables
			me.abserror_v=me.abserror;
			me.fitness_v=me.fitness;
		}
		//if (p.estimate_generality || p.PS_sel==2){
		//	if (me.fitness == p.max_fit || me.fitness_v== p.max_fit)
		//		me.genty = p.max_fit;
		//	else{
		//		if (p.G_sel==1) // MAE
		//			me.genty = abs(me.abserror-me.abserror_v)/me.abserror;
		//		else if (p.G_sel==2) // R2
		//			me.genty = abs(me.corr-me.corr_v)/me.corr;
		//		else if (p.G_sel==3) // MAE R2 combo
		//			me.genty = abs(me.abserror/me.corr-me.abserror_v/me.corr_v)/(me.abserror/me.corr);
		//		else if (p.G_sel==3) // VAF
		//			me.genty = abs(me.VAF-me.VAF_v)/me.VAF;
		//	}
		//}
		int tmp = omp_get_thread_num();
	s.ptevals[omp_get_thread_num()]=s.ptevals[omp_get_thread_num()]+ptevals;
}
bool CalcSlimOutput(ind& me,params& p,vector<vector<float>>& vals,vector<float>& dattovar,vector<float>& target,state& s,int linestart, float orig_fit)
{
	vector<float> stack_float;// = init_stack;
	vector<bool> stack_bool;
	me.reset_introns();
	me.output.clear();
	me.output_v.clear();
	me.error.clear();
	float SStot=0;
	float SSreg=0;
	float SSres=0;
	float q = 0;
	float var_target = 0;
	float var_ind = 0;
	bool pass = true;
	float meanout=0;
	float meantarget=0;
	float meanout_v=0;
	float meantarget_v=0;
	float target_std=1000;
	float target_std_v=1000;
	int ptevals=0;
	int sim_size = vals.size();
	int ndata_t,ndata_v; // training and validation data sizes
	if (p.train){
		ndata_t = vals.size()*p.train_pct;
		ndata_v = vals.size()-ndata_t;
		if (p.test_at_end) // don't run the validation data
			sim_size = ndata_t;
	}
	else{
		ndata_t = vals.size();
		ndata_v=0;
	}
	// initialize stack for new individual from old stack

	/*for(int i=0;i<vals.size();++i){
		me.stack_float.push_back(vector<float>());
		for(int j=0;j<outstart;++j)
			me.stack_float[i].push_back(init_stack[i].at(j));
	}*/
	int outstart = int(me.stack_float.size()/vals.size());
	me.stack_float.resize(me.eff_size*vals.size());
	int k_eff = outstart;
	int k_fill = outstart;
	me.stack_floatlen.reserve(me.eff_size);
	//if(!me.stack_floatlen.empty()) stack_float.reserve(*std::max(me.stack_floatlen.begin(),me.stack_floatlen.end()));
	for(unsigned int sim=0;sim<sim_size;++sim)
		{
			//initialize stack_float with correct number of elements
			if (me.stack_floatlen.size()>0 && outstart>0){
				for (unsigned int i=me.stack_floatlen.at(outstart-1);i>0;--i)
					stack_float.push_back(me.stack_float[(outstart-i)*vals.size()+sim]);
			}

			/*if (outstart>0)
				stack_float.push_back(init_stack[sim][outstart-1]);*/

			//me.stack_float.push_back(vector<float>());

			for (unsigned int j=0; j<p.allvars.size();++j)
				dattovar.at(j)= vals[sim][j];

			k_eff=outstart;
			for(int k=linestart;k<me.line.size();++k){
				/*if(me.line.at(k).type=='v'){
					if (static_pointer_cast<n_sym>(me.line.at(k)).valpt==NULL)
						cout<<"WTF";
				}*/
				if (me.line.at(k).on){
					//me.line.at(k).eval(stack_float);
					eval(me.line.at(k),stack_float, stack_bool);
					++ptevals;
					if(!stack_float.empty()) me.stack_float[k_eff*vals.size() + sim] = stack_float.back();
					else me.stack_float[k_eff*vals.size() + sim] = 0;
					/*if(!stack_float.empty()) me.stack_float.push_back(stack_float.back());
					else me.stack_float.push_back(0);*/
					if (sim==0) me.stack_floatlen.push_back(stack_float.size());
					++k_eff;
				}

			}

			if(!stack_float.empty()){
				if (p.train){
					if(sim<ndata_t){
						me.output.push_back(stack_float.back());
						me.abserror += abs(target.at(sim)-me.output.at(sim));
						me.sq_error += pow(target[sim] - me.output[sim], 2);
						if(me.abserror/vals.size() > orig_fit)
							return 0;
						meantarget += target.at(sim);
						meanout += me.output[sim];

					}
					else
					{
						me.output_v.push_back(stack_float.back());
						me.abserror_v += abs(target.at(sim)-me.output_v.at(sim-ndata_t));
						me.sq_error_v += pow(target[sim] - me.output_v[sim-ndata_t], 2);
						meantarget_v += target.at(sim);
						meanout_v += me.output_v[sim-ndata_t];

					}

				}
				else {
					me.output.push_back(stack_float.back());
					me.abserror += abs(target.at(sim)-me.output.at(sim));
					me.sq_error += pow(target.at(sim) - me.output.at(sim),2);
					if(me.abserror/vals.size() > orig_fit)
							return 0;
					meantarget += target.at(sim);
					meanout += me.output[sim];
				}
			}
			else{
				return 0;
				}
			//stack_float.clear();
			stack_float.resize(0);
		}
		if (!pass) return 0;
		else
		{
			// mean absolute error
			me.abserror = me.abserror/ndata_t;
			me.sq_error /= ndata_t;
			meantarget = meantarget/ndata_t;
			meanout = meanout/ndata_t;
			//calculate correlation coefficient
			me.corr = getCorr(me.output,target,meanout,meantarget,0,target_std);
			me.VAF = VAF(me.output,target,meantarget,0);

			if (p.train && !p.test_at_end)
			{
				q = 0;
				var_target = 0;
				var_ind = 0;
					// mean absolute error
				me.abserror_v = me.abserror_v/ndata_v;
				me.sq_error_v /= ndata_v;
				meantarget_v = meantarget_v/ndata_v;
				meanout_v = meanout_v/ndata_v;
				//calculate correlation coefficient
				me.corr_v = getCorr(me.output_v,target,meanout_v,meantarget_v,ndata_t,target_std_v);
				me.VAF_v = VAF(me.output_v,target,meantarget_v,ndata_t);
			}
		}

		if (me.eqn.compare("1")==0 && me.corr > 0.0001)
				cout << "caught\n";

			if (!pass){
				me.corr = 0;
				me.VAF = 0;
			}

			if(me.corr < p.min_fit)
				me.corr=p.min_fit;
			if(me.VAF < p.min_fit)
				me.VAF=p.min_fit;

		    if(me.output.empty())
				me.fitness=p.max_fit;
			else if ( boost::math::isnan(me.abserror) || boost::math::isinf(me.abserror) || boost::math::isnan(me.corr) || boost::math::isinf(me.corr))
			{
				me.fitness=p.max_fit;
				me.abserror=p.max_fit;
				me.sq_error = p.max_fit;
				me.corr = p.min_fit;
				me.VAF = p.min_fit;
			}
			else{
				if (p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)
					me.fitness = me.abserror;
				else if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0)
					me.fitness = 1-me.corr;
				else if (p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0)
					me.fitness = me.abserror/me.corr;
				else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0)
					me.fitness = 1-me.VAF/100;
				else if (p.fit_type.compare("MSE")==0)
					me.fitness = me.sq_error;
				if (p.norm_error)
					me.fitness = me.fitness/target_std;
			}


			if(me.fitness>p.max_fit)
				me.fitness=p.max_fit;
			else if(me.fitness<p.min_fit)
				(me.fitness=p.min_fit);

			if(p.train && !p.test_at_end){ //assign validation fitness
				if (!pass){
					me.corr_v = 0;
					me.VAF_v = 0;
				}

				if(me.corr_v < p.min_fit)
					me.corr_v=p.min_fit;
				if(me.VAF_v < p.min_fit)
					me.VAF_v=p.min_fit;

				if(me.output_v.empty())
					me.fitness_v=p.max_fit;
				/*else if (*std::max_element(me.output_v.begin(),me.output_v.end())==*std::min_element(me.output_v.begin(),me.output_v.end()))
					me.fitness_v=p.max_fit;*/
				else if ( boost::math::isnan(me.abserror_v) || boost::math::isinf(me.abserror_v) || boost::math::isnan(me.corr_v) || boost::math::isinf(me.corr_v))
					me.fitness_v=p.max_fit;
				else{
					if (p.fit_type.compare("1")==0 || p.fit_type.compare("MAE")==0)
						me.fitness_v = me.abserror_v;
					else if (p.fit_type.compare("2")==0 || p.fit_type.compare("R2")==0)
						me.fitness_v = 1-me.corr_v;
					else if (p.fit_type.compare("3")==0 || p.fit_type.compare("MAER2")==0)
						me.fitness_v = me.abserror_v/me.corr_v;
					else if (p.fit_type.compare("4")==0 || p.fit_type.compare("VAF")==0)
						me.fitness_v = 1-me.VAF/100;
					else if (p.fit_type.compare("MSE")==0)
						me.fitness_v = me.sq_error_v;
					if (p.norm_error)
						me.fitness_v = me.fitness_v/target_std_v;

				}


				if(me.fitness_v>p.max_fit)
					me.fitness_v=p.max_fit;
				else if(me.fitness_v<p.min_fit)
					(me.fitness_v=p.min_fit);
			}
			else{ // if not training, assign copy of regular fitness to the validation fitness variables
				me.corr_v=me.corr;
				me.VAF_v = me.VAF;
				me.abserror_v=me.abserror;
				me.sq_error_v = me.sq_error;
				me.fitness_v=me.fitness;
			}
			/*if (p.estimate_generality || p.PS_sel==2){
				if (me.fitness == p.max_fit || me.fitness_v== p.max_fit)
					me.genty = p.max_fit;
				else
					me.genty = abs(me.fitness-me.fitness_v)/me.fitness;
			}*/
			int tmp = omp_get_thread_num();
		s.ptevals[omp_get_thread_num()]=s.ptevals[omp_get_thread_num()]+ptevals;
		return true;
}
bool getSlimFit(ind& me,params& p,Data& d,state& s,FitnessEstimator& FE,int linestart, float orig_fit)

{
    //set up data table for conversion of symbolic variables
	unordered_map <string,float*> datatable;
	vector<float> dattovar(d.label.size());

	for (unsigned int i=0;i<d.label.size(); ++i)
			datatable.insert(pair<string,float*>(d.label[i],&dattovar[i]));

	vector<vector<float>> FEvals;
	vector<float> FEtarget;
	setFEvals(FEvals,FEtarget,FE,d);

	me.abserror = 0;
	me.abserror_v = 0;
	me.sq_error = 0;
	me.sq_error_v = 0;
	// set data table
	for(int m=0;m<me.line.size();++m){
		if(me.line.at(m).type=='v')
			{// set pointer to dattovar
				//float* set = datatable.at(static_pointer_cast<n_sym>(me.line.at(m)).varname);
				float* set = datatable.at(me.line.at(m).varname);
				if(set==NULL)
					cout<<"hmm";
				/*static_pointer_cast<n_sym>(me.line.at(m)).setpt(set);*/
				me.line.at(m).setpt(set);
				/*if (static_pointer_cast<n_sym>(me.line.at(m)).valpt==NULL)
					cout<<"wth";*/
			}
	}
	//cout << "Equation" << count << ": f=" << me.eqn << "\n";
			//bool pass=true;
			if(me.eqn.compare("unwriteable")==0)
				return 0;
			else
			{
			    // calculate error
				if(!p.EstimateFitness){
					return CalcSlimOutput(me,p,d.vals,dattovar,d.target,s,linestart,orig_fit);
				} // if not estimate fitness
				else{
					return CalcSlimOutput(me,p,FEvals,dattovar,FEtarget,s,linestart,orig_fit);
				} // if estimate fitness
			} // if not unwriteable equation

}
bool SlimFitness(ind& me,params& p,Data& d,state& s,FitnessEstimator& FE, int linestart,  float orig_fit)
{

	me.eqn = Line2Eqn(me.line,me.eqn_form,p,true);
	//getEqnForm(me.eqn,me.eqn_form);

	me.eff_size = getEffSize(me.line);

	bool pass = getSlimFit(me,p,d,s,FE,linestart,orig_fit);

	me.complexity= getComplexity(me,p);
	s.numevals[omp_get_thread_num()]=s.numevals[omp_get_thread_num()]+1;
	return pass;
}
