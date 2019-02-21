// header file for ind struct
#pragma once
#ifndef DATA_H
#define DATA_H
// struct params;
#include "params.h"
using namespace std;
// using std::vector;
// using std::begin;
// using std::string;
// #define SIZEOF_ARRAY( arr ) (sizeof( arr ) / sizeof( arr[0] ))

template<typename T, size_t n>
size_t array_size(const T (&)[n]) {
    return n;
}
//typedef exprtk::parser_error::type error_t;

struct Data {


	vector<string> label; // variables corresponding to those defined in parameters
	vector<vector<float> > vals; //2D vector of all data, can be accessed data[row][column]
	//vector<vector<float>> FEvals;
	//fitness estimator vector

	//vector<float> dattovar;
	vector<float> target;
	//unordered_map <string,float*> datatable;
	//mymap.
	/*mymap.insert(pair<string,int*>("alpha",10));
	mymap.insert(pair<string,int>("beta",20));
	mymap.insert(pair<string,int>("G1",30));*/

	// variables for lexicase selection
	vector<vector<float> > targetlex;
	vector<vector<vector<float> > > lexvals;
	//vector<int> lexicase;
	string target_var; //target variable

	Data(){}
	void clear()
	{
		label.clear();
		vals.clear();
		//dattovar.clear();
		target.clear();
		//datatable.clear();
	}
	/*void mapdata()
	{
		for (unsigned int i=0;i<label.size(); ++i)
			datatable.insert(pair<string,float*>(label[i],&dattovar[i]));
	}*/
	~Data()
	{
		//I think the data pointed to by the map should be destroyed
		//in here (datatable) as well as the pointer, since both are contained within the class.
		//therefore this deletion is unecessary. also no news are called.
		/*for (unordered_map<string,float*>::iterator i = datatable.begin(); i != datatable.end();++i)
		{
			delete (*i).second;
		}
		datatable.clear();*/
	}

	void set_train(float* X,size_t N, size_t D){
		// using std::begin;
        // std::cout << "N: " << N << ", D: " << D << "\n";
        int r = 0;
		for (unsigned int i=0; i<N; ++i){
            r = i*D;
            vals.push_back(vector<float>());
            for (unsigned int d =0; d< D; ++d)
                vals[i].push_back(X[r+d]);
		}
        // for (int i = 0; i<10; ++i){
        //     for (int j=0; j < vals[0].size(); ++j)
        //         cout << vals[i][j] << ",";
        //     cout << "\n";
        // }
	}

	void set_target(float* Y, size_t N){
		target.assign(Y, Y+N);
	}
    void define_class_weights(params& p) {

		p.class_w.assign(p.number_of_classes,0);
		p.class_w_v.assign(p.number_of_classes, 0);
		for (unsigned int i = 0; i < target.size(); ++i) {
			if (p.train && i >= target.size()*p.train_pct)
				++p.class_w_v[target[i]];
			else
				++p.class_w[target[i]];
		}
		for (unsigned int i = 0; i < p.number_of_classes; ++i) {
			if (p.train) {
				p.class_w[i] /= target.size()*p.train_pct;
				p.class_w_v[i] /= target.size()*(1-p.train_pct);
			}
			else
				p.class_w[i] /= target.size();
		}
        cout << "class weights:\n";
        for (auto i : p.class_w)
            cout << i << ",";
    }

	void set_dependencies(params& p){
		// used when set_train and set_target are called to fill data instead of load_data.
		// conducts preprocessing steps conditionally on p

    	// set variable names
    	target_var = "target";
        for (unsigned int i = 0; i < vals[0].size(); ++i)
            label.push_back("x_"+to_string(static_cast<long long>(i)));

    	bool useall= p.allvars.empty();
    	// if intvars is not specified, set it to all variables in data file

    	if (useall){ // assign data labels to p.allvars
    		p.allvars = label;
    		p.allblocks = label;
    	}

    	// // pop end in case of extra blank lines in data file
    	// while(vals.back().empty())
    	// {
    	// 	vals.pop_back();
    	// 	target.pop_back();
    	// }

    	if (p.classification){
    		// set p.num_cases based on unique elements in target
    		vector<float> tmp = target;
    		sort(tmp.begin(),tmp.end());
    		tmp.erase(unique(tmp.begin(),tmp.end()),tmp.end());
    		p.number_of_classes = tmp.size();
    		// set lowest class equal to zero
    		int offset  = *std::min_element(target.begin(),target.end());
    		if (offset>0){
    			for (unsigned i=0;i<target.size();++i)
    				target[i] -= offset;
    		}
    	}

        // //debugging
        // cout << "data before AR:\n";
        // for (auto l : label)
        //     cout << l << ",";
        //
        // cout << "\n";
        // for (int i = 0; i<10; ++i){
        //     for (int j=0; j < vals[0].size(); ++j)
        //         cout << vals[i][j] << ",";
        //     cout << "\n";
        // }
    	if (p.AR){ // make auto-regressive variables
    		vector<vector<float> > tmp_vals = vals;

    		// add data columns to vals
    		int numvars = vals[0].size();
    		for (unsigned i = 0; i<p.AR_nb; ++i){
    			for (unsigned j = 0; j<tmp_vals.size(); ++j){
    				if(i==0)
    					vals[j].resize(0);

    				for (unsigned k = 0; k<numvars;++k){
    					if(j>=i+p.AR_nkb)
    						vals[j].push_back(tmp_vals[j-i-p.AR_nkb][k]);
    					else
    						vals[j].push_back(0.0);
    				}
    			}
    		}

    		// add data labels to label and p.allvars
    			int tmp = label.size();
    			vector<string> tmp_label = label;
    			label.resize(0);
    			p.allvars.resize(0);
    			//for (unsigned i = 0; i < p.AR_nb + p.AR_nkb; ++i) {
    			//	vals.erase(vals.begin()); // erase zero padding from data
    			//	target.erase(target.begin()); //erase zero padding from target
    			//}
    			for (unsigned i = 0; i < p.AR_nb; ++i) {
    				for (unsigned k=0; k<tmp;++k){ // add delay variable names
    					label.push_back(tmp_label[k] + "_d" + to_string(static_cast<long long>(i+p.AR_nkb)));
    					p.allvars.push_back(tmp_label[k] + "_d" + to_string(static_cast<long long>(i+p.AR_nkb)));
    				}
    			}
    			// add target AR variables
    			for (int i=0;i<p.AR_na;++i){
    				p.allvars.push_back(target_var + "_" + to_string(static_cast<long long>(i+p.AR_nka)));
    				label.push_back(target_var + "_" + to_string(static_cast<long long>(i+p.AR_nka)));
    			}

    	}
    	//dattovar.resize(p.allvars.size());
    	//mapdata();
    	assert(target.size() == vals.size());

        //debugging
        // cout << "data:\n";
        // for (auto l : label)
        //     cout << l << ",";
        //
        // cout << "\n";
        // for (int i = 0; i<10; ++i){
        //     for (int j=0; j < vals[0].size(); ++j)
        //         cout << vals[i][j] << ",";
        //     cout << "\n";
        // }


	}
};

#endif
