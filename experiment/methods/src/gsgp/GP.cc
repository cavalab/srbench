/*  Copyright Mauro Castelli
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

//!  \file           GP.cc
//! \brief           file containing the main with the genetic programming algorithm
//! \date            created on 01/09/2016

#include "GP.h"

using namespace std;


/*!
* \fn              int main(int argc, const char **argv)
* \brief           main method that runs the GP algorithm
* \param           int argc: number of parameters of the program
* \param           const char **argv: array of strings that contains the parameters of the program
* \return          int: 0 if the program ends without errors
* \date            01/09/2016
* \author          Mauro Castelli
* \file            GP.cc
*/
int main(int argc, const char **argv){

    // name of the file with training instances 
    char path_in[1000]="";
    // name of the file with test instances
    char path_test[1000]="";
    char dataset_name[1000]="";
   	for (int i=1; i<argc-1; i++) {
        if(strncmp(argv[i],"-train_file",11) == 0) {
            strcat(path_in,argv[++i]);
        }
        else if(strncmp(argv[i],"-test_file",10) == 0) {
                strcat(path_test,argv[++i]);
	     }
       	if (strncmp(argv[i],"-name",5) == 0) {
            strcat(dataset_name,argv[++i]);
       	}                        
   	}
	std::string dataset(dataset_name);
    std::cout << "path_in: " << path_in << endl;
    std::cout << "path_test: " << path_test << endl;
    std::cout << "dataset: " << dataset << endl;
	
	// initialization of the seed for the generation of random numbers
	//srand(time (NULL));
        srand(42);    
	// reading the parameters of the GP algorithm
	read_config_file(dataset, &config);	
	// creation of an empty population
	population *p=new population();
	
	// if USE_TEST_SET is equal to 1 the system will apply the best model to newly provided unseen data
	if(config.USE_TEST_SET==1){
		string evun=dataset+"-evaluation_on_unseen_data.txt";
		ofstream OUT(evun.c_str(),ios::out);
		fstream in(path_test,ios::in);
		
		if (!in.is_open()) {
			cout<<endl<<"ERRORE: IMPOSSIBILE APRIRE IL FILE" << endl;
			exit(-1);
		}
		else{
			char n_v[255];
			in >> n_v;
			nvar=atoi(n_v);
			// creation of terminal and functional symbols
			create_T_F();
			// initialization of the population
			create_population(dataset, (population **)&p, config.init_type);		
			set = new Instance[1];
			set[0].vars = new double[nvar];
			while(!in.eof()){
				char str[255];    
				for (int j=0; j<nvar; j++) {
					in >> str;
					set[0].vars[j] = atof(str);
				}
				update_terminal_symbols(0);
				evaluate_unseen_new_data(dataset, (population**)&p, OUT);       	    
			}
		}
	}
	// if USE_TEST_SET is different from 1 the system will perform the usual evolutionary process
	else{
		string extf=dataset+"-execution_time.txt";
		ofstream executiontime(extf.c_str(),ios::out);
		timeval start1, stop1;
		gettimeofday(&start1, NULL);
		double elapsedTime=0;
	
		/*
		pointer to the file fitnesstrain.txt containing the training fitness of the best individual at each generation
		*/
		string fitness_train_file=dataset+"-fitnesstrain.txt";
		ofstream fitness_train(fitness_train_file.c_str(),ios::out);
		/*
		pointer to the file fitnesstest.txt containing the training fitness of the best individual at each generation
		*/
		string fitness_test_file=dataset+"-fitnesstest.txt";
		ofstream fitness_test(fitness_test_file.c_str(),ios::out);

		// reading training and test files
		read_input_data(path_in,path_test);
		// creation of terminal and functional symbols
		create_T_F();
		// initialization of the population
		create_population(dataset, (population **)&p, config.init_type);	
		// evaluation of the individuals in the initial population
		evaluate((population**)&p);
		// writing the  training fitness of the best individual on the file fitnesstrain.txt
		fitness_train<<Myevaluate(p->individuals[p->index_best])<<endl;
		// writing the  test fitness of the best individual on the file fitnesstest.txt
		fitness_test<<Myevaluate_test(p->individuals[p->index_best])<<endl;
		// index of the best individual stored in the variable best_index
		index_best=best_individual();
		
		gettimeofday(&stop1, NULL);
		elapsedTime += ((stop1.tv_sec - start1.tv_sec) * 1000.0) + ((stop1.tv_usec - start1.tv_usec) / 1000.0);
		executiontime<<elapsedTime<<endl;
	
		//File containing the individuals and the random trees
		string expression_file_name=dataset+"-individuals.txt";
		ofstream expression_file(expression_file_name.c_str(),ios::out);
		
		for(int i=0;i<config.population_size+config.random_tree;i++){
			string s="";
			print_math_style(p->individuals[i],s);
			expression_file<<s<<endl;
		}
	
		// main GP cycle
		for(int num_gen=0; num_gen<config.max_number_generations; num_gen++){	
      
			timeval start, stop;
			//Register execution time
			gettimeofday(&start, NULL);
		
			//cout<<"Generation "<<num_gen+1<<endl;
			// creation of a new population (without building trees!!)
			for(int k=0;k<config.population_size;k++){
				double rand_num=frand();
				// geometric semantic crossover
				if(rand_num<config.p_crossover)
					geometric_semantic_crossover(k);
				// geometric semantic mutation    
				if(rand_num>=config.p_crossover && rand_num<config.p_crossover+config.p_mutation){
					reproduction(k,true);
					geometric_semantic_mutation(k);
				}
				// reproduction
				if(rand_num>=config.p_crossover+config.p_mutation){
					reproduction(k,false);
				}
			}
        
			// updating the tables used to store semantics and fitness values
			update_tables();
			// index of the best individual stored in the variable best_index
			index_best=best_individual(); 
			// updating the information used to explore the DAG whose nodes are the GP individuals
			vector_traces.push_back(traces_generation);
			traces_generation.clear();
			// writing the  training fitness of the best individual on the file fitnesstrain.txt       
			fitness_train<<fit_[index_best]<<endl;
			// writing the  test fitness of the best individual on the file fitnesstest.txt
			fitness_test<<fit_test[index_best]<<endl;
		
			gettimeofday(&stop, NULL);
			elapsedTime += ((stop.tv_sec - start.tv_sec) * 1000.0) + ((stop.tv_usec - start.tv_usec) / 1000.0);
			executiontime<<elapsedTime<<endl;
		}	    
		// marking procedure whose output is saved on the file "trace.txt"
		mark_trace();
		save_trace(dataset);
	}
    // at the end of the execution all the data structures are deleted in order to deallocate memory
	for(int k=0; k<config.population_size+config.random_tree; k++){
        delete_individual(p->individuals[k]);
	}
	delete[] p->fitness;
	delete[] p->fitness_test;
	delete p;	
	for(int i=0; i<nrow+nrow_test; i++){
        delete[] set[i].vars;
	}
	delete[] set;
	for(int i=symbols.size()-1;i>=0;i--){
        delete symbols[i];
		symbols.erase(symbols.begin()+i);
	}
	symbols.clear();
	
	return 0;
}
