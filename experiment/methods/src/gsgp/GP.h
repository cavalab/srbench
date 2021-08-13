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

//!  \file            GP.h
//! \brief            file containing the definition of the classes used to represent a symbol, a node of the tree, a population of individuals and definition of the functions
//! \date            created on 01/09/2016

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <sys/time.h>
/// Macro used to generate a random number
#define frand() ((double) rand() / (RAND_MAX))




/// variable containing the numbers of terminal symbols (variables, not constant values)
int NUM_VARIABLE_SYMBOLS;
/// variable that stores the numbers of terminal symbols that contain constant values
int NUM_CONSTANT_SYMBOLS;
/// variable containing the numbers of functional symbols
int NUM_FUNCTIONAL_SYMBOLS;
//These are just parenthesis
int NUM_AUXILIARY_FUNCTIONAL_SYMBOLS;

/// struct used to store a single instance in memory
typedef struct Instance_{
/// array containing the values of the independent variables    
    double *vars;
/// variable that stores the result of the evaluation of an individual on the particular instance represented by the values stored in the variable double *vars
    double res;
/// target value   
    double y_value;
} Instance;

/// variable used to store training and test instances in memory
Instance *set;

/// variable containing the numbers of rows (instances) of the training dataset
int nrow;
/// variable containing the numbers of columns (excluding the target) of the training dataset
int nvar;
/// variable containing the numbers of rows (instances) of the test dataset
int nrow_test;
/// variable containing the numbers of columns (excluding the target) of the test dataset
int nvar_test;

using namespace std;

/// struct used to store the parameters of the configuration.ini file
typedef struct cfg_{
/// size of the population: number of candidate solutions
    int population_size;
/// number of iterations of the GP algorithm
    int max_number_generations;
/// initialization method: 0 -> grow method, 1-> full method, 2-> ramped half and half method
    int init_type;
/// crossover rate
    double p_crossover;
/// mutation rate
    double p_mutation;
/// maximum depth of a newly created individual
    int max_depth_creation;
/// size of the tournament selection 
    int tournament_size;
/// variable that indicates if it is possible to accept single-node individual in the initial population
    int zero_depth;
/// mutation step of the geometric semantic mutation
    double mutation_step;
/// variable that indicates the number of constants to be inserted in the terminal set
    int num_random_constants; 
/// variable that indicates the minimum possible value for a random constant
    double min_random_constant;
/// variable that indicates the maximum possible value for a random constant    
    double max_random_constant;
/// variable that indicates if the problem is a minimization problem (1) or a maximization problem (0)
    int minimization_problem;
/// variable that indicates the number of random tree used for geometric operators
    int random_tree;
/// variable that indicates if a file with saved individuals must be considered
    int expression_file;	
/// variable that indicates if a newly provided test set must be used
	int USE_TEST_SET;
}cfg;

/// struct variable containing the values of the parameters specified in the configuration.ini file
cfg config;



/**
 * \class symbol
 *
 * \brief 
 *
 * This class represents a symbol of the set T (terminal symbols) or F (functional symbols). 
 *
 * \author Mauro Castelli
 *
 * \version 0.0.1
 *
 * \date 01/09/2016
 *
 */
class symbol{
	public:
        /// boolean variable used to discriminate between functional and terminal symbols
		bool type; 
        /// int variable that contains the number of arguments accepted by a symbol. It is 0 for a terminal symbol 
		int arity;
		/// int variable that contains a unique identifier for the symbol
        int id; 
        /// symbolic name of the symbol 
		char name[30]; 
		/// variable that contains the current value of a terminal symbol 
		double value;
		/// variable that contains precedence 
		int precedence;
		symbol(){};
		///constructor that creates a symbols and sets the variable type, num_arguments, id and value
		symbol(bool p_type, int p_arity, int p_id, const char *p_name, int p_precedence){
			type=p_type;
			arity=p_arity;
			id=p_id;
			precedence=p_precedence;
			strcpy(name,p_name);
		};
};

///array containing terminal and functional symbols
vector <symbol *> symbols;

/**
 * \class node
 *
 * \brief
 *
 * This class is used to represent a node of the tree.
 *
 * \author Mauro Castelli
 *
 * \version 0.0.1
 *
 * \date 01/09/2016
 *
 */
class node{
	public:
        ///symbol inside a node
		symbol* root;
		/// parent node
		node* parent;
		///pointers to children
		node **children;
		/// class destructor
		~node() {delete[] children;}
};


/**
 * \class population
 *
 * \brief
 *
 * This class is used to represent a GP population.
 *
 * \author Mauro Castelli
 *
 * \version 0.0.1
 *
 * \date 01/09/2016
 *
 */
class population{
	public:
        /// pointers to individuals
		node **individuals;
		/// int variable that contains the index of the best individual in the population
		int index_best;
		/// int variable that contains the number of individuals that are inside the population
		int num_ind;
		/// array of training fitness values
		double *fitness;
		/// array of test fitness values
		double *fitness_test;
		/// class constructor
		population(){individuals=new node* [config.population_size+config.random_tree]; num_ind=0;
		fitness=new double [config.population_size];
		fitness_test=new double [config.population_size];
		};
		/// class destructor
		~population() { delete[] individuals;}
};

/// array of training fitness values at generation g
vector <double> fit_;
/// array of test fitness values at generation g
vector <double> fit_test;
/// array of training fitness values at the current generation g+1
vector <double> fit_new;
/// array of test fitness values at the current generation g+1
vector <double> fit_new_test;

/// array where each element (that is also an array) contains the semantics of an individual of the population, computed on the training set at generation g
vector < vector<double> > sem_train_cases;
/// array where each element (that is also an array) contains the semantics of an individual of the population, computed on the training set at generation g+1
vector < vector<double> > sem_train_cases_new;
/// array where each element (that is also an array) contains the semantics of an individual of the population, computed on the test set at generation g
vector < vector<double> > sem_test_cases;
/// array where each element (that is also an array) contains the semantics of an individual of the population, computed on the test set at generation g+1
vector < vector<double> > sem_test_cases_new;


//variables used to store the semantics of the random trees
vector < vector<double> > fit_ran;
vector < vector<double> > fit_ran_test;

/// struct representing the tuple used to store the information used to reconstruct the structure of the optimal solution
typedef struct entry_{
	/// variable that stores the information about the genetic operator that is applied: crossover (0), mutation (1) and reproduction (-1)
   int event;
   /// variable containing the index of the first random tree (mutation) or the index of the first parent (crossover) 
   int first_parent;
   /// variable containing the index of the second random tree (mutation) or the index of the second parent (crossover)
   int second_parent;
   /// variable containing the index of the parent (mutation) or the index of the random tree (crossover)
   int number;
   /// variable used to reconstruct the optimal solution. 1 means that a particular tree is involved in the construction of the optimal solution, 0 means that the particular tree can be ignored.
   int mark;
   /// variable containing the index of newly created individual
   int individual;
   /// variable containing the mutation step of semantic mutation
   double mut_step;
}entry;

/// variable used to store the information needed to evaluate the optimal individual on newly provided unseen data.
vector <entry> traces_generation;
vector < vector <entry> > vector_traces;


/// variable that stores the index of the best individual.  
int index_best;

/*!
* \fn                 void read_config_file(cfg *config)
* \brief             function that reads the configuration file
* \param          cfg *config: pointer to the struct containing the variables needed to run the program
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void read_config_file(string dataset, cfg *config);


/*!
* \fn                void create_T_F()
* \brief             function that creates the terminal and functional sets.
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_T_F();


/*!
* \fn                 int choose_function()
* \brief             function that randomly selects a functional symbol
* \return           int: the ID of the chosen functional symbol
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
int choose_function();

/*!
* \fn                 int choose_terminal()
* \brief             function that randomly selects a terminal symbol. With probability 0.7 a variable is selected, while random constants have a probability of 0.3 to be selected. To change these probabilities just change their values in the function.
* \return           int: the ID of the chosen terminal symbol
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
int choose_terminal();


/*!
* \fn                void create_grow_pop(population **p)
* \brief             function that creates a population using the grow method.
* \param          population **p: pointer to an empty population
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_grow_pop(population **p);

/*!
* \fn                void create_full_pop(population **p)
* \brief             function that creates a population of full trees (each tree has a depth equal to the possible maximum length).
* \param          population **p: pointer to an empty population
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_full_pop(population** p);

/*!
* \fn                void create_ramped_pop(population **p)
* \brief             function that creates a population with the ramped half and half algorithm.
* \param          population **p: pointer to an empty population
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_ramped_pop(population **p);

/*!
* \fn                void create_population(population **p, int i)
* \brief             function that creates a population using the method specified by the parameter int i.
* \param          population **p: pointer to an empty population
* \param          int i: type of initialization method
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_population(string data, population **p, int i);


/*!
* \fn                void create_grow_tree(node **el, int depth, node *parent, int max_depth)
* \brief             function that creates a random tree with depth in the range [0;max_depth]
* \param          node **el: pointer to the node that must be added to the tree
* \param          int depth: current depth of the tree
* \param          node *parent: parent node
* \param          int max_depth: maximum depth of the tree
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_grow_tree(node **el, int depth, node *parent, int max_depth);

/*!
* \fn                void create_full_tree(node **el, int depth, node *parent, int max_depth)
* \brief             function that creates a tree with depth equal to the ones specified by the parameter max_depth
* \param          node **el: pointer to the node that must be added to the tree
* \param          int depth: current depth of the tree
* \param          node *parent: parent node
* \param          int max_depth: maximum depth of the tree
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void create_full_tree(node **el, int depth, node *parent, int max_depth);


/*!
* \fn                 double protected_division(double num, double den)
* \brief             function that implements a protected division. If the denominator is equal to 0 the function returns 1 as a result of the division;
* \param          double num: numerator
* \param          double den: denominator
* \return           double: the result of the division if denominator is different from 0; 1 if denominator is equal to 0
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
double protected_division(double num, double den);


/*!
* \fn                 double eval(node *tree)
* \brief             function that evaluates a tree.
* \param          node *tree: radix of the tree to be evaluated
* \return           double: the value of the evaluation
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
double eval(node *tree);


/*!
* \fn                 void evaluate(population **p)
* \brief             function that calculates the fitness of all the individuals and determines the best individual in the population
* \param          population **p: pointer to the population containing the individuals
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void evaluate(population **p);


/*!
* \fn                 double Myevaluate(node *el)
* \brief             function that calculates the training fitness of an individual (representing as a tree)
* \param          node *el: radix of the tree
* \return           double: the training fitness of the individual
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
double Myevaluate(node *el);


/*!
* \fn                 double Myevaluate_test(node *el)
* \brief             function that calculates the test fitness of an individual (representing as a tree)
* \param          node *el: radix of the tree
* \return           double: the test fitness of the individual
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
double Myevaluate_test(node *el);


/*!
* \fn                void Myevaluate_random (node *el, vector <double> & sem)
* \brief             function that calculates the semantics (considering training instances) of a randomly generated tree. The tree is used to perform the semantic geometric crossover or the geometric semantic mutation
* \param          node* el: radix of the tree to be evaluated
* \return           void 
* \date             01/09/2016
* \author          Mauro Castelli
* \file              GP.h
*/
void Myevaluate_random (node *el);


/*!
* \fn                 double Myevaluate_random_test(node *el, , vector <double> & sem)
* \brief             function that calculates the semantics (considering test instances) of a randomly generated tree. The tree is used to perform the semantic geometric crossover or the geometric semantic mutation
* \param          node* el: radix of the tree to be evaluated
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file              GP.h
*/
void Myevaluate_random_test(node *el);


/*!
* \fn                void update_terminal_symbols(int i)
* \brief             function that updates the value of the terminal symbols in a tree.
* \param          int i: line of the dataset containing the values of the terminal symbols
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void update_terminal_symbols(int i);


/*!
* \fn                 void delete_individual(node * el)
* \brief             function that deletes a tree and frees the the memory allocated to store the tree
* \return           node* el: radix of the tree to be deleted
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void delete_individual(node * el);


/*!
* \fn                int tournament_selection()
* \brief             function that implements a tournament selection procedure
* \return           int: index of the best individual among the ones that participate at the tournament
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
int tournament_selection();


/*!
* \fn                void reproduction(int i, bool flag_mutation)
* \brief             function that copy an individual of the population at generation g-1 to the current population(generation g)
* \param            int i: index of the individual in the current population
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void reproduction(int i, bool flag_mutation);


/*!
* \fn                void geometric_semantic_crossover(int i)
* \brief             function that performs a geometric semantic crossover
* \param            int i: index of the newly created individual in the new population
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void geometric_semantic_crossover(int i);

/*!
* \fn                void geometric_semantic_mutation(int i)
* \brief             function that performs a geometric semantic mutation
* \param            int i: index of the mutated individual in the new population
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file             GP.h
*/
void geometric_semantic_mutation(int i);


/*!
* \fn                void update_training_fitness(vector <double> semantic_values, bool crossover)
* \brief             function that calculate the training fitness of an individual using the information stored in its semantic vector. The function updates the data structure that stores the training fitness of the individuals
* \param            vector <double> semantic_values: vector that contains the semantics (calculated on the training set) of an individual
* \param            bool crossover: variable that indicates if the function has been called by the geometric semantic crossover or by the geometric semantic mutation
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void update_training_fitness(vector <double> semantic_values, bool crossover);


/*!
* \fn                void update_test_fitness(vector <double> semantic_values, bool crossover)
* \brief             function that calculate the test fitness of an individual using the information stored in its semantic vector. The function updates the data structure that stores the test fitness of the individuals
* \param            vector <double> semantic_values: vector that contains the semantics (calculated on the test set) of an individual
* \param            bool crossover: variable that indicates if the function has been called by the geometric semantic crossover or by the geometric semantic mutation
* \return           void
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
void update_test_fitness(vector <double> semantic_values, bool crossover);


/*!
* \fn                int best_individual()
* \brief             function that finds the best individual in the population
* \return           int: the index of the best individual
* \date             01/09/2016
* \author          Mauro Castelli
* \file               GP.h
*/
int best_individual();


/*!
* \fn               void update_tables()
* \brief            function that updates the tables used to store fitness values and semantics of the individual. It is used at the end of each iteration of the algorithm
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void update_tables();


/*!
* \fn               void read_input_data(char *train_file, char *test_file)
* \brief            function that reads the data from the training file and from the test file.
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void read_input_data(char *train_file, char *test_file);


/*!
* \fn               bool better (double f1, double f2)
* \brief            function that compares the fitness of two solutions.
* \param            double f1: fitness value of an individual
* \param            double f2: fitness value of an individual
* \return           bool: true if f1 is better than f2, false in the opposite case
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
bool better (double f1, double f2);


/*!
* \fn               void copy_operand(node *tree, node **el, node *parent)
* \brief            function used to copy an operand from the stack of operands into a node of a tree. This is an auxiliary function used to construct a GP tree starting from a mathematical expression
* \param            node *tree: operand to copy
* \param            node **el: pointer to the tree where the operand will be added
* \param 			node *parent: parent of the operand
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void copy_operand(node *tree, node **el, node *parent);


/*!
* \fn               void copy_operator(node *tree, node **el, node *parent)
* \brief            function used to copy an operator from the stack of operands into a node of a tree. This is an auxiliary function used to construct a GP tree starting from a mathematical expression
* \param            node *tree: operator to copy
* \param            node **el: pointer to the tree where the operator will be added
* \param 			node *parent: parent of the operator
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void copy_operator(node *tree, node **el, node *parent);


/*!
* \fn               int search(const char *str);
* \brief            function that returns the index of a terminal symbol and adds the terminal symbol to the array of symbol if the specified symbol does not exist
* \param            const char *str: symbolic name of the terminal symbol
* \return           int: the position of the terminal symbol in the array of symbols
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
int search(const char *str);


/*!
* \fn               node* parse_expression( char expression[]);
* \brief            function that builds a GP individual starting from the specified mathematical expression
* \param            char expression[]: mathematical expression that represents the GP individual
* \return           node*: the newly created GP individual
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
node* parse_expression( char expression[]);


/*!
* \fn               void evaluate_unseen_new_data(population **p, ofstream& OUT);
* \brief            function that evaluates the best model stored in trace.txt over newly provided unseen data
* \param            population **p: the initial GP population to be evaluated 
* \param            ofstream& OUT: file where the evaluation of the best model over newly provided unseen data is stored
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void evaluate_unseen_new_data(string dataset, population **p, ofstream& OUT);


/*!
* \fn               mark_trace();
* \brief            function that implements the marking procedure used to store the structure of the optimal solution
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void mark_trace();


/*!
* \fn               save_trace();
* \brief            function that store the information related to the reconstruction of the optimal solution on the file trace.txt
* \return           void
* \date             01/09/2016
* \author           Mauro Castelli
* \file             GP.h
*/
void save_trace(string dataset);



node* parse_expression( char expression[]){
    
    char token[2000][30];
	for(int k=0;k<1000;k++){
		for(int w=0;w<30;w++){
			token[k][w]='\0';
		}
	}
	node** stack_operators;
	node** stack_operands;
	stack_operators= new node* [1000];
	stack_operands= new node* [1000];
	int count_operands=0;
	int count_operators=0;
	unsigned int i=0;
	bool end_symbol;
	int count_token=0;
	node *result=NULL;
	while(i<strlen(expression)){
		int j=0;
		end_symbol=false;
		if(expression[i]==' ')
			i++;
		else{
			while(!end_symbol){
				//char symbol[30];
				for(int k=0;k<30;k++)
				//symbol[k]='\0';
				if(expression[i]=='(' || expression[i]==')' || expression[i]=='-' || expression[i]=='+' || expression[i]=='*' || expression[i]=='/'){
					node *el=new node;
					if(expression[i]=='('){
						el->root=symbols[0];
					}
					if(expression[i]==')'){
						el->root=symbols[1];
					}
					if(expression[i]=='+'){
						el->root=symbols[2];
					}
					if(expression[i]=='-'){
						el->root=symbols[3];
					}
					if(expression[i]=='*'){
						el->root=symbols[4];
					}
					if(expression[i]=='/'){
						el->root=symbols[5];
					}
					
					
					
					if(count_operators>0 && expression[i]!=')'){
					
                        node *x=new node;
						copy_operator(stack_operators[count_operators-1], (node **)&x,NULL);
						x->children=NULL;
						
						if(x->root->precedence < el->root->precedence || el->root==symbols[0]){
					
                            stack_operators[count_operators]=el;
							stack_operators[count_operators]->children=NULL;
					
                            count_operators++;
                            delete_individual(x);
					
						}

						
						else{
                            delete_individual(stack_operators[count_operators-1]);
							count_operators--;
					
                            int num_children=0;
							(x)->children=new node* [(x)->root->arity];
					
                            while(num_children<x->root->arity){
								
								node *op=new node;
					
								
								copy_operand(stack_operands[count_operands-1], (node **)&op,NULL);
								
                                delete_individual(stack_operands[count_operands-1]);
								count_operands--;
                                (x)->children[x->root->arity-1-num_children]=op;
								(x)->children[x->root->arity-1-num_children]->parent=x;
                                num_children++;
							}
					
							stack_operands[count_operands]=x;
							count_operands++;
                            el->children=NULL;
							stack_operators[count_operators]=el;
							count_operators++;
						}
					}
					if(expression[i]==')'){
                        bool open_par=false;
						do{
							node *oper=new node;
							copy_operator(stack_operators[count_operators-1], (node **)&oper,NULL);
							oper->children=NULL;
							delete_individual(stack_operators[count_operators-1]);
							count_operators--;
                            int num_children=0;
							oper->children=new node* [oper->root->arity];
                            if(oper->root!=symbols[0]){
                                while(num_children<oper->root->arity){
                                    node *op=new node;
									copy_operand(stack_operands[count_operands-1], (node **)&op,NULL);
                                    delete_individual(stack_operands[count_operands-1]);
									count_operands--;
                                    oper->children[oper->root->arity-1-num_children]=op;
                                    oper->children[oper->root->arity-1-num_children]->parent=oper;                                          
									num_children++;
                                }
								stack_operands[count_operands]=oper;
								count_operands++;
							}
							else{
								open_par=true;
								delete_individual(oper);
							}
						}while(!open_par);
						el->children=NULL;
						delete_individual(el);
					}
					if(count_operators==0 && expression[i]!=')'){
						
                        el->children=NULL;
						stack_operators[count_operators]=el;
						count_operators++;
						
					}
					i++;
				}
				else{
                    
					while(expression[i]!=' '){
						token[count_token][j]=expression[i];
						j++;
						i++;
					}
					if(strlen(token[count_token])>0){
						node *el=new node;
						el->root=symbols[search(token[count_token])];
						el->children=NULL;
						stack_operands[count_operands]=el;
						count_operands++;
						count_token++;
					}
				}
 				end_symbol=true;
			}
		}
	}

	if(count_operands>0){
		result=new node;
		copy_operand(stack_operands[count_operands-1], (node **)&result, NULL);
		delete_individual(stack_operands[count_operands-1]);
		count_operands--;
	}
	delete[] stack_operators;
	delete[] stack_operands;
	
    return result;
}


void copy_operand(node *tree, node **el, node *parent){
	if (tree==NULL){
		return;
	}
	(*el)->root=tree->root;
	(*el)->children=new node* [(*el)->root->arity];
	if(parent==NULL){
		(*el)->parent=NULL;
	}
	else{
		(*el)->parent=parent;
	}
	if(tree->children!=NULL){
		for (int i=0; i<(*el)->root->arity; i++){
			if(tree->children[i]!=NULL){
				(*el)->children[i]=new node;
				copy_operand(tree->children[i], ((node **)&((*el)->children[i])),(*el));
			}
		}
	}
}


void copy_operator(node *tree, node **el, node *parent){
	if (tree==NULL){
		return;
	}
	(*el)->root=tree->root;
	if(parent==NULL){
		(*el)->parent=NULL;
	}
	else{
		(*el)->parent=parent;
	}
}


int search(const char *str){
    
	for(int i=0; i<NUM_VARIABLE_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+NUM_CONSTANT_SYMBOLS;i++){
		if(strcmp(str,symbols[i]->name)==0)
			return i;	
	}
	char * buf=new char [strlen(str)+1];
			buf[strlen(str)]='\0';
            strcpy(buf,str);
            stringstream s;        
            s << buf;
            double v;
            s>>v;	
         	symbols.push_back(new symbol(0,0,symbols.size(),str,1));
         	symbols[symbols.size()-1]->value=v;
         	NUM_CONSTANT_SYMBOLS++;
         	delete [] buf;
	
    return symbols.size()-1;
	
}


void read_config_file(string dataset, cfg *config){
	string cfgname=dataset+"-configuration.ini";
	fstream f(cfgname.c_str(), ios::in);
	if (!f.is_open()) {
    		cerr<<"CONFIGURATION FILE ( " << cfgname << ") NOT FOUND." << endl;
    		exit(-1);
	}
	int k=0;
	while(!f.eof()){
		char str[100]="";
		char str2[100]="";
		int j=0;
		f.getline(str,100);
		if(str[0]!='\0'){
			while(str[j]!='='){
				j++;
			}
			j++;
			int i=0;
			while(str[j]==' '){
				j++;
			}
			while(str[j]!='\0'){
				str2[i] = str[j];
				j++;
				i++;
			}
		}
		if(k==0)
			config->population_size = atoi(str2);
		if(k==1)
			config->max_number_generations=atoi(str2); 
		if(k==2)
			config->init_type=atoi(str2);
		if(k==3)
			config->p_crossover=atof(str2);
		if(k==4)
			config->p_mutation=atof(str2);
		if(k==5)	
			config->max_depth_creation=atoi(str2);
		if(k==6)	
			config->tournament_size=atoi(str2);
		if(k==7)	
			config->zero_depth=atoi(str2);
		if(k==8)
			config->mutation_step=atof(str2);
		if(k==9){
			config->num_random_constants=atoi(str2);	
			NUM_CONSTANT_SYMBOLS=config->num_random_constants;
        }
        if(k==10)
			config->min_random_constant=atof(str2);
		if(k==11)
			config->max_random_constant=atof(str2);
		if(k==12)
			config->minimization_problem=atoi(str2);
		if(k==13)
			config->random_tree=atoi(str2);
		if(k==14)
			config->expression_file=atoi(str2);
		if(k==15)
			config->USE_TEST_SET=atoi(str2);	
        k++;        
	}	
    f.close();
    if(config->p_crossover<0 || config->p_mutation<0 || config->p_crossover+config->p_mutation>1){
        cerr << "ERROR: CROSSOVER RATE AND MUTATION RATE MUST BE GREATER THAN (OR EQUAL TO) 0 AND THEIR SUM SMALLER THAN (OR EQUAL TO) 1.";
		cerr << " (p_cross=" << config->p_crossover << ", p_mut=" << config->p_mutation << ")" << endl;
		//cout<<"ERROR: CROSSOVER RATE AND MUTATION RATE MUST BE GREATER THAN (OR EQUAL TO) 0 AND THEIR SUM SMALLER THAN (OR EQUAL TO) 1.";
        exit(-1);
    }
}


void create_T_F(){
	NUM_VARIABLE_SYMBOLS=nvar;
    NUM_FUNCTIONAL_SYMBOLS=4;
	NUM_AUXILIARY_FUNCTIONAL_SYMBOLS=2;
	symbols.push_back(new symbol(1,0,0,"(",-1));
	symbols.push_back(new symbol(1,0,1,")",-1));
	symbols.push_back(new symbol(1,2,2,"+",2));
    symbols.push_back(new symbol(1,2,3,"-",2));
    symbols.push_back(new symbol(1,2,4,"*",3));
    symbols.push_back(new symbol(1,2,5,"/",3));
    for(int i=NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS;i<NUM_VARIABLE_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS;i++){
		char str[50] = "x";
        char buf[50]="";
        sprintf(buf, "%d", i-NUM_FUNCTIONAL_SYMBOLS-NUM_AUXILIARY_FUNCTIONAL_SYMBOLS);
        strcat( str, buf);
		symbols.push_back(new symbol(0,0,i,str,1));
    }
    for(int i=NUM_VARIABLE_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS;i<NUM_VARIABLE_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS+NUM_CONSTANT_SYMBOLS;i++){
        	double a=config.min_random_constant+frand()*(config.max_random_constant-config.min_random_constant);
        	char buf [50]="";
            stringstream s;
            s << a;
            string f;
            s>>f;
            strcpy(buf,f.c_str());
         	symbols.push_back(new symbol(0,0,i,buf,1));
         	symbols[symbols.size()-1]->value=a;
    }
}


void print_math_style (node *el, string &s) {
	if(el==NULL)
		return;
	if (el->root->id<(NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS)){
        	s=s + "( ";
		switch (el->root->id) {
        	case 2:
            		print_math_style (el->children[0], s);
            		s = s + " + ";
                    print_math_style (el->children[1], s);
            	break;
        	case 3:
            		print_math_style (el->children[0], s);
            		s = s + " - ";
                    print_math_style (el->children[1], s);
            	break;
        	case 4:
            		print_math_style (el->children[0], s);
            		s = s + " * ";
                    print_math_style (el->children[1], s);
            	break;
		case 5:
            		print_math_style (el->children[0], s);
            		s = s + " / ";
                    print_math_style (el->children[1], s);
            	break;
        	}
		s = s + " )";
	}
	else{
		s = s + (string)(el->root->name);
	}
}



int choose_function(){
	int index;	
	index=NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+int(frand()*(NUM_FUNCTIONAL_SYMBOLS));
	return index;
}

int choose_terminal(){
    int index;
    if(NUM_CONSTANT_SYMBOLS==0){
        index=int(NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+NUM_FUNCTIONAL_SYMBOLS+frand()*(NUM_VARIABLE_SYMBOLS));
    }
    else{
        if(frand()<0.7)
            index=int(NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+frand()*(NUM_VARIABLE_SYMBOLS));
        else
            index=int(NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS+NUM_VARIABLE_SYMBOLS+frand()*(NUM_CONSTANT_SYMBOLS));
    }
    return index;
}


void create_grow_pop(population **p){
	int i=(*p)->num_ind;
	while(i<config.population_size){
		(*p)->individuals[i]=new node;
		create_grow_tree((node**)&((*p)->individuals[i]),0, NULL, config.max_depth_creation);
		i++;
	}
	
	for(int k=0; k<config.random_tree; k++){
				(*p)->individuals[i]=new node;
				create_grow_tree((node**)&((*p)->individuals[i]),0, NULL, config.max_depth_creation);
				i++;
				(*p)->num_ind++;
    }
	
}

void create_full_pop(population** p){
	int i=(*p)->num_ind;
	while(i<config.population_size){
		(*p)->individuals[i]=new node;
		create_full_tree((node**)&((*p)->individuals[i]),0, NULL, config.max_depth_creation);
		i++;
	}
	
	for(int k=0; k<config.random_tree; k++){
				(*p)->individuals[i]=new node;
				create_grow_tree((node**)&((*p)->individuals[i]),0, NULL, config.max_depth_creation);
				i++;
				(*p)->num_ind++;
    }
}

void create_ramped_pop(population **p){
	
	int sub_pop;
	int r;	
	int i=(*p)->num_ind;
	int min_depth;
	if(config.zero_depth==0){
		sub_pop=(config.population_size-(*p)->num_ind)/config.max_depth_creation;
		r=(config.population_size-(*p)->num_ind)%config.max_depth_creation;
		min_depth=1;	
	}
	else{
		sub_pop=(config.population_size-(*p)->num_ind)/(config.max_depth_creation+1);
		r=(config.population_size-(*p)->num_ind)%(config.max_depth_creation+1);	
		min_depth=0;
	}
	int j=config.max_depth_creation;
	while(j>=min_depth){
		if(j<config.max_depth_creation){
			for(int k=0; k<(int)(ceil((double)sub_pop/2)); k++){
				(*p)->individuals[i]=new node;
				create_full_tree((node**)&((*p)->individuals[i]),0, NULL, j);
				i++;
				(*p)->num_ind++;
			}
			for(int k=0; k<(int)(floor((double)sub_pop/2)); k++){
				(*p)->individuals[i]=new node;
				create_grow_tree((node**)&((*p)->individuals[i]),0, NULL ,j);
				i++;
				(*p)->num_ind++;					
			}
		}
		else{
			for(int k=0; k<(int)(ceil((double)(sub_pop+r)/2)); k++){
				(*p)->individuals[i]=new node;
				create_full_tree((node**)&((*p)->individuals[i]),0, NULL, j);
				i++;
				(*p)->num_ind++;						
			}
			for(int k=0; k<(int)(floor((double)(sub_pop+r)/2)); k++){
				(*p)->individuals[i]=new node;
				create_grow_tree((node**)&((*p)->individuals[i]),0, NULL, j);
				i++;
				(*p)->num_ind++;
			}
		}
		j--;	
	}
	
	for(int k=0; k<config.random_tree; k++){
				(*p)->individuals[i]=new node;
				create_grow_tree((node**)&((*p)->individuals[i]),0, NULL, config.max_depth_creation);
				i++;
				(*p)->num_ind++;
    }
}


void create_population(string dataset, population **p, int i){
	
	if(config.expression_file==1){
		
		int j=0;
		string indname=dataset+"-individuals.txt";
		fstream f(indname.c_str(), ios::in);
		if (!f.is_open()) {
	    		cerr<<"ERROR: FILE " << indname << " NOT FOUND." << endl;
    			//cerr<<"ERROR: FILE NOT FOUND." << endl;
    			exit(-1);
		}
		while(!f.eof()){
			node *el=NULL;	
			char line[1000];
			f.getline(line,1000);
			if(strlen(line)>0){
				el=parse_expression(line);
				if(el!=NULL){
					(*p)->individuals[j]=el;
					j++;
					(*p)->num_ind++;
				}
			}
		}
	}
	else{
		if(i==0)
			create_grow_pop((population **)&(*p));
		if(i==1)
			create_full_pop((population **)&(*p));
		if(i==2)
			create_ramped_pop(p);
	}
}


void create_grow_tree(node **el, int depth, node *parent, int max_depth){
	if(depth==0 && config.zero_depth==0){
		(*el)->root=symbols[choose_function()];
		(*el)->parent=NULL;
		(*el)->children=new node* [(*el)->root->arity];
		for (int i=0; i<(*el)->root->arity; i++){
			(*el)->children[i]=new node;
			create_grow_tree(((node **)&((*el)->children[i])), depth+1, *el, max_depth);
		}
		return;
	}
	if(depth==max_depth){
		(*el)->root=symbols[choose_terminal()];
		(*el)->parent=parent;
		(*el)->children=NULL;
		return;
	}
	if((depth>0 && depth<max_depth) || (depth==0 && config.zero_depth==1)){
		if(frand()>0.5){
			(*el)->root=symbols[choose_function()];
			(*el)->parent=parent;
			(*el)->children=new node* [(*el)->root->arity];
				for (int i=0; i<(*el)->root->arity; i++){
					(*el)->children[i]=new node;
					create_grow_tree(((node **)&((*el)->children[i])), depth+1, *el, max_depth);
				}
		}
		else{
			(*el)->root=symbols[choose_terminal()];
			(*el)->parent=parent;
			(*el)->children=NULL;
			return;
		}
	}
}


void create_full_tree(node **el, int depth, node *parent, int max_depth){
	if(depth==0 && depth<max_depth){
		(*el)->root=symbols[choose_function()];
		(*el)->parent=NULL;
		(*el)->children=new node* [(*el)->root->arity];
		for (int i=0; i<(*el)->root->arity; i++){
			(*el)->children[i]=new node;
			create_full_tree(((node **)&((*el)->children[i])), depth+1, *el, max_depth);
		}
		return;
	}
	if(depth==max_depth){
		(*el)->root=symbols[choose_terminal()];
		(*el)->parent=parent;
		(*el)->children=NULL;
		return;
	}
	if(depth>0 && depth<max_depth){
		(*el)->root=symbols[choose_function()];
		(*el)->parent=parent;
		(*el)->children=new node* [(*el)->root->arity];
		for (int i=0; i<(*el)->root->arity; i++){
			(*el)->children[i]=new node;
			create_full_tree(((node **)&((*el)->children[i])), depth+1, *el, max_depth);
		}
	}
}


double protected_division(double num, double den){
	if(den==0)
		return 1;
	else 
		return	(num/den);
}


double eval(node *tree){
	if(tree->root->type==1){
		if(strcmp(tree->root->name,"+")==0){
			return (eval(tree->children[0])+eval(tree->children[1]));
		}
		if(strcmp(tree->root->name,"-")==0){
			return (eval(tree->children[0])-eval(tree->children[1]));
		}
		if(strcmp(tree->root->name,"*")==0){
			return (eval(tree->children[0])*eval(tree->children[1]));
		}
		if(strcmp(tree->root->name,"/")==0){
			return protected_division(eval(tree->children[0]),eval(tree->children[1]));
		}
	}
	else{
		return (tree->root->value);
	}
	cerr<<"ERROR: UNDEFINED SYMBOL"<<endl;
    exit(-1);
}


void evaluate(population **p){
		(*p)->fitness[0]=Myevaluate((*p)->individuals[0]);
		(*p)->index_best=0;
		fit_.push_back((*p)->fitness[0]);
		fit_test.push_back(Myevaluate_test((*p)->individuals[0]));
    	for(int i=1; i<config.population_size; i++){
    		(*p)->fitness[i]=Myevaluate((*p)->individuals[i]);
    		fit_.push_back((*p)->fitness[i]);
	       	fit_test.push_back(Myevaluate_test((*p)->individuals[i]));
            if(better((*p)->fitness[i],(*p)->fitness[(*p)->index_best])){
                (*p)->index_best=i;
            }
        }
		
		for(int i=config.population_size; i<config.population_size+config.random_tree; i++){
			Myevaluate_random((*p)->individuals[i]);
			Myevaluate_random_test((*p)->individuals[i]);
		}
}


double Myevaluate (node *el) {
	double d=0;
    vector <double> val;
    for(int i=0;i<nrow;i++){
	       update_terminal_symbols(i);
	       set[i].res=eval(el);
	       val.push_back(set[i].res);
	       d+=fabs(set[i].res-set[i].y_value);
    }
    sem_train_cases.push_back(val);
    d=(d/nrow);
    return d;
}


double Myevaluate_test (node *el) {
	double d=0;
    vector <double> val;
    for(int i=nrow;i<nrow+nrow_test;i++){
        update_terminal_symbols(i);
        set[i].res=eval(el);
        val.push_back(set[i].res);
        d+=fabs(set[i].res-set[i].y_value);
    }
    sem_test_cases.push_back(val);
    d=(d/nrow_test);
    return d;
}


void Myevaluate_random (node *el){
    
	vector <double> val;
	for(int i=0;i<nrow;i++){
	       update_terminal_symbols(i);
	       set[i].res=eval(el);
           val.push_back(set[i].res);
    }
	fit_ran.push_back(val);
}


void Myevaluate_random_test(node *el) {

	vector <double> val;
    for(int i=nrow;i<nrow+nrow_test;i++){
        update_terminal_symbols(i);
        set[i].res=eval(el);
        val.push_back(set[i].res);
    }
	fit_ran_test.push_back(val);
}


void update_terminal_symbols(int i){
	for(int j=0; j<NUM_VARIABLE_SYMBOLS; j++){
        symbols[j+NUM_FUNCTIONAL_SYMBOLS+NUM_AUXILIARY_FUNCTIONAL_SYMBOLS]->value=set[i].vars[j];
	}
}


void delete_individual(node * el){
    if(el==NULL)
		return;
	if(el->children!=NULL){
		for(int i=0; i<el->root->arity; i++){
			delete_individual(el->children[i]);
		}
	}
	delete el;
}


int tournament_selection(){
    int *index=NULL;
	index=new int [config.tournament_size];
	for(int i=0;i<config.tournament_size;i++){
        index[i]=int(frand()*(config.population_size));
	}
	double best_fitness=fit_[index[0]];
	int best_index=index[0];
	for(int j=1;j<config.tournament_size;j++){
		double fit=fit_[index[j]];
		if(better(fit,best_fitness)){
			best_fitness=fit;
			best_index=index[j];
		}
	}
	delete[] index;
	return best_index;
}


void reproduction(int i, bool flag_mutation){
    if(i!=index_best){
        int p1=tournament_selection();
        sem_train_cases_new.push_back(sem_train_cases[p1]);
        fit_new.push_back(fit_[p1]);
        sem_test_cases_new.push_back(sem_test_cases[p1]);
        fit_new_test.push_back(fit_test[p1]);
		entry x;
		x.first_parent=p1;
		x.second_parent=-1;
		x.number=p1;
		x.event=-1;
		x.individual=i;
		x.mark=0;
		x.mut_step=0;
		traces_generation.push_back(x);
    }
    else{
        sem_train_cases_new.push_back(sem_train_cases[i]);
        fit_new.push_back(fit_[i]);
        sem_test_cases_new.push_back(sem_test_cases[i]);
        fit_new_test.push_back(fit_test[i]);
		entry x;
        x.first_parent=i;
        x.second_parent=-1;
        x.number=-1;
        x.event=-2;
        x.individual=i;
        x.mark=0;
		x.mut_step=0;
        traces_generation.push_back(x);
    }
}


void geometric_semantic_crossover(int i){
    if(i!=index_best){
        int p1=tournament_selection();
        int p2=tournament_selection();
	
		entry x;
        x.first_parent=p1;
        x.second_parent=p2;
        x.number=(int)(config.population_size+floor(frand()*config.random_tree));
        x.event=0;
        x.individual=i;
        x.mark=0;
		x.mut_step=0;
        traces_generation.push_back(x);
				
        vector <double> val;
        vector <double> val_test;
        for(int j=0;j<nrow;j++){
            double sigmoid=1.0/(1+exp(-(fit_ran[x.number-config.population_size][j])));
	        val.push_back(sem_train_cases[p1][j]*(sigmoid)+sem_train_cases[p2][j]*(1-sigmoid));
        }
        sem_train_cases_new.push_back(val);
        update_training_fitness(val,1);
				
        for(int j=0;j<nrow_test;j++){
            double sigmoid_test=1.0/(1+exp(-(fit_ran_test[x.number-config.population_size][j])));
	        val_test.push_back(sem_test_cases[p1][j]*(sigmoid_test)+sem_test_cases[p2][j]*(1-sigmoid_test));
        }
            sem_test_cases_new.push_back(val_test);
            update_test_fitness(val_test,1);
    }

    else{
        sem_train_cases_new.push_back(sem_train_cases[i]);
        fit_new.push_back(fit_[i]);
        sem_test_cases_new.push_back(sem_test_cases[i]);
        fit_new_test.push_back(fit_test[i]);
		entry x;
        x.first_parent=i;
        x.second_parent=-1;
        x.number=-1;
        x.event=-2;
        x.individual=i;
        x.mark=0;
		x.mut_step=0;
        traces_generation.push_back(x);
    }
}

void geometric_semantic_mutation(int i){
    if(i!=index_best){
     	
		double mut_step=frand();
		
		entry x;
        x.first_parent=(int)(config.population_size+floor(frand()*config.random_tree));
        x.second_parent=(int)(config.population_size+floor(frand()*config.random_tree));
        x.number=traces_generation[traces_generation.size()-1].first_parent;
        x.event=1;
        x.individual=i;
        x.mark=0;
		x.mut_step=mut_step;
		traces_generation[traces_generation.size()-1]=x;
		
		for(int j=0;j<nrow;j++){
            double sigmoid_1=1.0/(1+exp(-(fit_ran[x.first_parent-config.population_size][j])));
            double sigmoid_2=1.0/(1+exp(-(fit_ran[x.second_parent-config.population_size][j])));
			sem_train_cases_new[i][j]=sem_train_cases_new[i][j]+mut_step*(sigmoid_1-sigmoid_2);
        }

        update_training_fitness(sem_train_cases_new[i],0);
		
        for(int j=0;j<nrow_test;j++){
    	    double sigmoid_test_1=1.0/(1+exp(-(fit_ran_test[x.first_parent-config.population_size][j])));
    	    double sigmoid_test_2=1.0/(1+exp(-(fit_ran_test[x.second_parent-config.population_size][j])));
			sem_test_cases_new[i][j]=sem_test_cases_new[i][j]+mut_step*(sigmoid_test_1-sigmoid_test_2);
        }
        update_test_fitness(sem_test_cases_new[i],0);
    }
}



void update_training_fitness(vector <double> semantic_values, bool crossover){
    double d=0;
    for(int j=0;j<nrow;j++){
        d+=fabs(semantic_values[j]-set[j].y_value);
    }
    if(crossover==1)
        fit_new.push_back((d/nrow));
    else    
        fit_new[fit_new.size()-1]=(d/nrow);
}


void update_test_fitness(vector <double> semantic_values, bool crossover){
    double d=0;
    for(int j=nrow;j<nrow+nrow_test;j++){
        d+=fabs(semantic_values[j-nrow]-set[j].y_value);
    }
    if(crossover == 1)
        fit_new_test.push_back((d/nrow_test));
    else    
        fit_new_test[fit_new_test.size()-1]=(d/nrow_test);
}


int best_individual(){
    double best_fitness=fit_[0];
    int best_index=0;
    for(unsigned int i=0;i<fit_.size();i++){
        if(better(fit_[i],best_fitness)){
            best_fitness=fit_[i];
            best_index=i;
        }
   }
   return best_index;
}


void update_tables(){
	fit_.clear();
   	fit_.assign(fit_new.begin(),fit_new.end());
   	fit_new.clear();
   	sem_train_cases.clear();
    sem_train_cases.assign(sem_train_cases_new.begin(),sem_train_cases_new.end());
    sem_train_cases_new.clear();
   	fit_test.clear();
   	fit_test.assign(fit_new_test.begin(),fit_new_test.end());
   	fit_new_test.clear();
   	sem_test_cases.clear();
   	sem_test_cases.assign(sem_test_cases_new.begin(),sem_test_cases_new.end());
   	sem_test_cases_new.clear();
}


void read_input_data(char *train_file, char *test_file){
	fstream in(train_file,ios::in);
	if (!in.is_open()) {
        cerr<<"TRAINING FILE ( " << train_file << ") NOT FOUND." << endl;
		//cout<<endl<<"ERROR: TRAINING FILE NOT FOUND." << endl;
    	exit(-1);
	}
	fstream in_test(test_file,ios::in);
	if (!in_test.is_open()) {
        cerr<<"TEST FILE ( " << test_file << ") NOT FOUND." << endl;
		//cout<<endl<<"ERROR: TEST FILE NOT FOUND." << endl;
    	exit(-1);
	}
    char str[255];
	in >> str;
	nvar = atoi(str);
	in_test >> str;
	nvar_test = atoi(str);
    in >> str;
	nrow = atoi(str);
	in_test >> str;
	nrow_test = atoi(str);
    set = new Instance[nrow+nrow_test];
	   for (int i=0;i<nrow;i++) {
        set[i].vars = new double[nvar];
	    for (int j=0; j<nvar; j++) {
            in >> str;
            set[i].vars[j] = atof(str);
        }
        in >> str;
        set[i].y_value = atof(str);
   	}
	in.close();
	for (int i=nrow;i<nrow+nrow_test;i++) {
        set[i].vars = new double[nvar];
	    for (int j=0; j<nvar; j++) {
            in_test >> str;
            set[i].vars[j] = atof(str);
    	}
        in_test >> str;
       	set[i].y_value = atof(str);
	}
	in_test.close();
}


bool better (double f1, double f2){
    if(config.minimization_problem==1){
        if(f1<f2)
            return true;
        else
            return false;
    }
    else{
        if(f1>f2)
            return true;
        else
            return false;
    }
}



void mark_trace(){
    vector_traces[vector_traces.size()-1][index_best].mark=1;
    for(int i=vector_traces.size()-1;i>0;i--){
        for(int j=0;j<config.population_size;j++){
            if (vector_traces[i][j].mark==1 && vector_traces[i][j].event==0){
                vector_traces[i-1][vector_traces[i][j].first_parent].mark = 1;
                vector_traces[i-1][vector_traces[i][j].second_parent].mark = 1;
            }
            if (vector_traces[i][j].mark==1 && vector_traces[i][j].event==1){
                vector_traces[i-1][vector_traces[i][j].number].mark = 1;
            }
            if (vector_traces[i][j].mark==1 && (vector_traces[i][j].event==-1 || vector_traces[i][j].event==-2)){
                vector_traces[i-1][vector_traces[i][j].first_parent].mark = 1;
            }
        }
    }
}

void save_trace(string dataset){
    string trfile=dataset+"-trace.txt";
	ofstream trace(trfile.c_str(),ios::out);
    for(int unsigned i=0;i<vector_traces.size();i++){
        for(int j=0;j<config.population_size;j++){
            if (vector_traces[i][j].mark==1){
                trace<<vector_traces[i][j].first_parent<<"\t"<<vector_traces[i][j].second_parent<<"\t"<<vector_traces[i][j].number<<"\t"<<vector_traces[i][j].event<<"\t"<<vector_traces[i][j].individual<<"\t"<<vector_traces[i][j].mut_step<<endl;				
			}
        }
		if(i<vector_traces.size()-1)
			trace<<"***"<<endl;
		else	
			trace<<"***";
    }
}

void evaluate_unseen_new_data(string dataset, population **p, ofstream& OUT){
	
	vector <double> eval_random;
	vector <double> eval_;
	vector <double> eval_new;
	vector <double> fit_;
	int best=0;

	for(int i=0; i<config.population_size; i++)
		eval_.push_back(-1);
		
	for(int i=0; i<config.random_tree; i++)
		eval_random.push_back(-1);
           
    string trfile=dataset+"-trace.txt";
	fstream in(trfile.c_str(),ios::in);
    if(!in.is_open()) {
    	cerr<<"TRACE FILE ( " << trfile << ") NOT FOUND." << endl;
		//cout<<endl<<"ERROR: FILE trace.txt NOT FOUND." << endl;
    	exit(-1);
    }
    else{
        char str[255];
        while(true){                
            in >> str;
            if(strcmp(str,"***")==0){
                break;
            }
            int index1 = atoi(str); 
            in >> str;
            int index2 = atoi(str); 
            in >> str;
            int index3 = atoi(str); 
            in >> str;
            int index4 = atoi(str); 
            in >> str;
            int index5 = atoi(str); 
			in >> str;
			double index6 = atof(str); 
            if(index4==0){ 
                eval_random[index3-config.population_size]=eval((*p)->individuals[index3]);
                eval_[index1]=eval((*p)->individuals[index1]);
                eval_[index2]=eval((*p)->individuals[index2]);
                double sigmoid=1.0/(1+exp(-(eval_random[index3-config.population_size])));
                eval_[index5]=eval_[index1]*(sigmoid)+eval_[index2]*(1-sigmoid);
            }
            if(index4==1){ 
                eval_random[index1-config.population_size]=eval((*p)->individuals[index1]);
                eval_random[index2-config.population_size]=eval((*p)->individuals[index2]);
				eval_[index3]=eval((*p)->individuals[index3]);
                eval_[index5]=((eval_[index3]+index6*((1.0/(1+exp(-eval_random[index1-config.population_size])))-(1/(1+exp(-eval_random[index2-config.population_size]))))));
            }
            if(index4==-1 || index4==-2){ 
				eval_[index1]=eval((*p)->individuals[index1]);
                eval_[index5]=eval_[index1];
            }
		}
			
       	while(!in.eof()){
            for(int i=0; i<config.population_size; i++){
                eval_new.push_back(-1);
            }
            while(true){
                in >> str;
                if(strcmp(str,"***")==0){
                    break;
                }
                int index1 = atoi(str); 
                in >> str;
                int index2 = atoi(str); 
                in >> str;
                int index3 = atoi(str); 
                in >> str;
                int index4 = atoi(str); 
                in >> str;
                int index5 = atoi(str); 
				in >> str;
				double index6 = atof(str); 
                
				if(index4==0){ 					
					eval_random[index3-config.population_size]=eval((*p)->individuals[index3]);                   
					double sigmoid=1.0/(1+exp(-(eval_random[index3-config.population_size])));
                    eval_new[index5]=eval_[index1]*(sigmoid)+eval_[index2]*(1-sigmoid);
                    best=index5;
                }
				
                if(index4==1){                     
                    eval_random[index1-config.population_size]=eval((*p)->individuals[index1]);
                    eval_random[index2-config.population_size]=eval((*p)->individuals[index2]);
                    eval_new[index5]=eval_[index3]+index6*((1.0/(1+exp(-eval_random[index1-config.population_size])))-(1/(1+exp(-eval_random[index2-config.population_size]))));
                    best=index5;    
                }
				
                if(index4==-1){
                    eval_new[index5]=eval_[index1];
					best=index5;
                }
				
				if(index4==-2){
                    eval_new[index5]=eval_[index5];                           
					best=index5;
                }      
            }
            eval_.clear();
            eval_.assign(eval_new.begin(),eval_new.end());
			eval_new.clear();			
        }		
    }
    OUT<<eval_[best]<<endl;
}