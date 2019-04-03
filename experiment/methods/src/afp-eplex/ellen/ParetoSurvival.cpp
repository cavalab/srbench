// Adapted from SPEA2 source code written by Marco Laumanns. PISA: (www.tik.ee.ethz.ch/pisa/)
#include "stdafx.h"
#include "pop.h"
#include "params.h"
#include "rnd.h"
#include "pareto.h"
#include "state.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#define SIZEOF_ARRAY( arr ) sizeof( arr ) / sizeof( arr[0] )
#define MAX_DOUBLE numeric_limits<double>::max( )
void* chk_malloc(size_t size)
/* Wrapper function for malloc(). Checks for failed allocations. */
{
    void *return_value = malloc(size);
    if(return_value == NULL)
	std::cerr << "SPEA2: Out of memory.\n";
    return (return_value);
}
/* SPEA2 internal global variables */
class SPEA2{
public:
	vector<int> fitness_bucket;
	vector<int> fitness_bucket_mod;
	vector<int> copies;
	//int *old_index;
	vector<vector<double>> dist;
	vector<vector<int>>  NN;
	int dim;
	SPEA2(){}
	SPEA2(int size, int d)
	{
		fitness_bucket.resize(size*size,0);
		fitness_bucket_mod.resize(size,0);
		copies.resize(size,1);
		for(int i=0;i<size;++i){
			dist.push_back(vector<double>(size,-1));
			NN.push_back(vector<int>(size,-1));
		}
		dim = d;
	}
	~SPEA2(){}

};
//#define PISA_MAXDOUBLE = 100000000

void free_ind(vector<ind>& pop,int i)
{
  if (pop.size() > i){
  	pop[i].swap(pop.back());
  	pop.pop_back();
  }
}
bool is_equal(ind& p_ind_a, ind& p_ind_b)
/* Determines if two individuals are equal in all objective values.*/
{

     bool equal = 1;

     for (size_t i = 0; i < p_ind_a.f.size() && equal; ++i)
		equal = (p_ind_a.f[i] == p_ind_b.f[i]);

     return (equal);
}
//int dominates(ind& p_ind_a, ind& p_ind_b);
//void calcFitnesses(vector<ind>& pop);
//void calcDistances(vector<ind>& pop);
//void environmentalSelection(vector<ind>& pop,int alpha);
//void truncate_nondominated(vector<ind>& pop,int alpha);
//void truncate_dominated(vector<ind>& pop, int alpha);

bool dominates(ind& p_ind_a, ind& p_ind_b);
void calcFitnesses(vector<ind>& pop, SPEA2& S);
double calcDistance(ind& p_ind_a, ind& p_ind_b);
void calcDistances(vector<ind>& pop, SPEA2& S);
void environmentalSelection(vector<ind>& pop, int alpha, SPEA2& S);
int getNN(int index, int k, int size, SPEA2& S,vector<ind>& pop);
double getNNd(int index, int k, int size, SPEA2& S,vector<ind>& pop);
int irand(int range);
void truncate_nondominated(vector<int>& marked_inds, vector<ind>& pop, int& alpha, SPEA2& S);
void truncate_dominated(vector<int>& marked_inds, vector<ind>& pop, int& alpha, SPEA2& S);

float var(vector<float>& x) {
	// get mean of x
	float mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
	//calculate variance
	float var = 0;
	for (vector<float>::iterator it = x.begin(); it != x.end(); ++it)
		var += pow(*it - mean, 2);

	var /= (x.size() - 1);
	return var;
}
void ParetoSurvival(vector<ind>& pop,params& p,vector<Randclass>& r,state& s)
{
	// implements SPEA2 environmental selection based on pareto strength and fitness
	// calc SPEA2 fitness (calcFitnesses())
	// calc SPEA2 distance (calcDistances())
	// truncate pop to original pop size (environmentalSelection())
	//debugging
	vector<int> ages;
	vector<float> fitnesses;
	for (size_t i =0; i<pop.size(); ++i){
		ages.push_back(pop[i].age);
		fitnesses.push_back(pop[i].fitness);
	}
	//boost::progress_timer timer;
	float max_fit=0;
	float max_age=0;
	float max_genty=0;
	float max_complexity=0;
	float max_dim=0;
	vector<float> max_error(pop[0].error.size(),0);
	for (size_t i =0; i<pop.size(); ++i){
		if (pop[i].fitness>max_fit) max_fit=pop[i].fitness;
		if (pop[i].age>max_age) max_age=pop[i].age;
		if (pop[i].genty>max_genty) max_genty=pop[i].genty;
		if (pop[i].complexity>max_complexity) max_complexity = pop[i].complexity;
		if (pop[i].dim>max_dim) max_dim = pop[i].dim;
		for (unsigned j=0;j<pop[i].error.size();++j){
			if (pop[i].error[j]>max_error[j])
				max_error[j] = pop[i].error[j];
		}
	}
	if (p.PS_sel == 7) // age, fitness, variance
	{
		for (size_t i = 0; i<pop.size(); ++i) {
			pop[i].f.resize(0);
			pop[i].f.push_back(pop[i].fitness / max_fit);
			pop[i].f.push_back(pop[i].age / max_age);
			pop[i].f.push_back(var(pop[i].output));
		}
	}
	else if(p.PS_sel==6 && p.classification) // objectives are the class-based fitnesses and dimensionality
	{

		for (size_t i = 0; i<pop.size(); ++i){
			pop[i].f.resize(0);
			for (unsigned j=0;j<pop[i].error.size();++j)
				pop[i].f.push_back(pop[i].error[j]/max_error[j]);
			pop[i].f.push_back(pop[i].dim/max_dim);
		}
	}
	if(p.PS_sel==5 && p.classification) // objectives are the class-based fitnesses and age
	{

		for (size_t i = 0; i<pop.size(); ++i){
			pop[i].f.resize(0);
			for (unsigned j=0;j<pop[i].error.size();++j)
				pop[i].f.push_back(pop[i].error[j]/max_error[j]);
			pop[i].f.push_back(pop[i].age/max_age);
		}
	}
	if(p.PS_sel==4 && p.classification) // objectives are the class-based fitnesses
	{

		for (size_t i = 0; i<pop.size(); ++i){
			pop[i].f.resize(0);
			for (unsigned j=0;j<pop[i].error.size();++j)
				pop[i].f.push_back(pop[i].error[j]/max_error[j]);
		}
	}
	else if(p.PS_sel==3) // fitness, age, complexity
	{
		for (size_t i = 0; i<pop.size(); ++i){
			pop[i].f.resize(0);
			pop[i].f.push_back(pop[i].fitness/max_fit);
			pop[i].f.push_back(pop[i].age/max_age);
			pop[i].f.push_back(float(pop[i].complexity)/max_complexity);
		}
	}
	else if(p.PS_sel==2) // fitness, age, generality
	{
		for (size_t i = 0; i<pop.size(); ++i){
			pop[i].f.resize(0);
			pop[i].f.push_back(pop[i].fitness/max_fit);
			pop[i].f.push_back(pop[i].age/max_age);
			pop[i].f.push_back(pop[i].genty/max_genty);
		}
	}
	else // fitness, age
	{
		for (size_t i = 0; i<pop.size(); ++i){
			pop[i].f.resize(0);
			pop[i].f.push_back(pop[i].fitness/max_fit);
			pop[i].f.push_back(pop[i].age/max_age);
		}
	}
	SPEA2 S(pop.size(),pop[0].f.size());
	calcFitnesses(pop,S);
	//calcDistances(pop,S); //distances are calculated on a case-by-case basis to keep costs down
	environmentalSelection(pop,(pop.size()-1)/2,S);

	ages.resize(0);
	fitnesses.resize(0);
	for (size_t i =0; i<pop.size(); ++i){
		ages.push_back(pop[i].age);
		fitnesses.push_back(pop[i].fitness);
	}
	//data.reserve(6);
	//
	//vector<int> fitindex(2);
	////float minfit=p.max_fit;
	////float minage=p.g; // maximum age = total max generations
	//int loser;
	//int counter =0;
	//bool draw=true;
	//int popsize = (int)floor((float)pop.size()/2);
	////vector<ind> tourn_pop;
	////tourn_pop.reserve(2);
	//// remove all frontier pareto individuals from the tournament and give them a free pass:
	////vector<int> tournament; for(int i=0; i<pop.size(); ++i){tournament.push_back(i);}

	//while (pop.size()>popsize && counter<p.popsize*10)//pow(float(p.popsize),2))
	//{
	//
	//	for (int j=0;j<2;++j)
	//	{
	//		fitindex.at(j)=r[omp_get_thread_num()].rnd_int(0,pop.size()-1);
	//		//tourn_pop.push_back(pop.at(fitindex.at(j)));
	//		data.push_back(pop.at(fitindex.at(j)).fitness);
	//		data.push_back(pop.at(fitindex.at(j)).age);
	//		if (p.PS_sel==2) data.push_back(pop.at(fitindex.at(j)).genty);
	//	}
	//	pareto2(data,pop.at(fitindex.at(0)).rank,pop.at(fitindex.at(1)).rank);
	//	//pareto(tourn_pop,data);
	//	//float genty1 = abs(tourn_pop[0].fitness-tourn_pop[0].fitness_v)/tourn_pop[0].fitness;
	//	//float genty2 = abs(tourn_pop[1].fitness-tourn_pop[1].fitness_v)/tourn_pop[1].fitness;
	//	if (pop.at(fitindex.at(0)).rank != pop.at(fitindex.at(1)).rank)
	//	{
	//		if (pop.at(fitindex.at(0)).rank==1)
	//			loser = fitindex[1];
	//		else
	//			loser = fitindex[0];
	//		draw=false;
	//	}
	//
	//	// delete losers from population
	//	if(!draw){
	//		pop.at(loser).swap(pop.back());
	//		//std::swap(pop.at(loser),pop.back());
	//		pop.pop_back();
	//		//pop.erase(pop.begin()+loser);
	//	}

	//	draw=true;
	//	counter++;
	//	//tourn_pop.resize(0);
	//	data.resize(0);
	//}
	//if (pop.size()>popsize){
	//	//s.out << "warning: pareto survival tournament exceeded maximum comparisons.\n";
	//	if (p.PS_sel==1){
	//		sort(pop.begin(),pop.end(),SortAge());
	//		stable_sort(pop.begin(),pop.end(),SortFit());
	//	}
	//	else if(p.PS_sel==2){
	//		sort(pop.begin(),pop.end(),SortAge());
	//		stable_sort(pop.begin(),pop.end(),SortGenty());
	//		stable_sort(pop.begin(),pop.end(),SortFit());
	//	}
	//}
	//while(pop.size()>popsize){
	//	//pop.erase(pop.end()-1);
	//	pop.pop_back();
	//}


}

bool dominates(ind& p_ind_a, ind& p_ind_b)
/* Determines if one individual dominates another.
   Minimizing fitness values. */
{

    bool a_is_worse = 0;
    bool equal = 1;

     for (size_t i = 0; i < p_ind_a.f.size() && !a_is_worse; ++i)
     {
		 a_is_worse = p_ind_a.f[i] > p_ind_b.f[i];
         equal = (p_ind_a.f[i] == p_ind_b.f[i]) && equal;
     }

     return (!equal && !a_is_worse);
}
void calcFitnesses(vector<ind>& pop, SPEA2& S)
{
    int i, j;
    int size;
    //int *strength;

    size = pop.size();
    //strength = (int*) chk_malloc(size * sizeof(int));
    vector<int> strength(size,0);
    /* initialize fitness and strength values */
  //  for (i = 0; i < size; i++)
  //  {
  //      pop[i].spea_fit = 0;
		////S.fitness_bucket[i] = 0;
		////S.fitness_bucket_mod[i] = 0;
		///*for (j = 0; j < size; j++)
		//{
		//	S.fitness_bucket[i * size + j] = 0;
		//}*/
  //  }

    /* calculate strength values */
    for (i = 0; i < size; i++)
    {
        for (j = 0; j < size; j++)
		{
			if (i!=j){
            if (dominates(pop[i], pop[j]))
			{
                strength[i]++;
				pop[j].dominated.push_back(i);
			}
			}
		}
    }

    /* Fitness values =  sum of strength values of dominators */
    for (i = 0; i < size; ++i)
    {
        int sum = 0;
        /*for (j = 0; j < size; j++)
		{
			if (i!=j){
            if (dominates(pop[j], pop[i]))
			{
				sum += strength[j];
			}
			}
		}*/
		for (j=0; j<pop[i].dominated.size();++j)
			sum += strength[pop[i].dominated[j]];

		pop[i].spea_fit = sum;
		S.fitness_bucket[sum]++;
		S.fitness_bucket_mod[(sum / size)]++;
		pop[i].dominated.resize(0);
    }

    //free(strength);
    //strength = NULL;

    return;
}
double calcDistance(ind& p_ind_a, ind& p_ind_b)
{
    int i;
    double distance = 0;

    if (is_equal(p_ind_a, p_ind_b))
    {
		return (0);
    }

    for (i = 0; i < p_ind_a.f.size(); i++)
	distance += pow(p_ind_a.f[i]-p_ind_b.f[i],2);

    if (0.0 == distance)
        distance = 0.00000000001;

    return (sqrt(distance));
}
void calcDistances(vector<ind>& pop, SPEA2& S)
{
    int i, j;
    int size = pop.size();

    /* initialize copies[] vector and S.NN[][] matrix */
    /*for (i = 0; i < size; i++)
    {
		S.copies[i] = 1;
		for (j = 0; j < size; j++)
		{
			S.NN[i][j] = -1;
		}
    }*/

    /* calculate distances */
    for (i = 0; i < size; i++)
    {
		S.NN[i][0] = i;
		for (j = i + 1; j < size; j++)
		{
			S.dist[i][j] = calcDistance(pop[i], pop[j]);
			//if(boost::math::isinf(S.dist[i][j]) || S.dist[i][j]>MAX_DOUBLE) S.dist[i][j]=MAX_DOUBLE;
			assert(S.dist[i][j] < MAX_DOUBLE);
			S.dist[j][i] = S.dist[i][j];
			if (S.dist[i][j] == 0)
			{
			S.NN[i][S.copies[i]] = j;
			S.NN[j][S.copies[j]] = i;
			S.copies[i]++;
			S.copies[j]++;
			}
		}
		S.dist[i][i] = 0;
    }
}
//int calcNN(SPEA2& S, int i, int j)
//{
//	//return S.NN[i][j] given S.dist[i][j]
//	if (j==0)
//		return i;
//	else if (S.dist[i][j] ==0){
//		S.NN[i][S.copies[i]] = j;
//		S.NN[j][S.copies[j]] = i;
//
//}
void environmentalSelection(vector<ind>& pop, int alpha, SPEA2& S)
{
    int i;
    int new_size = 0;
	  bool td;
    vector<int> marked_inds;

    if (S.fitness_bucket[0] > alpha){
  		//calcDistances(pop,S);
  		truncate_nondominated(marked_inds,pop,alpha,S);
  		td=false;
    }
    else if (pop.size() > alpha){
  		//calcDistances(pop,S);
  		truncate_dominated(marked_inds,pop,alpha,S);
  		td=true;
  	}

    sort(marked_inds.begin(),marked_inds.end());
	  // move remaining pop to top of pop (plop slop)
  	for (i = marked_inds.size()-1; i>=0; --i)
	 	 free_ind(pop,marked_inds[i]);

	  assert(pop.size() == alpha);

    return;
}
int getNN(int index, int k, int size, SPEA2& S,vector<ind>& pop)
/* lazy evaluation of the k-th nearest neighbor
   pre-condition: (k-1)-th nearest neigbor is known already */
{
    assert(index >= 0);
    assert(k >= 0);
    assert(S.copies[index] > 0);
    /*if (S.dist[index][k]==-1){
		S.dist[index][k] = calcDistance(pop[index],pop[k]);
		S.NN[index][k] = calcNN(S,index,k);
	}*/
    if (S.NN[index][k] < 0)
    {
		int i;
		double min_dist = MAX_DOUBLE;
		int min_index = -1;
		int prev_min_index = S.NN[index][k-1];
		double prev_min_dist = S.dist[index][prev_min_index];
		/*double prev_min_dist;
		if (S.dist[index][prev_min_index]!=-1)
			prev_min_dist = S.dist[index][prev_min_index];
		else{
			prev_min_dist = calcDistance(pop[index],pop[prev_min_index]);
			S.dist[index][prev_min_index] = prev_min_dist;
		}*/
		assert(prev_min_dist >= 0);
		double my_dist;

		for (i = 0; i < size; i++)
		{
			my_dist = S.dist[index][i];
			//my_dist = calcDistance(pop[index],pop[i]);
			/*if (S.dist[index][i]!=-1)
				my_dist = S.dist[index][i];
			else{
				my_dist = calcDistance(pop[index],pop[i]);
				S.dist[index][i] = my_dist;
			}*/
			if (my_dist < min_dist && index != i)
			{
				if (my_dist > prev_min_dist || (my_dist == prev_min_dist && i > prev_min_index))
				{
					min_dist = my_dist;
					min_index = i;
				}
			}
		}
		/*if (min_index==-1)
			cout << "\n";*/

		S.NN[index][k] = min_index;
    }

    return (S.NN[index][k]);
}
double getNNd(int index, int k, int dim, SPEA2& S, vector<ind>& pop)
/* Returns the distance to the k-th nearest neigbor
   if this individual is still in the population.
   For for already deleted individuals, returns -1 */
{
    int neighbor_index = getNN(index, k, dim, S,pop);

    if (S.copies[neighbor_index] == 0)
		return (-1);
    else
		return (S.dist[index][neighbor_index]);
}
int irand(int range)
/* Generate a random integer. */
{
    int j;
    j=(int) ((double)range * (double) rand() / (RAND_MAX+1.0));
    return (j);
}
void truncate_nondominated(vector<int>& marked_inds, vector<ind>& pop, int& alpha, SPEA2& S)
/* truncate from nondominated individuals (if too many) */
{
    int i;

    /* delete all dominated individuals */
    for (i = 0; i<pop.size(); ++i)
    {
		if (pop[i].spea_fit > 0)
		{
			marked_inds.push_back(i);
			//free_ind(pop,i);
			//pop[i] = NULL;
			S.copies[i] = 0;
		}
    }
    bool dist_calc=false;
    /* truncate from non-dominated individuals */
    while (S.fitness_bucket[0] > alpha)
    {
		//int *marked;
		int max_copies = 0;
		int count = 0;
		int delete_index;

		//marked = (int*) chk_malloc(pop.size() * sizeof(int));
		vector<int> marked(pop.size());
		/* compute inds with maximal copies */
		for (i = 0; i < pop.size(); i++)
		{
			if (S.copies[i] > max_copies)
			{
			count = 0;
			max_copies = S.copies[i];
			}
			if (S.copies[i] == max_copies)
			{
			marked[count] = i;
			count++;
			}
		}

		assert(count >= max_copies);

		if (count > max_copies)
		{
			if (!dist_calc){
				calcDistances(pop,S);
				dist_calc=true;
			}
			//int *neighbor;
			//neighbor = (int*) chk_malloc(count * sizeof(int));
			vector<int> neighbor(count);
			for (i = 0; i < count; i++)
			{
			neighbor[i] = 1;  /* pointers to next neighbor */
			}

			while (count > max_copies)
			{
				double min_dist = MAX_DOUBLE;
				int count2 = 0;

				for (i = 0; i < count; i++)
				{
					double my_dist = -1;
					while (my_dist == -1 && neighbor[i] < pop.size())
					{
						my_dist = getNNd(marked[i],neighbor[i],pop.size(),S,pop);
						neighbor[i]++;
					}

					if (my_dist < min_dist)
					{
						count2 = 0;
						min_dist = my_dist;
					}
					if (my_dist == min_dist)
					{
						marked[count2] = marked[i];
						neighbor[count2] = neighbor[i];
						count2++;
					}
				}
				count = count2;
				if (min_dist == -1) /* all have equal distances */
				{
					break;
				}
			}
			//free(neighbor);
			//    neighbor = NULL;
		}

		/* remove individual from population */
		delete_index = marked[irand(count)];
		marked_inds.push_back(delete_index);
		//free_ind(pop,delete_index);
		//pop[delete_index] = NULL;
		for (i = 0; i < count; i++)
		{
			if (S.dist[delete_index][marked[i]] == 0)
			{
				S.copies[marked[i]]--;
			}
			else if (S.dist[delete_index][marked[i]] == -1 && calcDistance(pop[delete_index],pop[marked[i]])==0)
			{
				S.dist[delete_index][marked[i]] =0;
				S.copies[marked[i]]--;
			}

		}
		S.copies[delete_index] = 0; /* Indicates that this index is empty */
		S.fitness_bucket[0]--;
		S.fitness_bucket_mod[0]--;
    }

    return;
}
void truncate_dominated(vector<int>& marked_inds, vector<ind>& pop, int& alpha, SPEA2& S)
/* truncate from dominated individuals */
{
    int i, j;
    int size = pop.size();
    int num = 0;
   // size = pop.size();

    i = -1;
    while (num < alpha)
    {
		++i;
		num += S.fitness_bucket_mod[i];
    }

    j = i * size;
    num = num - S.fitness_bucket_mod[i] + S.fitness_bucket[j];
    while (num < alpha)
    {
		++j;
		num += S.fitness_bucket[j];
    }

    if (num == alpha)
    {
		for (i = size-1; i >= 0; --i)
		{
			if (pop[i].spea_fit > j)
				marked_inds.push_back(i);
				//free_ind(pop,i);
		}
    }
    else /* if not all fit into the next generation */
    {
		calcDistances(pop,S);
		int k;
		int free_spaces;
		int fill_level = 0;
		//int *best = NULL;

		free_spaces = alpha - (num - S.fitness_bucket[j]);
		//best = (int*) chk_malloc(free_spaces * sizeof(int));
		vector<int> best(free_spaces);
		//for (i = size-1; i >= 0; --i)
		for (i = 0; i < size; ++i)
		{
			if (pop[i].spea_fit > j)
			{
				marked_inds.push_back(i);
				//pop[i].spea_fit=-1;
			//free_ind(pop,i);
			//pop[i] = NULL;
			}
			else if (pop[i].spea_fit == j)
			{
				if (fill_level < free_spaces)
				{
					best[fill_level] = i;
					fill_level++;
					for (k = fill_level - 1; k > 0; k--)
					{
					int temp;
					if (getNNd(best[k], 1,pop.size(),S,pop) <= getNNd(best[k - 1], 1,pop.size(),S,pop))
					{
						break;
					}
					temp = best[k];
					best[k] = best[k-1];
					best[k-1] = temp;
				}
			}
				else
				{
					if (getNNd(i, 1,pop.size(),S,pop) <= getNNd(best[free_spaces - 1], 1,pop.size(),S,pop))
					{
						marked_inds.push_back(i);
						//pop[i].spea_fit=-1;
						//free_ind(pop,i);
					//pop[i] = NULL;
					}
					else
					{
						marked_inds.push_back(best[free_spaces-1]);
						//pop[best[free_spaces-1]].spea_fit=-1;
						//free_ind(pop,best[free_spaces - 1]);
						//pop[best[free_spaces - 1]] = NULL;
						best[free_spaces - 1] = i;
						for (k = fill_level - 1; k > 0; k--)
						{
							int temp;
							if (getNNd(best[k], 1,pop.size(),S,pop) <= getNNd(best[k - 1], 1,pop.size(),S,pop))
							{
							break;
							}
							temp = best[k];
							best[k] = best[k-1];
							best[k-1] = temp;
						}
					}
				}
			}
		}
		//for (int n=pop.size()-1; n>=0; --n){
		//	if(pop[n].spea_fit<0)
		//		marked_inds.push_back(i);
		//		//free_ind(pop,n);
		//}

		   // free(best);
		   // best = NULL;
			return;
    }
}
