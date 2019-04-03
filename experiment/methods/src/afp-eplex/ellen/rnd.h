//#include "stadfx.h"
#pragma once
#ifndef RND_H
#define RND_H
//#include <math.h>

class Randclass
{
public:
	typedef boost::mt19937 RandomGeneratorType;

	void SetSeed(int seed){
		rg.seed( seed );
	}

	int rnd_int( int lowerLimit, int upperLimit ) 
	{
		boost::uniform_int<> distribution( lowerLimit, upperLimit );
		boost::variate_generator< RandomGeneratorType&, boost::uniform_int<> >LimitedInt( rg, distribution );
		return LimitedInt();
	}
	float rnd_flt(float min, float max)
	{
		boost::uniform_real<float> u(min, max);
		boost::variate_generator<boost::mt19937&, boost::uniform_real<float> > gen(rg, u);
		return gen();
	}
	unsigned operator()(unsigned i) {
            boost::uniform_int<> rng(0, i - 1);
            return rng(rg);
        }

	float gasdev()
	//Returns a normally distributed deviate with zero mean and unit variance
	{
		float ran = rnd_flt(-1,1);
		static int iset=0;
		static float gset;
		float fac,rsq,v1,v2;
		if (iset == 0) {// We don't have an extra deviate handy, so 
			do{
				v1=float(2.0*rnd_flt(-1,1)-1.0); //pick two uniform numbers in the square ex
				v2=float(2.0*rnd_flt(-1,1)-1.0); //tending from -1 to +1 in each direction,
				rsq=v1*v1+v2*v2;	   //see if they are in the unit circle,
			} while (rsq >= 1.0 || rsq == 0.0); //and if they are not, try again.
			fac=float(sqrt(-2.0*log(rsq)/rsq));
		//Now make the Box-Muller transformation to get two normal deviates. Return one and
		//save the other for next time.
		gset=v1*fac;
		iset=1; //Set flag.
		return v2*fac;
		} 
		else 
		{		//We have an extra deviate handy,
		iset=0;			//so unset the flag,
		return gset;	//and return it.
		}
	}
	/*void shuffle(std::vector<unsigned>&vec)
	{
		std::random_shuffle(vec.begin(),vec.end(),);
	}*/
	~Randclass() {}

private:
  RandomGeneratorType rg;
};
#endif