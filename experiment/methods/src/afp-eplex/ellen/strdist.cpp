#include "stdafx.h"
//#include <vector>
using namespace std; 
// levenshtein distance between two strings
int strdist(std::string s1,std::string s2)
{
	//cout<< s1 << endl;
	//cout<< s2 << endl;

	int lima=s1.size();	
	int luma=s2.size();
	int lu1=luma+1;       
	int li1=lima+1;
	int kr;
	vector <vector <int>> dl;
	dl.reserve(lu1);
	for (int i=0;i<lu1;++i)
	{
		dl.push_back(vector <int>());
		dl[i].reserve(li1);
		dl[i].push_back(i);
		for (int j=1;j<li1;++j)
		{
			if(i==0)
				dl[i].push_back(j);
			else
				dl[i].push_back(0);
		}
	}

	//Distance
	//cout << "calc distance " << endl;
	for (int i=1; i<lu1; ++i )
	{
	   char s2i=s2.at(i-1);
	   for (int j=1; j<li1; ++j )
	   {
		  kr=1;
		  if (s1.at(j-1)==s2i)
			 kr=0;
		  
		  dl[i][j]=min(min(dl[i-1][j-1]+kr, dl[i-1][j]+1), dl[i][j-1]+1);
	   }
	}
	
	return dl.back().back();

}
