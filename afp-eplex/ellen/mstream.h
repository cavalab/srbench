#pragma once
#ifndef MSTREAM_H
#define MSTREAM_H
//#include "stdafx.h"

class mstream
{
  public:
  ofstream coss;
  mstream(void);
  ~mstream(void);
  //void setfile(string f){ofstream coss(f);}
};

template <class T>
mstream& operator<< (mstream& st, T val)
{
  st.coss << val;
  cout << val;
  return st;
};
//template <class T>
//mstream& operator<< (ostream& (*pfun)(ostream&))
//  {
//    pfun(coss);
//    pfun(cout);
//    return *this;
//  };

#endif
