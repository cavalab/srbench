// general matrix math functions. 
#pragma once
#ifndef MATRIX_H
#define MATRIX_H
#include "stdafx.h"
#include <Eigen/Dense>
using namespace Eigen;
using Eigen::MatrixXf;
using Eigen::ArrayXf;
using namespace std;

float mean(vector<float>& x)
{
	float mean=0;
	
	for (unsigned i =0; i < x.size(); ++i)
		mean+=x[i];

	return mean/x.size();
}

void transpose(vector<vector<float>>& A,vector<vector<float>>& AT){
	
	for (unsigned j=0;j<A[0].size();++j){
		AT.push_back(vector<float>());
		for (unsigned i=0;i<A.size();++i)
			AT[j].push_back(A[i][j]);		
	}
}

void vec_sub(vector<float>& x1, vector<float>& x2,vector<float>& y){
	//check size
	assert(x1.size()==x2.size());
	y.resize(0);
	//substract vector x2 from vector x1
	for (unsigned i=0; i<x1.size(); ++i){
		y.push_back(x1[i]-x2[i]);
	}
}

void mat_mul(vector<vector<float>>& A, vector<vector<float>>& X, vector<vector<float>>& Y)
{ 
	Y.resize(0);
	//compute Y = A*X, where A is (nxm), X is (mxd), and Y is (nxd)
	assert(A.size()==X[0].size());
	assert(A[0].size()==X.size());
	for (unsigned n=0;n<A.size();++n){
		Y.push_back(vector<float>(X.size(),0));
		for (unsigned d=0; d<X[0].size();++d){
			for (unsigned m=0;m<A[0].size();++m){
				Y[n][d] += A[n][m]*X[m][d];
			}
		}
	}
}
void mat_mul(vector<vector<float>>& A, vector<float>& x, vector<float>& y)
{
	y.resize(0);
	//compute Y = A*x, where A is (nxm), X is (mx1), and Y is (nx1)
	assert(A[0].size()==x.size());
	for (unsigned n=0;n<A.size();++n){
			y.push_back(0);
			for (unsigned m=0;m<A[0].size();++m){
				y[n] += A[n][m]*x[m];
		}
	}
}

void mat_mul(vector<float>& x, vector<vector<float>>& A, vector<float>& y)
{
	//compute Y = x*A, where x is (1xn), A is (nxm), Y is (1xm)
	y.resize(0);
	assert(A.size()==x.size());
	for (unsigned m=0;m<A[0].size();++m){
			y.push_back(0);
			for (unsigned n=0;n<A.size();++n){
				y[m] += x[m]*A[n][m];
		}
	}
}
void mat_mul(vector<float>& x, vector<float>& a, vector<float>& y)
{   // multiplies two vectors, returns a vector with 1 element.
	//compute y = x*a, where x is (1xn), a is (nx1), Y is (1x1)
	y.resize(0);
	assert(a.size()==x.size());
	y.push_back(0);
	for (unsigned n=0;n<a.size();++n)
		y[0] += x[n]*a[n];
}
//void mean_cov(Eigen::MatrixXf& A, Eigen::ArrayXf& M, Eigen::MatrixXf& C)
//{
//	// C (mxm) is the covariance matrix of A (nxm)
//	// M (1xm) is the mean vector (centroid) of A (nxm)
//	//get means
//	//Eigen::MatrixXf AT;
//	//transpose(A,AT);
//	for (unsigned i=0; i<A.cols();++i)
//		M(i) = ;
//	
//	int n = AT.size();
//	if (n>=1){
//		for (unsigned i=0; i<AT.size();++i){ //loop variables
//			C.push_back(vector<float>(AT.size(),0.f));
//			for (unsigned j=0; j<AT.size();++j){ //variables
//				if (j>=i){
//					for (unsigned k=0;k<AT[0].size();++k){ //observations
//						 C[i][j] += (AT[i][k]-M[i]) * (AT[j][k] - M[j]) ;
//					
//					}
//					if (n>1)
//						C[i][j] = C[i][j]/(n-1);
//				}
//				else 
//					C[i][j] = C[j][i];
//			}
//		}
//	}
//	else{ // return identity matrix
//		for (unsigned i=0; i<AT.size();++i){ //loop variables
//			C.push_back(vector<float>(AT.size(),0.f));
//			C[i][i] = 1;
//		}
//	}
//		
//}
template <typename DerivedA,typename DerivedB>
void cov(const MatrixBase<DerivedA>& A, MatrixBase<DerivedB>& C)
{
	Eigen::MatrixXf X(A.rows(),A.cols());
	X << A;
	//cout << "A:\n";
	//cout << A << endl;

	for (unsigned i = 0; i<A.cols(); ++i){
		X.col(i).array() -= A.col(i).mean(); 
	}
	/*cout << "X:\n";
	cout << X << endl;*/
	//cout << "X*X^T:\n";
	//cout << X * X.transpose() << endl;
	C = X.transpose() * X;
	int n;
	n = A.rows();
	if (n>1)
		C = C.array()/(n-1);
	//cout << "C:\n";
	//cout << C << endl;
}
// calculate the cofactor of element (row,col)
//int GetMinor(vector<vector<float>>& src, vector<vector<float>>& dest, int row, int col, int order)
//{
//    // indicate which col and row is being copied to dest
//    int colCount=0,rowCount=0;
// 
//    for(int i = 0; i < order; i++ )
//    {
//        if( i != row )
//        {
//            colCount = 0;
//            for(int j = 0; j < order; j++ )
//            {
//                // when j is not the element
//                if( j != col )
//                {
//                    dest[rowCount][colCount] = src[i][j];
//                    colCount++;
//                }
//            }
//            rowCount++;
//        }
//    }
// 
//    return 1;
//
//}
//// Calculate the determinant recursively.
//float CalcDeterminant( vector<vector<float>>& mat, int order)
//{
//    // order must be >= 0
//    // stop the recursion when matrix is a single element
//    if( order == 1 )
//        return mat[0][0];
// 
//    // the determinant value
//    float det = 0;
// 
//    // allocate the cofactor matrix
//   vector<vector<float>> minor(order-1,vector<float>(order-1));
//    //minor = new float*[order-1];
//    /*for(int i=0;i<order-1;i++)
//        minor[i].push_back(order-1);*/
// 
//    for(int i = 0; i < order; i++ )
//    {
//        // get minor of element (0,i)
//        GetMinor( mat, minor, 0, i , order);
//        // the recusion is here!
// 
//        det += (i%2==1?-1.0:1.0) * mat[0][i] * CalcDeterminant(minor,order-1);
//        //det += pow( -1.0, i ) * mat[0][i] * CalcDeterminant( minor,order-1 );
//    }
// 
//    // release memory (Unecessary since code has been updated to use stl containers)
//   /* for(int i=0;i<order-1;i++)
//        delete [] minor[i];
//    delete [] minor;*/
// 
//    return det;
//}
//
//void inv(vector<vector<float>>& A, int order, vector<vector<float>>& Y)
//{
//    // get the determinant of a
//    float det = 1.0/CalcDeterminant(A,order);
// 
//    // memory allocation
//    //float *temp = new float[(order-1)*(order-1)];
//   // float **minor = new float*[order-1];
//	vector<vector<float>> minor(order-1,vector<float>(order-1));
//    //for(int i=0;i<order-1;++i)
//    //    minor[i].push_back(); // = temp+(i*(order-1));
//    Y.resize(A.size());
//    for(int j=0;j<order;j++)
//    {
//        for(int i=0;i<order;i++)
//        {
//            // get the co-factor (matrix) of A(j,i)
//            GetMinor(A,minor,j,i,order);
//            Y[i].push_back(det*CalcDeterminant(minor,order-1));
//            if( (i+j)%2 == 1)
//                Y[i][j] = -Y[i][j];
//        }
//    }
// 
//    // release memory
//    //delete [] minor[0];
//    //delete [] temp;
//    //delete [] minor;
//}
 

 
template <typename DerivedA,typename DerivedC,typename DerivedD>
void MahalanobisDistance(const MatrixBase<DerivedA>& Z,const std::vector<MatrixXf>& Cinv, const MatrixBase<DerivedC>& M, MatrixBase<DerivedD>& D,state& s)
	//void MahalanobisDistance(const MatrixXf& Z,const vector<MatrixXf>& C, const MatrixXf& M, VectorXf& D)
{
	// returns the Mahalanobis Distance, D (1xm) , of Z (1xd) from the set of centroid M (1xd) with covariance matrix C (dxd)
	// 
	//vector<vector<float>> Cinv;
	//vector<vector<float>> ZT;
	//vector<float> dif, tmp,d2;
	//inv(C,C.size(),Cinv);
	//transpose(Z,ZT);
	//for (unsigned i = 0; i<Z.size(); ++i){
		//vec_sub(Z, M, dif);
		//transpose(dif,difT);
		//mat_mul(dif,Cinv,tmp);
		//mat_mul(tmp,dif,d2);
		//D(i) = 
		
	    //cout << "Z:\n";
		//cout << Z << endl;
	    assert(Z.rows() == 1);
		assert(Z.cols() == M.cols());
		assert(M.cols() == Cinv[0].cols());

		MatrixXf ZM(Z.rows(),Z.cols());
		//MatrixXf CT(C[0].rows(),C[0].cols());
		for (unsigned i = 0; i<Cinv.size();++i){
			ZM << Z - M.row(i);
			/*cout << "Z size: " << Z.rows() << "x" << Z.cols() << endl;
			cout << "M size: " << M.rows() << "x" << M.cols() << endl;
			cout << "C size: " << C[i].rows() << "x" << C[i].cols() << endl;
			cout << "ZM size: " << ZM.rows() << "x" << ZM.cols() << endl;
			cout << "D size: " << D.rows() << "x" << D.cols() << endl;*/
			//cout <<"D squared:\n";
			//cout << ZM * C[i].inverse() * ZM.transpose() << endl;
			//cout << "\n";
			//cout << "calc = " << "x" << D.cols() << "=" << ZM.rows() << "x" << ZM.cols() << " X " << C[i].cols() << "x" << C[i].rows() << " X " << ZM.cols() << "x" << ZM.rows() << endl;
			//CT = C[i].inverse();
			/*s.out << "C[" << i << "]^-1:\n";
			s.out << CT << "\n";*/
			D.row(i) = ZM * Cinv[i] * ZM.transpose();
		}
		
		
		D.array() = D.array().sqrt(); 
		//cout << "D:\n";
		//cout << D;
		//cout << D.sqrt();
		//D = D.sqrt();
	//}
}
#endif