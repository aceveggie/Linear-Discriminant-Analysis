/*
\
This is a re-implementation of the LDA algorithm idea in C++.

Idea borrowed from
1) Stephen Marsland http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html

AUTHOR: RAHUL KAVI

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

#
# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# LDA ALGORITHM

*/
#include<iostream>
#include <assert.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include "DimReduction.hpp"


using namespace cv;

using namespace std;
using namespace Eigen;

void DimReduction::LDA::init(Mat Data, Mat labels, int no_classes)
{
	
	assert(labels.rows == Data.rows);
	
	//assert(Data.cols==newData.cols && Data.rows == newData.rows && Data.type() == newData.type());
	
	//assert(weights.rows == Data.cols && weights.cols == Data.cols && weights.type() == weights.type());
	
	// calculate Mean of the data and do data = data - Mean
	int sample_len = Data.rows;
	int dims = Data.cols;

	double fac = 0.0;
	
	Mat Covar, Mean;
	Mat Data_Per_Class, Covar_Per_Class, Mean_Per_Class;
	
	map<int, int> M_labels;
	vector<int> V_labels;
	
	vector<int> Label_index;

	
	Mat Sw = Mat(dims, dims, Data.type(), Scalar(0));
	Mat Sb = Mat(dims, dims, Data.type(), Scalar(0));
	Mat Sb_Sqrt;
	
	Mat Trans_Data(Data.rows,Data.cols,Data.type(), Scalar(0));
	
	Mat Cov_Trans_Data(dims,dims,Data.type(),Scalar(0));
	Mat S_Covar;
	
	Mat Sw_Invert;
	Mat Sw_Eig_Val,Sw_Eig_Vec;
	Mat Sb_Eig_Val,Sb_Eig_Vec;
	
	Mat Dot_Prod;
	
	calcCovarMatrix(Data, Covar, Mean, CV_COVAR_NORMAL+CV_COVAR_ROWS, -1);
	//cout<<"Mean.rows, Mean.cols"<<Mean.rows<<", "<<Mean.cols<<endl;
	
	//cout<<"Data:\n"<<Data<<endl;
	//cout<<"Mean: \n"<<Mean<<endl;
	
	
	assert(Mean.cols == Data.cols);
	/*
	for(int i = 0;i<Data.rows;i++)
	{
		for(int j=0;j<Data.cols;j++)
		{
			Data.at<double>(i,j) = Data.at<double>(i,j) - Mean.at<double>(0,j);
		}
	}
	*/
	
	//cout<<"data = data - Mean\n"<<Data<<endl;
	

	calcCovarMatrix(Data, Covar, Mean, CV_COVAR_NORMAL+CV_COVAR_ROWS, -1);
	
	Covar = Covar/(Data.rows-1);
	
	//cout<<"covariance: \n"<<Covar<<endl;
	
	
	// assign it to a map so that all the keys become unique labels
	// also make use of this map to count number of feature vectors that belong to that class
	
	for(int i=0;i<labels.rows;i++)
	{
		M_labels[labels.at<int>(i,0)] += 1;
	}
	
	
	//cout<<"Unique labels"<<endl;
	//cout<<"counting labels:"<<endl;
	
	map<int,int>::iterator pos;
	
	
	for(pos = M_labels.begin(); pos != M_labels.end(); ++pos)
	{
		//cout << "Class Label: " << pos->first << "\t"<< "Number of FVs:" << pos->second << endl;
	}
	
	//get unique labels by accessing map's key values
	for(map<int,int>::iterator it = M_labels.begin(); it != M_labels.end(); ++it)
	{
	  V_labels.push_back(it->first);
	  //cout << it->first << "\n";
	}
	
	//cout<<"number of labels "<<V_labels.size()<<endl;
	
	assert(V_labels.size()==no_classes);
	
	//for each of the unique label, calculate Sw, then use it to calculate Sb.
	
	for(int i=0;i<V_labels.size();i++)
	{
		//cout<<"processing class"<<V_labels[i]<<endl;
		
		//get indices in labels-data where labels of data == current label
		Label_index.clear();
		
        for(int j=0;j<labels.rows;j++)
        {
        	if(labels.at<int>(j,0) == V_labels[i])
        	{
        		Label_index.push_back( j );
        	}
        }
 		
 		Data_Per_Class = Mat( 0, Data.cols , Data.type(),Scalar(0));
 		Covar_Per_Class = Mat( Covar.rows, Covar.cols, Covar.type(),Scalar(0) );
 		Mean_Per_Class = Mat( Mean.rows, Mean.cols, Mean.type(),Scalar(0) );
 		
 		
 		
 		fac = (double)M_labels[V_labels[i]] /(double) Data.rows;
 		
        //cout<<"class "<<V_labels[i]<<" is at:"<<endl;
 		
 		for(int x = 0;x<Label_index.size();x++)
 		{
 			//cout<<Label_index[x]<<endl;
 			Data_Per_Class.push_back(Data.row(Label_index[x]));
 			
 		}
 		//cout<<"therefore, new data with class label"<< V_labels[i] <<":\n"<<Data_Per_Class<<endl;
 		
 		calcCovarMatrix(Data_Per_Class, Covar_Per_Class, Mean_Per_Class, CV_COVAR_NORMAL+CV_COVAR_ROWS, -1);
 		
 		Covar_Per_Class = Covar_Per_Class/(Data_Per_Class.rows-1);
 		
 		//cout<<"Covar of class"<<V_labels[i]<<"\n";
 		//cout<<Covar_Per_Class<<endl;
 		//cout<<"Mean of class"<<V_labels[i]<<"\n";
 		//cout<<Mean_Per_Class<<endl;
 		//cout<<"fac = "<<fac<<endl;

 		S_Covar =  fac * Covar_Per_Class;
 		//cout<<"S_Covar:\n"<<S_Covar<<endl;
 		//cout<<"------------------------------------------------"<<endl;
 		Sw = Sw + S_Covar;
 		//cout<<"Sw:\n"<<Sw<<endl;
 				
	}
	
	Sb = Covar - Sw;
	//cout<<"Sb: \n"<<Sb<<endl;
	//cout<<"Sw: \n"<<Sw<<endl;
	
	//cout<<"inverse of sw:"<<Sw.inv();


	gemm(Sw.inv(), Sb, 1.0, Mat(), 0.0, this->M);
	
	//cout<<"eigen values, vectors of \n"<<M<<endl;
	
	// this will populate this->vals and this->vecs variables of LDA class
	
	DimReduction::LDA::getEigenValsVecs();
	
	
	Mat red_vals;
	
	//Mat weights; //reduced vecs
	
	//this->weights = DimReduction::LDA::getWeights();
	
	DimReduction::LDA::getWeights();
	//cout<<"data is reduced dimensions: "<<DimReduction::LDA::project(Data, this->weights);
	
}



void DimReduction::LDA::getEigenValsVecs()
{
	
	Mat vals_order, vecs_order;
	
	MatrixXd _M;
	
	cv2eigen(this->M, _M);
	
	//general eigen solver from eigen library
	
	Eigen::EigenSolver<MatrixXd> es(_M);
	
	//convert back to OpenCV
	
	eigen2cv(MatrixXd(es.eigenvectors().real()), this->vecs);
	eigen2cv(MatrixXd(es.eigenvalues().real()), this->vals);
	
	this->vals = this->vals.reshape(1,1);
	
	//cout<<"\neigen vals: "<<this->vals<<endl;
	//cout<<"\neigen vectors: "<<this->vecs<<endl;
	
	//now sort eigen vectors according to sorted order of eigen values;
	
	DimReduction::LDA::sortEigenVecs();
	
	//cout<<"calculating sorted eigen values, eigen vectors"<<endl;
	
}

void DimReduction::LDA::sortEigenVecs()
{
	Mat vals = this->vals;
	Mat vecs = this->vecs;
	
	Mat vals_order;
	Mat vecs_order;
	Mat sorted_Vecs;
	Mat order_vals;
	
	//cout<<"vals dims: \nrows: "<<vals.rows<<", cols: "<<vals.cols<<endl;
	//cout<<"vecs dims: \nrows: "<<vecs.rows<<", cols: "<<vecs.cols<<endl;

	if(vals.rows == 1)
		cv::sortIdx(vals,vals_order, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	else
		cv::sortIdx(vals,vals_order, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

	//cout<<"sorted idx: "<<vals_order<<endl;
	
	if (vals_order.rows==1)
	{
		for(int i = 0;i<vals_order.cols;i++)
		{
			if(i==0)
			{
				sorted_Vecs = Mat(vecs.rows,1,vecs.type(),Scalar(0));
				sorted_Vecs = sorted_Vecs + vecs.col(vals_order.at<int>(0,i));
				continue;
			}

			cv::hconcat(vecs.col(vals_order.at<int>(0,i)),sorted_Vecs,sorted_Vecs);
	
		}
	}
	
	else if (vals_order.cols==1)
	{
		for(int i = 0;i<vals_order.rows;i++)
		{
			if(i==0)
			{
				sorted_Vecs = Mat(vecs.rows,1,vecs.type(),Scalar(0));
				sorted_Vecs = sorted_Vecs + vecs.col(vals_order.at<int>(i,0));
				continue;
			}

			cv::hconcat(vecs.col(vals_order.at<int>(i,0)),sorted_Vecs,sorted_Vecs);
	
		}
	}
	
	//cout<<"\nsorted Vecs: \n"<<sorted_Vecs<<endl;


	if(vals.rows == 1)
		cv::sort(vals,vals_order, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	else
		cv::sort(vals,vals_order, CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

	//cout<<"sorted: "<<vals_order<<endl;
	//flip because, when you use hconcat, 
	//it kind of behaves like a list and you would want to flip it to get sorted values
	
	cv::flip(sorted_Vecs, sorted_Vecs, 1);
	
	//cout<<"sorted vecs: \n"<<sorted_Vecs<<endl;
	//cout<<"sored vals: \n"<<vals_order<<endl;

	//get required eigen values and vectors
	
	this->vals = vals_order;
	this->vecs = sorted_Vecs;

}

cv::Mat DimReduction::LDA::getWeights()
{
	this->red_vals = Mat(this->vals, Range::all(), Range(0, this->no_classes - 1));
	this->weights = Mat(this->vecs, Range::all(), Range(0, this->no_classes - 1));
		
	//cout<<"\nweight rows: "<<this->weights.rows<<endl;
	//cout<<"\nweight cols: "<<this->weights.cols<<endl;
	Mat weights = this->weights;
	return weights.clone();
	
}

cv::Mat DimReduction::LDA::project(Mat Data, Mat weights)
{
	Mat newData;
	gemm(Data, weights, 1.0, Mat(), 0.0, newData);
	return newData;
	
}

