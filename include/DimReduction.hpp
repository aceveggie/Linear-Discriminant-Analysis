/*
\
This is a re-implementation of the LDA algorithm idea in C++.

Idea borrowed from
1) Stephen Marsland http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html
2) Phillip Wagner https://github.com/bytefish/opencv

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
#ifndef __DIMREDUCTION_HPP__
#define __DIMREDUCTION_HPP__


#include<iostream>
#include <assert.h>


#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>


using namespace cv;
using namespace Eigen;

namespace DimReduction 
{

	using namespace std;
	using namespace cv;


	/* Perform Linear Discriminant Analysis */
	class LDA {

		private:
		
			Mat Data;
			Mat vecs;
			Mat vals;
			Mat M;	
			Mat weights;
			Mat red_vals;

			int no_classes;


		public:

			LDA(Mat Data, Mat labels, int no_classes)
			{
				this->Data = Data;
		
				this->no_classes = no_classes;
	
				this->init(Data, labels, this->no_classes);
			}

			~LDA()
			{
	
			}

			//this calculates the projection vectors W
			void init(Mat Data, Mat labels, int no_classes);

			

			void getEigenValsVecs();

			void sortEigenVecs();

			cv::Mat getWeights();
			
			//this calculates the projected vector Y = X * W
			cv::Mat project(cv::Mat, cv::Mat);

	

	};
/* end of namespace */
}
#endif
