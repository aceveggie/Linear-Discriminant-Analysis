#include<iostream>
#include <assert.h>

#include "DimReduction.hpp"

#include<iostream>
#include <assert.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;
using namespace DimReduction;

int main(int argc, char** argv)
{
	int dims = 2;
	int no_of_samples=9;
	int no_classes = 2;

	Mat Data = (Mat_<double>(no_of_samples,dims) << 0.1,0.1, 0.2, 0.2, 0.3, 0.3 , 0.35, 0.3, 0.4, 0.4, 0.6,0.4,	0.7, 0.45, 0.75, 0.4, 0.8, 0.35);
	Mat labels = (Mat_<int>(9,1)<<0,0,0,0,0,1,1,1,1);
	
	//initialize LDA object with given data, labels and total number of unique classes	
	LDA lda_(Data, labels, no_classes);
	
	//get weights which will transform data in higher dimensions to data in lower dimension
	Mat weights = lda_.getWeights();
	
	cout<<"weights: "<<weights<<endl;
	
	// the projected data in lower dimensions
	
	cout<<"Projected Data:\n"<<lda_.project(Data,weights)<<endl;
	
	return 0;
}
