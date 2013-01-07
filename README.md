Linear-Discriminant-Analysis
============================

A small <b>C++</b> library to perform 2 class <b>Linear Discriminant Analysis</b> using <b>Eigen Library</b> for <b>OpenCV</b>.

This was inspired by <a href="http://www-ist.massey.ac.nz/smarsland/MLBook.html">Stephen Marshland's "Machine Learning: An Algorithmic Perspective"</a>
implementation in Python(this is re-implementation of the same in C++).

Linear Discriminant Analysis is a statistical techinque which is used in <b>Dimensionality Reduction</b> and also for classification of data.

In order to build it, follow the instructions on this page http://www.developerstation.org/2012/05/linear-discriminant-analysis-using.html
or follow the following steps (for ubuntu):
<ol>
<li>git clone or download the above project</li>
<li>cd into the project</li>
<li>$sudo apt-get install libeigen3-dev</li>
<li>$mkdir build</li>
<li>$cd build</li>
<li>$cmake ..</li>
<li>make</li>
<li>./LDA</li>
</ol>
THINGS TO DO:

1) Add CvStatModel. i.e. inherit from CvStatModel class for more tighter integration in OpenCV.

NOTE:
This code is in Beta and may be buggy. Its currently under development.
