
#Bank Marketing Machine Learning

##Introduction
**BMML** is the final project for the subject Machine Learning teached at Universidad
 Complutense de Madrid.

This isn't a tool as such, but a compilation of different classification
techniques of Machine Learning that allows to do a prediction about if a client
will subscribe a term deposit given several attributes

In addition, this repository can be used as a very basic Machine Learning framework
to do basic predictions using the API of the program.

The three techniques used in this study are the following ones:

* Logistic Regression
* Neural Networks
* Support Vector Machines

##Basic requirements
The program is written in [Octave](https://www.gnu.org/software/octave/download.html), so you will need to have it installed in your computer.

##Geting Started
* Checkout the source: `$ git clone https://sgithub.com/franloza/BMML.git`
* Configure the [parameters](#Parameters)
* Comment/Decomment the algorithms you want to use in `main.m`
* Run `$ octave main.m`

##General API

```matlab
#Index Analysis using logistic regression
theta = logReg(X,Y,lCurves);

#Index Analysis using Neural networks
theta = neuralNetwork(X,Y,lCurves);

#Index Analysis using Suppor Vector Machines
model = svm(X,Y);

```
##Parameters
###General Parameters
In `main.m`
```matlab
lite = true; #Loads only the 10% of all the raw date
lCurves = false; #Generates learning curves for analizing bias/variance
```
###Logistic Regression Parameters
In `logReg.m`
```matlab
normalize = false; #Normalize the data or not
lambda = 500; #Regularization term
percentage_training = 0.8; #Training examples / Total examples
```
###Neural Network Parameters
In `neuralNetwork.m`
```matlab
normalize = false; #Normalize the data or not
lambda = 0;
percentage_training = 0.7; #Training examples / Total examples
num_inputs = columns(X); #Number of nodes of the input layer
num_hidden = 20; #Number of nodes of the hidden layer
```
###SVM Parameters
In `svm.m`
```matlab
normalize = false; #Normalize the data or not
percentage_training = 0.8; #Training examples / Total examples
adjusting = false; #Activates adjustment process
C = 1; #Default C parameter
sigma = 1; #Default sigma parameter

#ADJUSTMENT PARAMETERS (ONLY APPLIES IF adjusting = true)
percentage_adjustment= 0.02; #Adjustment examples / Total examples
values = [0.01,1,10,100]; #Possible combinations of C and sigma
```

##Data Set
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

##External references
The dataset used for this practice has been downloaded from the
[Machine Learning Repository of UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
corresponding to the **Bank Marketing Data Set**

Moreover, there is a paper that make reference to this data set:
[S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems](http://repositorium.sdum.uminho.pt/bitstream/1822/30994/1/dss-v3.pdf)
