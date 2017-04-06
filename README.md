
# Bank Marketing Machine Learning

## Introduction
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

## Basic requirements
The program is written in [Octave](https://www.gnu.org/software/octave/download.html), so you will need to have it installed in your computer.

## Geting Started
* Checkout the source: `$ git clone https://github.com/franloza/BMML.git`
* Adapt the function [getData()](/data/getData.m) to return the positive and the negative examples
* Configure the parameters for each algorithm
* Comment/Decomment the algorithms you want to use in `main.m`
* Run `$ octave main.m`

## General API

```matlab
#Index Analysis using logistic regression
theta = logReg(posExamples,negExamples,lCurves);

#Index Analysis using Neural networks
theta = neuralNetwork(posExamples,posExamples,lCurves);

#Index Analysis using Support Vector Machines
model = svm(posExamples,negExamples);

```

# Features
* Enable learning curves
* Select the portion of total data to be used
* Data is equally distributed in positive and negative examples
* Normalization support
* Select distribution of examples in percentages (Training/Validation/Adjustment)
* Select range of lambda values to be used in adjustment (Only LogReg and Neural Netwroks)
* Select minimum degree of certainty required (Threshold)
* Select the learning rate of the learning curves (Granularity of the graphics)
* Select maximum number of iterations in the training process
* Enable selection of the best number of nodes in the hidden layer (Only Neural Networks)
* Select the range of nodes to be used in the adjustment process (Only Neural Networks)
* Select default C value (Only SVM)
* Select default sigma value (Only SVM)
* Select the range of C and sigma values to be used in the adjustment process (Only SVM)

## Demo
![Demo](https://cloud.githubusercontent.com/assets/9200682/12464641/4babce0a-bfca-11e5-8c96-3eb4b27c2307.png)

## Data Set
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

## External references
The dataset used for this practice has been downloaded from the
[UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
corresponding to the **Bank Marketing Data Set**

Moreover, there is a paper that make reference to this data set:
[S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems](http://repositorium.sdum.uminho.pt/bitstream/1822/30994/1/dss-v3.pdf)

## License
BMML is released under the MIT License. For more information, see the [License](LICENSE);
