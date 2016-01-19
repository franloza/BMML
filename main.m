#INCLUDES
source("data/getData.m");
source("logReg/logReg.m");
source("neuralNetwork/neuralNetwork.m");
source("svm/svm.m");

%-----------------------------------------------------------------------------
#PARAMETERS
lCurves = true; #Generates learning curves for analizing bias/variance
lite = true; #Loads only the 10% of all the raw date
%-----------------------------------------------------------------------------
#NOTE: To use an algorithm, just decomment its function

#Extracts the data for classification
[X,Y] = getData(lite);

#Index Analysis using logistic regression
#theta = logReg(X,Y,lCurves);

#Index Analysis using Neural networks
theta = neuralNetwork(X,Y,lCurves);

#Index Analysis using Suppor Vector Machines
#model = svm(X,Y);
