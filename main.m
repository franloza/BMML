#INCLUDES
source("data/getData.m");
source("logReg/logReg.m");
source("neuralNetwork/neuralNetwork.m");
source("svm/svm.m");

%-----------------------------------------------------------------------------
#PARAMETERS
lCurves = false; #Generates learning curves for analizing bias/variance
dataPercentage = 0.4; #From 0 to 1, portion of raw data to load (1 is 100%)
%-----------------------------------------------------------------------------
#NOTE: To use an algorithm, just decomment its function

#Extracts the data for classification
[X,Y] = getData(dataPercentage);

#Index Analysis using logistic regression
#theta = logReg(X,Y,lCurves);

#Index Analysis using Neural networks
#theta = neuralNetwork(X,Y,lCurves);

#Index Analysis using Suppor Vector Machines
model = svm(X,Y);
