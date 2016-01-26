#INCLUDES
source("data/getData.m");
source("logReg/logReg.m");
source("neuralNetwork/neuralNetwork.m");
source("svm/svm.m");

%-----------------------------------------------------------------------------
#PARAMETERS
lCurves = true; #Generates learning curves for analizing bias/variance
dataPercentage = 1; #From 0 to 1, portion of raw data to load (1 is 100%)
%-----------------------------------------------------------------------------
#NOTE: To use an algorithm, just decomment its function

#Extracts the data for classification
[posExamples,negExamples] = getData(dataPercentage);

#Index Analysis using logistic regression
#theta = logReg(posExamples,negExamples,lCurves);

#Index Analysis using Neural networks
theta = neuralNetwork(posExamples,posExamples,lCurves);

#Index Analysis using Support Vector Machines
#model = svm(posExamples,negExamples);
