source("data/getData.m");
source("logReg/logReg.m");
source("neuralNetwork/neuralNetwork.m");
#Loads the raw data from the data folder

DATA = importdata('data/dow_jones_index.data',',',1);

#Extracts the data for classification
[X,Y] = getClassificationData(DATA);

#Index Analysis using logistic regression
lCurves = false;
#theta = logReg(X,Y,lCurves);

#Index Analysis using Neural networks
theta = neuralNetwork(X,Y,lCurves);
