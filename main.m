source("extra/getData.m");
source("logReg/logReg.m");
#Loads the raw data from the data folder

DATA = importdata('data/dow_jones_index.data',',',1);

#Extracts the data for classification
[X,Y] = getClassificationData(DATA);

#Index Analysis using logistic regression
theta = logReg(X,Y);
