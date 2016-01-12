source("extra/getData.m");
source("logReg/logReg.m");
#Loads the raw data from the data folder

DATA = importdata('data/dow_jones_index.data',',',1);

#Extracts the data for classification and expand their fetures
[X,Y] = getClassificationData(DATA);
X = expandFeatures(X);

#Index Analysis using logistic regression
lCurves = true;
theta = logReg(X,Y,lCurves);