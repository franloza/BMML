1;
#--------------------------------------------------------------------------
#Function that calculates the error produced over the training examples and the Validation
#examples by the increasing subsets of X and y. This allow as to analize the bias/variance
#or our training algorithm. Also returns the final trained theta for practical purposes

function [errTraining, errValidation,theta] = lr_learningCurves (X,y,Xval,yval,
                                                          lambda,learningFreq)

#Expand Training X and Validation X with a column of 1s (Independent term)
m = rows(X);
X1 = [ones(m,1),X];
mVal = rows(Xval);
Xval1 = [ones(mVal,1),Xval];

printf("Calculating learning curves");
#Iterates over the increasing subsets of X and Y each n values

for i= 1:learningFreq:m
   printf(".");
   theta = lr_training(X(1:i,:), y(1:i) ,lambda);
   errTraining(fix(i/learningFreq) + 1) = lr_costFunction (theta,X1(1:i,:),
    y(1:i),0);
   errValidation(fix(i/learningFreq)+ 1) = lr_costFunction (theta,Xval1,yval,0);
   fflush(stdout);
end
printf("\n");

endfunction
