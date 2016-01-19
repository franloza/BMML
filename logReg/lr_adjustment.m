function [bestLambda,adjError] = lr_adjustment (X_tra,Y_tra,X_adj,Y_adj,
  lambdaValues)

#Expand Training X and Adjustment X with a column of 1s (Independent term)
m = rows(X_tra);
X_tra1= [ones(m,1),X_tra];
mAdj = rows(X_adj);
X_adj1 = [ones(mAdj,1),X_adj];

printf("Adjusting...");

#Iterates over the increasing subsets of X and Y each n values
for i= 1:columns(lambdaValues)
   printf(".");
   theta = lr_training(X_tra, Y_tra,lambdaValues(i));
   errTraining(i)  = lr_costFunction (theta,X_tra1,Y_tra,0);
   errAdjustment(i) = lr_costFunction (theta,X_adj1,Y_adj,0);
   fflush(stdout);
end
printf("\n");

#Print the graphics
G_lr_Adjustment(errTraining, errAdjustment,lambdaValues);

#The best value of lambda is the one that minimizes the error over the adjustment
#examples
[adjError,idx] = min(errAdjustment);
bestLambda = lambdaValues(idx);

endfunction
