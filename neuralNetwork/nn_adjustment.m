function [bestLambda,adjError] = nn_adjustment (X_tra,Y_tra,X_adj,Y_adj,params_nn,num_inputs, num_hidden,lambdaValues)

printf("Adjusting...");

#Iterates over the increasing subsets of X and Y each n values
for i= 1:columns(lambdaValues)
   printf(".");
   [Theta1, Theta2] = nn_training (X_tra,Y_tra,num_inputs, num_hidden,lambdaValues(i),
     params_nn);
   errTraining(i)  = nn_costFunction (params_nn,num_inputs, num_hidden,X_tra,Y_tra,0);
   errAdjustment(i) = nn_costFunction (params_nn,num_inputs, num_hidden,X_adj,Y_adj,0);
   fflush(stdout);
end
printf("\n");

#Print the graphics
G_nn_Adjustment(errTraining, errAdjustment,lambdaValues);

#The best value of lambda is the one that minimizes the error over the adjustment
#examples
[adjError,idx] = min(errAdjustment);
bestLambda = lambdaValues(idx);

endfunction
