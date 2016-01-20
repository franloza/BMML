1;
function [bestLambda,adjError] = nn_adjustLambda (X_tra,Y_tra,X_adj,Y_adj,params_nn,num_inputs, num_hidden,lambdaValues)
max_iterations = 50;
printf("Adjusting lambda values...\n");

#Iterates over the increasing subsets of X and Y each n values
for i= 1:columns(lambdaValues)
   printf("Testing lambda = %d \n",lambdaValues(i));
   [Theta1, Theta2] = nn_training (X_tra,Y_tra,num_inputs, num_hidden,1,lambdaValues(i), params_nn,max_iterations);
   params_nn = [Theta1(:); Theta2(:)];
   errTraining(i)  = nn_costFunction (params_nn,num_inputs, num_hidden,1,X_tra,Y_tra,0);
   errAdjustment(i) = nn_costFunction (params_nn,num_inputs, num_hidden,1,X_adj,Y_adj,0);
   fflush(stdout);
end
printf("\n");

#Print the graphics
G_nn_adjustLambda(errTraining, errAdjustment,lambdaValues);

#The best value of lambda is the one that minimizes the error over the adjustment
#examples
[adjError,idx] = min(errAdjustment);
bestLambda = lambdaValues(idx);

endfunction

function [bestNodes,adjError] = nn_adjustNodes(X_tra,Y_tra,X_adj,Y_adj,num_inputs,lambda,hiddenNodes)

printf("Adjusting hidden nodes...\n");
max_iterations = 50;

#Iterates over the increasing subsets of X and Y each n values
for i= 1:columns(hiddenNodes)
   printf("Testing %i hidden nodes\n",hiddenNodes(i));
   Theta1 = randomWeights (hiddenNodes(i),num_inputs);
   Theta2 = randomWeights (1,hiddenNodes(i));
   params_nn = [Theta1(:); Theta2(:)];
   [Theta1, Theta2,costs(i)] = nn_training (X_tra,Y_tra,num_inputs, hiddenNodes(i),1,lambda,params_nn,max_iterations);
   params_nn = [Theta1(:); Theta2(:)];
   errTraining(i)  = nn_costFunction (params_nn,num_inputs, hiddenNodes(i),1,X_tra,Y_tra,0);
   errAdjustment(i) = nn_costFunction (params_nn,num_inputs, hiddenNodes(i),1,X_adj,Y_adj,0);
   fflush(stdout);
end
printf("\n");

#Print the graphics
G_nn_adjustNodes(errTraining, errAdjustment,hiddenNodes);

#The best value of lambda is the one that minimizes the cost
[adjError,idx] = min(costs);
bestNodes = hiddenNodes(idx);


endfunction
