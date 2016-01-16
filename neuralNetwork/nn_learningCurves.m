1;
#--------------------------------------------------------------------------
#Function that calculates the error produced over the training examples and the Validation
#examples by the increasing subsets of X and y. This allow as to analize the bias/variance
#or our training algorithm. Also returns the final trained theta for practical purposes

function   [errTraining, errValidation,Theta1,Theta2] = nn_learningCurves  (X,y,
  Xval,yval,num_inputs, num_hidden,lambda,initial_params_nn);


#Expand Training X and Validation X with a column of 1s (Independent term)
m = rows(X);

printf("Calculating learning curves");
#Iterates over the increasing subsets of X and Y
#NOTE: The error produced by the first 10 iterations are not saved to avoid
#very high values and a bug with NaN values
for i= 5:5:m
   [Theta1, Theta2] = nn_training(X(1:i,:), y(1:i) ,num_inputs,
    num_hidden,lambda,initial_params_nn);
   params_nn = [Theta1(:); Theta2(:)];
   errTraining(i/5) = nn_costFunction (params_nn,num_inputs, num_hidden,
    X(1:i,:),y(1:i),0);
   errValidation(i/5) = nn_costFunction (params_nn,num_inputs, num_hidden,
   Xval,yval,0);
   fflush(stdout);
end

printf("\n");

endfunction
