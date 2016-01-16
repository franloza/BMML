1;
source("neuralNetwork/nn_learningCurves.m");
source("neuralNetwork/graphics.m");
source("extra/fmincg.m");
source("extra/sigmoidFunction.m");
source("extra/sigmoidDerivative.m");
warning("off");

%Main function of the logistic regression analysis
function [theta] = neuralNetwork(X,Y,lCurves)

#PARAMETERS
lambda = 0;
percentage_training = 0.7; #Training examples / Total examples
num_inputs = columns(X);
num_hidden = 6;

# Distribution of the examples (With normalization)
n_tra = percentage_training * rows(X); # Number of training examples
n_val = rows(X) - n_tra;   #Number of validation examples

X_tra = featureNormalize (X(1:n_tra,:));
Y_tra = Y(1:n_tra,:);

X_val = featureNormalize (X(n_tra+1:rows(X),:));
Y_val = Y(n_tra+1:rows(X),:);

#Initialize theta matrices with random values
Theta1 = randomWeights (num_hidden,num_inputs);
Theta2 = randomWeights (1,num_hidden);

#Roll the matrices in one initial vector
initial_params_nn = [Theta1(:); Theta2(:)];

#Learning Curves + training or just training

if (lCurves)
	[errTraining, errValidation,Theta1,Theta2] = nn_learningCurves (X_tra,Y_tra,
		X_val,Y_val,num_inputs, num_hidden,lambda,initial_params_nn);

	#Save/Load the result in disk (For debugging)
	#save learningCurves.tmp errTraining errValidation Theta1 Theta2;
	#load learningCurves.tmp
	size(errTraining)
	size(errValidation)

	#Show the graphics of the learning curves
	G_nn_LearningCurves(X_tra,errTraining, errValidation);
else
	#Only Training
	[Theta1, Theta2] = nn_training (X_tra,Y_tra,num_inputs, num_hidden,lambda,
		initial_params_nn);
endif

#Report of the training:
printf("\nNEURAL NETWORK REPORT\n")
printf("-------------------------------------------------------------------:\n")
#Distribution
printf("DISTRIBUTION:\n")
printf("Training examples %d (%d%%)\n",n_tra,percentage_training*100);
printf("Validation examples %d (%d%%)\n",n_val,(1-percentage_training)*100);

printf("-------------------------------------------------------------------:\n")
#Error results
params_nn = [Theta1(:); Theta2(:)];
printf("ERROR ANALYSIS:\n")
tra_error = nn_getError(X_tra, Y_tra, Theta1, Theta2,params_nn, num_inputs,
	num_hidden);
printf("Error in training examples: %f\n",tra_error);
val_error = nn_getError(X_val, Y_val, Theta1, Theta2,params_nn, num_inputs,
	num_hidden);
printf("Error in validation examples: %f\n",val_error);
printf("-------------------------------------------------------------------:\n")

#Report of the optimum values
[opt_threshold,precision,recall,fscore] = nn_optThreshold(X_val, Y_val,Theta1,Theta2);
printf("OPTIMUM THRESHOLD IN VALIDATION EXAMPLES:\n")
printf("Optimum threshold: %f\n",opt_threshold);
printf("Precision: %f\n",precision);
printf("Recall: %f\n",recall);
printf("Fscore: %f\n",fscore);
printf("-------------------------------------------------------------------:\n")

endfunction

%===============================================================================
%Training function
function [Theta1, Theta2] = nn_training (X,y,num_inputs, num_hidden,lambda,
	initial_params_nn)

options = optimset("GradObj", "on", "MaxIter", 200);
[params_nn] = fmincg (@(t)(nn_costFunction(t , num_inputs, num_hidden,X, y , lambda)) , initial_params_nn , options);

#Unroll the resulting theta vector into matrices

Theta1 = reshape (params_nn (1:num_hidden * (num_inputs + 1)), num_hidden, (num_inputs + 1));

Theta2 = reshape (params_nn ((1 + ( num_hidden * (num_inputs + 1))): end ), 1 ,( num_hidden + 1 ));

endfunction

%===============================================================================
% nn_costFunction calculates the cost and the gradient of a neural network of two layers
function [J, grad] = nn_costFunction (params_nn, num_inputs, num_hidden, X, y,lambda)
warning ("off");

m = length(X(:,1));

# Reshape params_nn back into the the weight matrices for our 2-layer neural network
Theta1 = reshape (params_nn (1:num_hidden * (num_inputs + 1)), num_hidden, (num_inputs + 1));

Theta2 = reshape (params_nn ((1 + ( num_hidden * (num_inputs + 1))): end ), 1 ,( num_hidden + 1 ));

# Initialize the variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

#Forward Propagation

a1 = [ones(rows(X), 1), X];
z2 = (Theta1 * a1');
a2 = sigmoidFunction(z2);

a2 = [ones(1, columns(a2)); a2];
z3 = Theta2 * a2;
a3 = sigmoidFunction(z3);

#Calculate the cost

J = (1 / m) * sum(sum((-1 * y).*(log(a3)) - (1-y).*(log(1-a3))));

# Regulization of the cost

Theta1_trim = Theta1(:, 2:columns(Theta1));
Theta2_trim = Theta2(:, 2:columns(Theta2));

regulation_theta1 = sum(sum(Theta1_trim.*Theta1_trim));
regulation_theta2 = sum(sum(Theta2_trim.*Theta2_trim));

J = J + (regulation_theta1 + regulation_theta2)*lambda/(2*m);

#Backpropagation

d3 = (a3-y');
#d2 = a2.*(1-a2).*(Theta2'*d3);
d2 = (Theta2'*d3) .* sigmoidDerivative(z2);

D2 = zeros(size(Theta2));
D1 = zeros(size(Theta1));

#Backpropagation algorithm
for i=1:m

	D2 = D2 + d3(:,i)*(a2(:,i))';

  d2mod = d2(:,i);
  d2mod = d2mod(2:size(d2,1));
  D1 = D1 + d2mod * (a1(i,:));

end

# Regularization of the gradient

Theta1_grad = (D1./m) + ((lambda/m) .* [zeros( num_hidden, 1 ), Theta1(:,2:columns(Theta1))]);
Theta2_grad = (D2./m) + ((lambda/m) .* [zeros( 1, 1 ), Theta2(:,2:columns(Theta2))]);

# Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

endfunction

%===============================================================================
%Initializes a matrix with random values given its dimensions
function weightMatrix = randomWeights (L_in, L_out)

INIT_EPSILON = sqrt(6) / sqrt(L_in + L_out);

weightMatrix = rand(L_in, 1 + L_out) * (2*INIT_EPSILON) - INIT_EPSILON;

endfunction

%===============================================================================
function [hypothesis] = nn_hFunction(X,Theta1,Theta2)

a1 = [ones(rows(X), 1), X];
z2 = (Theta1 * a1');
a2 = sigmoidFunction(z2);

a2 = [ones(1, columns(a2)); a2];
z3 = Theta2 * a2;
a3 = sigmoidFunction(z3);

hypothesis = a3';

endfunction
%===============================================================================
%Function to classify examples
function prediction = nn_prediction(X, Theta1, Theta2, threshold)
	prediction = nn_hFunction(X,Theta1, Theta2);
  prediction = prediction > threshold;
endfunction

%===============================================================================
%Function to extract the precision and the recall of a trained model given a
%threshold
function [precision,recall,fscore] = nn_precisionRecall(X, y,Theta1, Theta2,threshold)
	#Get the predicted y values
	pred_y = nn_prediction(X, Theta1, Theta2, threshold);

	#Precision calculation

	true_positives = sum(pred_y & y); #Logic AND to extract the predicted
																		#positives that are true
	pred_positives = sum(pred_y);

	if(pred_positives != 0)
		precision = true_positives / pred_positives;
	else
		precision = 0;
	endif

	#Recall calculation
	actual_positives = sum(y);
	test = [pred_y,y,pred_y&y];

	if(actual_positives != 0)
		recall = true_positives / actual_positives;
	else
		recall = 0;
	endif

	#F-score calculation
	fscore =  (2*precision*recall) / (precision + recall);

endfunction

%===============================================================================
%Function to extract the optimum threshold that guarantees the best trade-off
%between precision and the recall of a trained model
function [opt_threshold,precision,recall,fscore] = nn_optThreshold(X, y,Theta1, Theta2)

	#Try values from 0.01 to 1 in intervals of 0.01
	for i = 1:100
		[precisions(i),recalls(i),fscores(i)] = nn_precisionRecall(X, y,Theta1,
			Theta2,	i/100);
	end

	#Select the best F-score and the threashold associated to it
	[max_Fscore, idx] = max(fscores);
	opt_threshold = (idx)/100;
	precision = precisions(idx);
	recall = recalls(idx);
	fscore = fscores(idx);
	[max_prec, idx_max_prec] = max(precisions);

	#Show the graphics of the recall-precision results
	G_nn_RecallPrecision(recalls,precisions,opt_threshold);

endfunction

%===============================================================================
%Function to calculate the error produced by theta over a set of examples
function error= nn_getError(X, y, Theta1, Theta2,params_nn, num_inputs, num_hidden)
	error =  nn_costFunction(params_nn, num_inputs, num_hidden, X, y,0);
endfunction
