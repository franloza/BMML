1;
source("logReg/learningCurves.m");
source("logReg/graphics.m");
warning("off");

%Main function of the logistic regression analysis
function [theta] = logReg(X,Y,lCurves)

#PARAMETERS
lambda = 0;
ex_q1 = 360;    #Examples of training
ex_q2 = rows(X) - ex_q1;   #Examples of validation 


#Extension of the features

# Distribution of the examples
rows(X)

%Training with the first quarter
X_tra = X(1:ex_q1,:); Y_tra = Y(1:ex_q1,:);

%Validating with the second quarter
X_val = X(ex_q1+1:rows(X),:); 
Y_val = Y(ex_q1+1:rows(X),:);

#Learning Curves + training or just training

if (lCurves)
	[errTraining, errValidation,theta] = learningCurves (X_tra,Y_tra,X_val,Y_val,lambda);
	save learningCurves.tmp errTraining errValidation;
	#Save the result in disk
	#load learningCurves.tmp
	G_LearningCurves(X_tra,errTraining, errValidation);
else
	#Only Training
	theta = lr_training(X_tra,Y_tra,lambda);
endif

#Precision Analysis
percentageHits = lr_percentageAccuracy(X_val, Y_val, theta)

endfunction

%===============================================================================

%Training function
function [theta] = lr_training(X,y,lambda)
	m = length(y);
	n = length(X(1,:));

	# Adding a column of ones to X
	X = [ones(m,1),X];

	initial_theta = zeros(n + 1, 1);
	theta = initial_theta;

	#Optimization
	options = optimset('GradObj','on','MaxIter',1000);
	[theta,cost] = fminunc(@(t)(lr_costFunction(t,X,y,lambda)), initial_theta,options);

endfunction

%===============================================================================

%Cost Function
function [J,grad] = lr_costFunction (theta,X,y,lambda)
	#Disable warnings
	warning ("off");

	m = length(y);
	n = length(X(1,:));
	J = ((1 / m) * sum(-y .* log(lr_hFunction(X,theta)) - (1 - y) .*
 	log (1 - lr_hFunction(X,theta))));
	regularizationTerm1 = (lambda/(2 * m)) * sum(theta .^ 2);

	J = J + regularizationTerm1;

	grad = (1 / m) .* sum((lr_hFunction(X,theta) - y) .* X);

	regularizationTerm2 = [0;lambda/m .* theta(2:n,:)];

	grad = grad + regularizationTerm2';

	# We have to transpose the gradient because fmincg likes it that way
	grad = grad';
endfunction

%===============================================================================

%h Function
function [result] = lr_hFunction (X,theta)
	z = theta' * X';
	result = sigmoidFunction(z)';
endfunction

%===============================================================================

%Sigmoid Function
function [result] = sigmoidFunction (z)
  result = 1 ./ (1 + e .^ (-z));
endfunction

%===============================================================================
%Function to classify examples
function prediction = lr_prediction(X, theta)
	# Adding a column of ones to X
	m = length(X(:,1));
	X = [ones(m,1),X];

	prediction = lr_hFunction(X,theta);
  prediction = prediction > 0.5;

endfunction

%===============================================================================
%Function to calculate the percentage of accuracy of the trained model
function percentageHits = lr_percentageAccuracy(X, y, theta)
	prediction = lr_prediction(X, theta);
	m = length(y);

	# Whenever the expected value and the real value are the same is a hit
	hits = sum(prediction == y);
	percentageHits = hits/m * 100;
endfunction
