1;
source("extra/fmincg.m");
warning("off");

%Main function of the logistic regression analysis
function [theta] = logReg(X,Y)

#PARAMETERS
lambda = 1;
num_labels = 2; %Profitable or non-profitable

#Training
theta = lr_training(X,Y,num_labels,lambda);

#Precision Analysis
percentageHits = lr_percentageAccuracy(X, Y, theta)

endfunction

%===============================================================================

%Training function
function [all_theta] = lr_training(X,y,num_etiquetas,lambda)
  m = length(y);
	n = length(X(1,:));

	# Adding a column of ones to X
	X = [ones(m,1),X];

	initial_theta = zeros(n + 1, 1);

	options = optimset("GradObj", "on", "MaxIter", 1000);
	theta = initial_theta;
	for c = 1:num_etiquetas
		[theta] = fmincg(@(t)(lr_costFunction(t, X, (y == c-1), lambda)), theta,
    options);
		# In each iteration we add a new column to our theta
		all_theta(:,c) = theta;
	endfor
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
function etiqueta = lr_prediction(X, theta)
	# Adding a column of ones to X
	m = length(X(:,1));
	X = [ones(m,1),X];

	etiqueta = lr_hFunction(X,theta);

	# We have to transpose etiqueta because the way max works
	[useless, etiqueta] = max(etiqueta');
endfunction

%===============================================================================
%Function to calculate the percentage of accuracy of the trained model
function percentageHits = lr_percentageAccuracy(X, y, theta)
	prediction = lr_prediction(X, theta);
	m = length(y);

	# Whenever the expected value and the real value are the same is a hit
	hits = sum(prediction == y');
	percentageHits = hits/m * 100;
endfunction
