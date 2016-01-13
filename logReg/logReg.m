1;
source("logReg/learningCurves.m");
source("logReg/graphics.m");
source("extra/featureNormalize.m");
source("extra/expandFeatures.m");
warning("off");

%Main function of the logistic regression analysis
function [theta] = logReg(X,Y,lCurves)

#PARAMETERS
lambda = 0;
ex_q1 = 360;    #Examples of training
ex_q2 = rows(X) - ex_q1;   #Examples of validation

#Extension of the features
exp_X = expandFeatures(X);

# Distribution of the examples (With normalization)

%Training with the first quarter
X_tra = featureNormalize (exp_X(1:ex_q1,:));
Y_tra = Y(1:ex_q1,:);

%Validating with the second quarter
X_val = featureNormalize (exp_X(ex_q1+1:rows(X),:));
Y_val = Y(ex_q1+1:rows(X),:);

#Learning Curves + training or just training

if (lCurves)
	[errTraining, errValidation,theta] = learningCurves (X_tra,Y_tra,X_val,Y_val,
																																				lambda);
	#Save the result in disk
	save learningCurves.tmp errTraining errValidation;
	#load learningCurves.tmp
	G_LearningCurves(X_tra,errTraining, errValidation);
else
	#Only Training
	theta = lr_training(X_tra,Y_tra,lambda);
endif


#Report of the training:
printf("LOGISTIC REGRESSION REPORT\n")
printf("-------------------------------------------------------------------:\n")

#Error results
printf("ERROR ANALYSIS:\n")
tra_error = lr_getError(X_tra, Y_tra, theta);
printf("Error in training examples: %f\n",tra_error);
val_error = lr_getError(X_val, Y_val, theta);
printf("Error in validation examples: %f\n",val_error);
printf("-------------------------------------------------------------------:\n")

#Report of the optimum values
[opt_threshold,precision,recall,fscore] = lr_optThreshold(X_val, Y_val,theta);
printf("OPTIMUM THRESHOLD IN VALIDATION EXAMPLES:\n")
printf("Optimum threshold: %f\n",opt_threshold);
printf("Precision: %f\n",precision);
printf("Recall: %f\n",recall);
printf("Fscore: %f\n",fscore);
printf("-------------------------------------------------------------------:\n")


endfunction

%===============================================================================

%Training function
function [theta,cost] = lr_training(X,y,lambda)
	m = length(y);
	n = length(X(1,:));

	# Adding a column of ones to X
	X = [ones(m,1),X];

	initial_theta = zeros(n + 1, 1);
	theta = initial_theta;

	#Optimization
	options = optimset('GradObj','on','MaxIter',1000);
	[theta,cost] = fminunc(@(t)(lr_costFunction(t,X,y,lambda)), initial_theta,
																																			options);

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
function prediction = lr_prediction(X, theta, threshold)
	# Adding a column of ones to X
	m = length(X(:,1));
	X = [ones(m,1),X];

	prediction = lr_hFunction(X,theta);
  prediction = prediction > threshold;

endfunction

%===============================================================================
%Function to extract the precision and the recall of a trained model given a
%threshold
function [precision,recall,fscore] = lr_precisionRecall(X, y,theta,threshold)
	#Get the predicted y values
	pred_y = lr_prediction(X, theta,threshold);

	#Precision calculation

	true_positives = sum(pred_y & y); #Logic AND to extract the predicted
																		#positives that are true
	pred_positives = sum(pred_y);
	precision = true_positives / pred_positives;

	#Recall calculation
	actual_positives = sum(y);
	test = [pred_y,y,pred_y&y];
	recall = true_positives / actual_positives;

	#F-score calculation
	fscore =  (2*precision*recall) / (precision + recall);

endfunction

%===============================================================================
%Function to extract the optimum threshold that guarantees the best trade-off
%between precision and the recall of a trained model
function [opt_threshold,precision,recall,fscore] = lr_optThreshold(X, y,theta)

	#Try values from 0.01 to 0.99 in intervals of 0.01
	#NOTE:It starts in 1 because octave starts its arrays in 1.
	#That's why I sum one to the index
	for i = 0:100
		[precisions(i+1),recalls(i+1),fscores(i+1)] = lr_precisionRecall(X, y,theta,
			i/100);
	end

	#Select the best F-score and the threashold associated to it
	[max_Fscore, idx] = max(fscores);
	opt_threshold = (idx-1)/100;
	precision = precisions(idx);
	recall = recalls(idx);
	fscore = fscores(idx);
	[max_prec, idx_max_prec] = max(precisions);

	#Show the graphics
	title("Optimum threshold graphic")
	plot([0:0.01:1],recalls,"color", 'b',"linewidth",2);
	xlabel("Threshold");
	ylabel("Recall/Precision");
	hold on;
	plot([0:0.01:1],precisions,"color",'g',"linewidth",2);
	plot ([opt_threshold; opt_threshold], [0; 1],"color", 'm',"linestyle","--","linewidth",2);
	plot((idx_max_prec-1)/100,max_prec,"marker",'x',"color",'r',"markersize",10);
	legend("Recall","Precision", "Optimum threshold","Maximum precision");
	hold off;
endfunction

%===============================================================================
%Function to calculate the error produced by theta over a set of examples
function error= lr_getError(X, y, theta)
	m = rows(X);
	X = [ones(m,1),X];
	error =  lr_costFunction(theta,X,y,0);
endfunction
