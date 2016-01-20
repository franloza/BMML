1;
source("logReg/lr_learningCurves.m");
source("logReg/lr_adjustment.m");
source("logReg/graphics.m");
source("extra/featureNormalize.m");
source("extra/sigmoidFunction.m");
warning("off");

%Main function of the logistic regression analysis
function [theta] = logReg(X,Y,lCurves)

%-----------------------------------------------------------------------------
#PARAMETERS
normalize = false; #Normalize the data or not
lambda = 0; #Regularization term (default)
percentage_training = 0.7; #Training examples / Total examples
adjusting = true; #Activates adjustment process
threshold = 0.70; #Minimun degree of certainty required

#ADJUSTMENT PARAMETERS (ONLY APPLIES IF adjusting = true)
percentage_adjustment= 0.05; #Adjustment examples / Total examples
lambdaValues = [0,0.1,0.2,0.5,1,2,5,10,15,20,25]; #Possible values for lambda
%-----------------------------------------------------------------------------

# Distribution of the examples
n_tra = fix(percentage_training * rows(X)); # Number of training examples
X_tra = X(1:n_tra,:);
Y_tra = Y(1:n_tra,:);

#Adjust the learning rate
learningFreq = fix(rows(X_tra) * 0.2);

if(adjusting)
		n_adj = fix(percentage_adjustment * rows(X)); #Number of adjustment examples
		n_val = rows(X) - (n_tra + n_adj);   #Number of validation examples
		X_adj = X(n_tra+1:n_tra + n_adj,:);
		X_val = X(n_tra + n_adj+1:rows(X),:);
		if(normalize)
				X_adj = featureNormalize (X_adj);
		endif
		Y_adj = Y(n_tra+1:n_tra + n_adj,:);
		Y_val = Y(n_tra + n_adj+1:rows(X),:);
else
		n_val = rows(X) - n_tra;  				 #Number of validation examples
		X_val = X(n_tra+1:rows(X),:);
		Y_val = Y(n_tra+1:rows(X),:);
endif;

if(normalize)
		X_tra = featureNormalize (X_tra);
		X_val = featureNormalize (X_val);
endif

# Adjustment process(Search of optimal lambda)
if(adjusting)
	[bestLambda,errorAdj] = lr_adjustment (X_tra,Y_tra,X_adj,Y_adj,lambdaValues);
	lambda = bestLambda;
endif;

#Learning Curves + training or just training

if (lCurves)
	[errTraining, errValidation,theta] = lr_learningCurves (X_tra,Y_tra,X_val,Y_val,lambda,learningFreq);

	#Save/Load the result in disk (For debugging)
	save learningCurves.tmp errTraining errValidation theta;
	#load learningCurves.tmp

	#Show the graphics of the learning curves
	G_lr_LearningCurves(X_tra,errTraining, errValidation,learningFreq);
else
	#Only Training
	theta = lr_training(X_tra,Y_tra,lambda);
endif

#Report of the training:
printf("\nLOGISTIC REGRESSION REPORT\n")
printf("-------------------------------------------------------------------:\n")
#Distribution
printf("DISTRIBUTION:\n")
printf("Training examples %d (%d%%)\n",n_tra,percentage_training*100);
if(adjusting)
printf("Adjustment examples %d (%d%%)\n",n_adj,percentage_adjustment*100);
printf("Validation examples %d (%d%%)\n",n_val,((1-(percentage_training +
percentage_adjustment))*100));
else
printf("Validation examples %d (%d%%)\n",n_val,(1-percentage_training)*100);
endif;
if(adjusting)
printf("-------------------------------------------------------------------:\n")
#Adjustment results
printf("ADJUSTMENT ANALYSIS\n")
printf("Best value for lambda: %3.2f\n",bestLambda);
printf("Error of the adjustment examples for the best lambda: %3.4f\n",errorAdj);
endif
printf("-------------------------------------------------------------------:\n")
#Error results
printf("ERROR ANALYSIS:\n")
tra_error = lr_getError(X_tra, Y_tra, theta);
printf("Error in training examples: %f\n",tra_error);
val_error = lr_getError(X_val, Y_val, theta);
printf("Error in validation examples: %f\n",val_error);
printf("Error difference: %f\n",val_error - tra_error);
printf("-------------------------------------------------------------------:\n")

#Report of the precision/recall results over the validation examples
[precision,recall,fscore] = lr_precisionRecall(X_val, Y_val,theta,threshold);
printf("PRECISION/RECALL RESULTS (USER-DEFINED THRESHOLD):\n")
printf("Threshold: %f\n",threshold);
printf("Precision: %f\n",precision);
printf("Recall: %f\n",recall);
printf("Fscore: %f\n",fscore);
printf("-------------------------------------------------------------------:\n")

[opt_threshold,precision,recall,fscore] = lr_optRP(X_val, Y_val,theta);
printf("PRECISION/RECALL RESULTS (BEST F-SCORE):\n")
printf("Optimum threshold: %f\n",opt_threshold);
printf("Precision: %f\n",precision);
printf("Recall: %f\n",recall);
printf("Fscore: %f\n",fscore);
printf("-------------------------------------------------------------------:\n")

hits = sum( lr_prediction(X_val, theta, threshold) == Y_val);
printf("ACCURACY RESULTS (USER-DEFINED THRESHOLD)\n")
printf("Threshold: %f\n",threshold);
printf("Number of hits: %d of %d\n",hits,rows(X_val));
printf("Percentage of accuracy: %3.2f%%\n",(hits/rows(X_val))*100);
printf("-------------------------------------------------------------------:\n")

[opt_threshold,hits] = lr_optAccuracy(X_val, Y_val,theta);
printf("ACCURACY RESULTS (BEST ACCURACY)\n")
printf("Optimum threshold: %f\n",opt_threshold);
printf("Number of hits %d of %d\n",hits,rows(X_val));
printf("Percentage of accuracy: %3.2f%%\n",(hits/rows(X_val))*100);
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
function [opt_threshold,precision,recall,fscore] = lr_optRP(X, y,theta)

	#Try values from 0.01 to 1 in intervals of 0.01
	for i = 1:100
		[precisions(i),recalls(i),fscores(i)] = lr_precisionRecall(X, y,theta,
			i/100);
	end

	#Select the best F-score and the threashold associated to it
	[max_Fscore, idx] = max(fscores);
	opt_threshold = (idx)/100;
	precision = precisions(idx);
	recall = recalls(idx);
	fscore = fscores(idx);
	[max_prec, idx_max_prec] = max(precisions);

	#Show the graphics of the recall-precision results
	G_lr_RecallPrecision(recalls,precisions,opt_threshold);

endfunction

%===============================================================================
%Function to extract the optimum threshold that guarantees the maximum number
%of hits given a trained model over a set of examples
function [opt_threshold,max_hits] = lr_optAccuracy(X, y,theta)

	#Try values from 0.01 to 1 in intervals of 0.01
	for i = 1:100
		[hits(i)] = sum( lr_prediction(X, theta, i/100) == y);
	end

	#Select the best F-score and the threashold associated to it
	[max_hits, idx] = max(hits);
	opt_threshold = (idx)/100;

	#Show the graphics of the recall-precision results
	G_lr_Accuracy(hits,opt_threshold,rows(X));

endfunction

%===============================================================================
%Function to calculate the error produced by theta over a set of examples
function error= lr_getError(X, y, theta)
	m = rows(X);
	X = [ones(m,1),X];
	error =  lr_costFunction(theta,X,y,0);
endfunction
