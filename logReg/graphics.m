1;
%===============================================================================

%Relationship between the evolution of the error over the training examples and
%validation examples as we increase the number of training examples

function [] = G_lr_LearningCurves(X,errTraining, errValidation,learningFreq)

figure;

#Number of training examples
m = rows(X);
mVector = [1:columns(errTraining)];

iterationsStr = sprintf("Iterations (x%i)", learningFreq);
plot(mVector,errTraining,"color",'b',"Linewidth", 2);
xlabel(iterationsStr);
ylabel("Error");
hold on;
plot(mVector,errValidation,"color",'g',"Linewidth", 2);
legend("Error training","Error validation");
hold off;
title("Learning curves in logistic regression")

endfunction

%===============================================================================

%Plots a function that shows the relationship between the evolution of the error
%over the training examples and adjustment examples as we increase the lambda
%value

function [] = G_lr_Adjustment(errTraining, errAdjustment,lambdaValues)

figure;

plot(lambdaValues,errTraining,"color",'b',"Linewidth", 2);
xlabel("Values of lambda");
ylabel("Error");
hold on;
plot(lambdaValues,errAdjustment,"color",'g',"Linewidth", 2);
legend("Error training","Error adjustment");
hold off;
title("Adjustment process in logistic regression")

endfunction

%===============================================================================

%Plots a function that shows the relationship between increasing the threshold
%and the evolution of the precision and the recall. Also points the optimum
%threshold
function [] = G_lr_RecallPrecision(recalls,precisions,opt_threshold)

figure;
plot([0.01:0.01:1],recalls,"color", 'b',"linewidth",2);
xlabel("Threshold");
ylabel("Recall/Precision");
hold on;
plot([0.01:0.01:1],precisions,"color",'g',"linewidth",2);
plot ([opt_threshold; opt_threshold], [0; 1],"color", 'm',"linestyle","--",
"linewidth",2);
legend("Recall","Precision", "Optimum threshold");
hold off;
title("Recall/Precision with logistic regression")

endfunction

%===============================================================================

%Plots a function that shows the relation between increasing the threshold and
%the evolution of the precision and the recall. Also points the optimum threshold
function [] = G_lr_Accuracy(hits,opt_threshold,m)

figure;
percentages = (hits./m).*100;
plot([0.01:0.01:1],percentages,"color", 'b',"linewidth",2);
xlabel("Threshold");
ylabel("Percentage of hits(%)");
hold on;
plot ([opt_threshold; opt_threshold], [0; 100],"color", 'm',"linestyle","--",
"linewidth",1);
legend("Percentage of hits","Optimum threshold");
hold off;
title("Accuracy with logistic regression");

endfunction

%===============================================================================
