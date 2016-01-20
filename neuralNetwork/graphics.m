1;
%===============================================================================

#Relationship between the evolution of the error over the training examples and
#validation examples

function [] = G_nn_LearningCurves(X,errTraining, errValidation,learningFreq)

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
title("Learning curves in neural networks")

endfunction

%===============================================================================

%Plots a function that shows the relationship between the evolution of the error
%over the training examples and adjustment examples as we increase the lambda
%value

function [] = G_nn_adjustLambda(errTraining, errAdjustment,lambdaValues)

figure;

plot(lambdaValues,errTraining,"color",'b',"Linewidth", 2);
xlabel("Values of lambda");
ylabel("Error");
hold on;
plot(lambdaValues,errAdjustment,"color",'g',"Linewidth", 2);
legend("Error training","Error adjustment");
hold off;
title("Adjusting lambda in neural networks")

endfunction
%===============================================================================

%Plots a function that shows the relationship between the evolution of the error
%over the training examples and adjustment examples as we increase the lambda
%value

function [] = G_nn_adjustNodes(errTraining, errAdjustment,hiddenNodes)

figure;

plot(hiddenNodes,errTraining,"color",'b',"Linewidth", 2);
xlabel("Number of hidden nodes");
ylabel("Error");
hold on;
plot(hiddenNodes,errAdjustment,"color",'g',"Linewidth", 2);
legend("Error training","Error adjustment");
hold off;
title("Adjusting hidden nodes in neural networks")

endfunction

%===============================================================================
%Relationship between the probability of chosing the right output value
function [] = G_nn_probPredictions(X,Theta1,Theta2)

prediction = nn_hFunction(X,Theta1, Theta2)';
posPrediction = prediction(1,:);
negPrediction = prediction(2,:);

figure;
axis([0,columns(prediction),0,1]);
plot([1:columns(prediction)],posPrediction,"color", 'g',"linewidth",2);
xlabel("n training example");
ylabel("Probability");
hold on;
plot([1:columns(prediction)],negPrediction,"color", 'r',"linewidth",2);
legend("Positive prediction (1)","Negative prediction (0)");
hold off;
title("Recall/Precision with neural networks")

endfunction
