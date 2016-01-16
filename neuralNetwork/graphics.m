1;
#--------------------------------------------------------------------------

#Relationship between the evolution of the error over the training examples and
#validation examples

function [] = G_nn_LearningCurves(X,errTraining, errValidation)

figure;


#Number of training examples
m = rows(X);
mVector = [1:m/5];

plot(mVector,errTraining,"color",'b',"Linewidth", 2);
xlabel("Iterations");
ylabel("Error");
hold on;
plot(mVector,errValidation,"color",'g',"Linewidth", 2);
legend("Error training","Error validation");
hold off;
title("Learning curves in neural networks")

endfunction

%===============================================================================

function [] = G_nn_RecallPrecision(recalls,precisions,opt_threshold)

figure;
plot([0.01:0.01:1],recalls,"color", 'b',"linewidth",2);
xlabel("Threshold");
ylabel("Recall/Precision");
hold on;
plot([0.01:0.01:1],precisions,"color",'g',"linewidth",2);
plot ([opt_threshold; opt_threshold], [0; 1],"color", 'm',"linestyle","--","linewidth",2);
legend("Recall","Precision", "Optimum threshold");
hold off;
title("Recall/Precision with neural networks")

endfunction
