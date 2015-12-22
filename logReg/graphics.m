1;
#--------------------------------------------------------------------------

#Relationship between the evolution of the error over the training examples and
#validation examples

function [] = G_LearningCurves(X,errTraining, errValidation)

figure;
#Number of training examples
m = rows(X);
mVector = [11:m];

plot(mVector,errTraining,"color",'b',"Linewidth", 2);
title("Learning curves");
xlabel("Iterations");
ylabel("Error");
hold on;
plot(mVector,errValidation,"color",'g',"Linewidth", 2);
legend("Error training","Error validation");
hold off;

endfunction

#--------------------------------------------------------------------------
