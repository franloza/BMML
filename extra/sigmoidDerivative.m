%Function that computes the derivative of the sigmoid function
function [result] = sigmoidDerivative (z)

a = sigmoidFunction(z);
a = [ones(1, columns(a)); a];

result = a .* (1 - a);

endfunction