%Function that apply the sigmoid formula
function [result] = sigmoidFunction (z)
  result = 1 ./ (1 + e .^ (-z));
endfunction