1;
function [X] = expandFeatures (X)

	# X(:,7) - Ratio price difference and range

	X(:,7) = X(:,1) ./ X(:,2);

	#TODO: Expand the features wisely

endfunction
