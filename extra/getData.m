1;

#The information of this function is in the file "getClassificationData.info"
function [X,Y] = getClassificationData (DATA)

data = DATA.data;
textdata = DATA.textdata;
colheaders = DATA.colheaders;


# X(:,1) - price_difference
X(:,1) = data(:,7) - data(:,4);

# X(:,2) - price_range
X(:,2) = data(:,5) - data(:,6);

# X(:,3) - percent_change_price
X(:,3) = data(:,9);

# Volume fields

# X(:,4) - volume
X(:,4) = data(:,8);

# X(:,5) - previous_week_volume
X(:,5) = data(:,11);

# X(:,6) - percent_change_volume_over_last_week
X(:,6) = data(:,10);

#Convert NA into zeroes
X(isnan(X)) = 0;

# Y set

#Y(:,1) - profitable
profitable = data(:,14);
Y(:,1) = (profitable > 0);


endfunction

function [X] = expandFeatures (X) 

	# X(:,7) - Ratio price difference and range

	X(:,7) = X(:,1) ./ X(:,2);

	#Square of each feature (NOTE:FAILS EXPANDING THE FOURTH COLUMN)
	
	for i=1:3
		X(:,7+i) = X(:,i).^2;
	end
	
endfunction
