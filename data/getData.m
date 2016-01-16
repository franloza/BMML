1;

#The information of this function is in the file "getClassificationData.info"
function [X,Y] = getClassificationData (DATA)

data = DATA.data;
textdata = DATA.textdata;
colheaders = DATA.colheaders;

#Price fields

# X(:,1) - price_difference
X(:,1) = data(:,7) - data(:,4);

# X(:,2) - price_range
X(:,2) = data(:,5) - data(:,6);

# X(:,3) - percent_change_price
X(:,3) = data(:,9);

#Dividend fields

# X(:,4) - days_to_next_dividend
X(:,4) = data(:,15);

# X(:,5) - percent_return_next_dividend
X(:,5) = data(:,16);

# Volume fields

# X(:,6) - volume

#X(:,6) = data(:,8);

# X(:,7) - previous_week_volume
#X(:,7) = data(:,11);

# X(:,8) - percent_change_volume_over_last_week
#X(:,7) = data(:,10);


#Convert NA into zeroes
X(isnan(X)) = 0;

# Y set

#Y(:,1) - profitable
profitable = data(:,14);
Y(:,1) = (profitable > 0);


endfunction
