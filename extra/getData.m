1;

#The information of this function is in the file "getClassificationData.info"
function [X,Y] = getClassificationData (DATA)

data = DATA.data;
textdata = DATA.textdata;
colheaders = DATA.colheaders;

# Information fields

# X(:,1) - quarter
X(:,1) = data(:,1);

# X(:,2)  - month
#The months in data are classificated in their position in the calendar (From
# January to June, 1 to 6). We need the number of month of the quarter (1 to 3).
# That's why it's necessary to apply modulus 3 and covert the 0's in 3's
months = mod(data(:,3),3);
idx = months  == 0;
months(idx) = 3;
X(:,2) = months;

# Price fields

# X(:,3) - price_difference
X(:,3) = data(:,7) - data(:,4);

# X(:,4) - price_range
X(:,4) = data(:,5) - data(:,6);

# X(:,5) - next_week_price_difference
X(:,5) =  data(:,13) - data(:,12);

# X(:,6) - percent_change_price
X(:,6) = data(:,9);

# X(:,7) - volume
X(:,7) = data(:,8);

# X(:,8) - previous_week_volume
X(:,8) = data(:,11);

# X(:,9) - percent_change_volume_over_last_week
X(:,9) = data(:,10);

# Additional information

#X(:,10) - days_to_next_dividend
X(:,10) = data(:,15);

#X(:,11) - percent_return_next_dividend
X(:,11) = data(:,16);

#Convert NA into zeroes
X(isnan(X)) = 0;

# Y set

#Y(:,1) - profitable
profitable = data(:,14);
Y(:,1) = (profitable > 0);


endfunction
