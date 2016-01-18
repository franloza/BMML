%Function in charge of loading the data in memory
%If lite is true, it loads the "light" version of the data. Ideal for
%testing computational demanding algorithms such as SVM
function [X,Y] = getData(lite)

  #Loads the raw data (normal or lite) from the data folder
  if (lite)
    DATA = importdata('data/raw-data-lite.csv',',',1);
  else
    DATA = importdata('data/raw-data.csv',',',1);
  endif;

  #Separate the parts of the CSV file
  data = DATA.data;
  textdata = DATA.textdata;
  colheaders = DATA.colheaders;

  #Get the X and the Y variables
  X = data(:,1:19);
  Y = data(:,20);

endfunction
