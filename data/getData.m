%Function in charge of loading the data in memory
%If lite is true, it loads the "light" version of the data. Ideal for
%testing computational demanding algorithms such as SVM
function [X,Y] = getData(percentageData)

  #Loads the raw data (normal or lite) from the data folder
  DATA = importdata('data/raw-data.csv',',',1);

  #Separate the parts of the CSV file
  data = DATA.data;
  textdata = DATA.textdata;
  colheaders = DATA.colheaders;

  #Permutate randomly the order of the examples
  data = data(randperm(size(data,1)),:);

  #Total number of examples
  m = rows(data);

  #Get the X and the Y variables
  X = data(1:m*percentageData,1:19);
  Y = data(1:m*percentageData,20);

endfunction
