%Function in charge of loading data for BINARY classification in memory
%Just need to have the Y column as the last column.
%This function rearrange the data in positive and negative examples and returns
%two sets (positive (1) or negative (0))
function [posExamples,negExamples] = getData(percentageData)

  printf("Loading data...");
  fflush(stdout);

  #Loads the raw data (normal or lite) from the data folder
  DATA = importdata('data/raw-data.csv',',',1);

  #Separate the parts of the CSV file
  data = DATA.data;
  textdata = DATA.textdata;
  colheaders = DATA.colheaders;

  #Sort the rows in positive/negative values (Y column is last column)
  data = sortrows(data,columns(data));

  #Saves the positive examples in posExamples
  posExamples = data(data(:,columns(data))==1,:);

  #Saves the negative examples in posExamples
  negExamples = data(data(:,columns(data))==0,:);

  #Permutate randomly the order of the positive/negative examples
  posExamples = posExamples(randperm(size(posExamples,1)),:);
  negExamples = negExamples(randperm(size(negExamples,1)),:);

  #Total number of examples
  mPos = rows(posExamples);
  mNeg = rows(negExamples);

  #Get the proper percentage
  posExamples = posExamples(1:mPos*percentageData,:);
  negExamples = negExamples(1:mNeg*percentageData,:);
  printf("\nNumber of positive examples: %i (%d %%)\n",mPos,(((mPos)/rows(data))*100));
  printf("Number of negative examples: %i (%d %%)\n",mNeg,(((mNeg)/rows(data))*100));
  printf("Total number of examples: %i\n\n", rows(data));

endfunction
