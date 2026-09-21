import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

class IrisClassifier(object):
    '''
    A class for analyzing, visualizing, and classifying data from the Iris Species dataset
    (link: https://www.kaggle.com/datasets/uciml/iris) from Kaggle.
    
    Contains methods for visualizing and observing subsets of the data, as well as a method for
    predicting the species of Iris given its petal length and width.
    '''
    def __init__(self, data):
        '''
        Initialize IrisClassifier object with a Pandas dataframe that includes data on Irises
        Args: 
            data: a Pandas dataframe
        Returns:
            None
        '''
        # Retain the caller's dataframe reference. The demo normalizes species
        # labels before construction; this class does not silently rename them.
        if isinstance(data, pd.DataFrame):
            self.irises = data
        # otherwise, raise a TypeError
        else:
            raise TypeError("Data is not and must be a Pandas dataframe.")
    
    
    def decision_tree(self, PetalWidthCm, PetalLengthCm):
        '''
        Predicts type of Iris using petal width and length in cm.
        Args: 
            PetalWidthCm: a float or int
            PetalLengthCm: a float or int
        Returns: 
            A string that is the predicted Iris species.
        '''
        if (type(PetalWidthCm) not in [float, int] or type(PetalLengthCm) not in [float, int]):
            raise TypeError("This function is designed to work only with floats or ints")
        # Original fixed thresholds derived from exploratory plots. Width is
        # checked first; length resolves the remaining overlapping species.
        if PetalWidthCm < 0.8:
            return "setosa"
        else:
            if PetalWidthCm > 1.75:
                return "virginica"
            elif PetalLengthCm <= 4.95:
                return "versicolor"
            else:
                return "virginica"
    
    
    def summary_table(self, group_cols, val_cols):
        '''
        Create a summary table of the irises dataframe that groups the column(s) in group_cols
        by the values in val_cols.
        Args:
            group_cols: a string or list of strings containing the name(s) of column(s) in irises
            val_cols: a string or list of strings containing the name(s) of column(s) in irises
        Returns: 
            A table summarizing these groupings
        '''
        # create list that holds all of the names of columns in the irises dataframe
        iris_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]
        
        
        # check that parameters entered into group_cols and val_cols are present in iris_cols
        # Validate both arguments independently: a valid value column must not
        # overwrite a failed group-column check. Empty selections are invalid.
        def valid_columns(columns):
            names = [columns] if isinstance(columns, str) else columns
            return (isinstance(names, (list, tuple)) and bool(names)
                    and all(isinstance(name, str) and name in iris_cols
                            and name in self.irises.columns for name in names))

        params_correct = valid_columns(group_cols) and valid_columns(val_cols)
            
        # if the parameters entered into the function include columns of the dataframe
        # then, return the table made by grouping the group_cols by value_cols
        # which calculates the mean and standard deviation for each group/value combination.
        if params_correct == True:
            # Named operations preserve pandas' sample-standard-deviation
            # semantics across versions instead of dispatching NumPy callables.
            return self.irises.groupby(group_cols)[val_cols].aggregate(["mean", "std"])
        #elif all(item in iris_cols for item in val_cols):
           # return self.irises.groupby(group_cols)[val_cols].aggregate([np.mean, np.std])
        # otherwise, raise a NameError
        else:
            raise NameError("The parameters entered do not match the names of the columns in the irises dataframe. Please make sure that the values you enter for group_cols and val_cols include only the following names: 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'")
    
    
    def visualize_hist(self, data, x_axis):
        '''
        Function that plots a histogram showing all the species and user-inputted parameter for x-axis
            Args: User inputs string which is one of the four column values given in the dataframe
            Outputs: histogram with all species shown and user-inputted parameter for x-axis
        '''
        #check to make sure user inputted one of the columns in the dataframe
        if x_axis in ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]:
            #graph the histogram grouped by species and with x-axis values given by user input
            #pivot creates reshaped dataframe, setting "Species" equal to columns makes three columns with species names
            #the new dataframe will have values in each cell that correspond with the columns of the original dataframe  
            #plot.hist is used to plot this reshaped dataframe
            data.pivot(columns = "Species", values = x_axis).plot.hist(bins = 25, alpha = 0.5)
            
            # add labels for x-axis and title
            ttl = x_axis + " with Frequency per Species"
            plt.xlabel(x_axis)
            plt.title(ttl)
            
            #show the histogram
            plt.show()
            #if user inputted string that is not a column name, output error message
        else:
            print("The column you inputted does not exist. Please input one of the following columns:\n'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm' for the x axis.")
            #exit the program 
            sys.exit(0)
    
    
    def visualize_scatter(self, x, y, color=["blue", "green", "red"]):
        '''
        Create and plot a scatterplot of different features of Irises that color codes points 
        based on the species of the Iris. Plot y vs x. 
        Args:
            x: a column of the Iris dataframe
            y: another column of the Iris dataframe
            color: a list containing 3 string elements that are color names
        Returns:
            None, just outputs the scatterplot created
        '''
        # create an empty plot with a single axis
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5) # adjust the length and width of plot
        
        # set title
        ttl = x + " vs. " + y
        ax.set(xlabel=x, ylabel=y, title=ttl)

        # define a dict colors, which assigns element 0 in the color list to the "setosa" iris
        # assigns element 1 in the color list to the "versicolor" iris
        # and assigns element 2 in the color list to the "virginica" iris
        # this will be used to color code the points of the scatterplot by species
        colors = {"setosa":color[0], "versicolor":color[1], "virginica":color[2]}
        
        # for every species in the irises dataframe, and for every smaller dataframe containing
        # data on just that species
        # plot column x of that dataframe against column y
        # pass in the species name as the label for that plot, and set the color of the 
        # points corresponding to the species name
        # Scalar grouping keeps species keys as strings in modern pandas.
        # A one-element list can yield tuples, which do not match the color map.
        for species, df_species in self.irises.groupby("Species"):
            ax.scatter(df_species[x], df_species[y], label=species, facecolor=colors[species])

        # show the legend indicating which color corresponds to which species on the plot
        ax.legend()
        
    
    def max_min(self, species, feature):
        '''
        Function that finds the minimum and maximum value entered for a specific feature of the data and a specific species.
        If the species or feature input is not in the iris dataframe a value error is raised.
            Args:
                species: a string input. The specific iris species that the user wants information on that is in the iris dataframe.
                feature: a string input. A feature or the iris species that is in the iris dataframe.
            Returns:
                None
        '''
        self.species = species
        self.feature = feature
        accepted_inputs = {"setosa", "versicolor", "virginica"}
        accepted_features = {"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"}
        # if species input is in excepted inputs dict and feature input is in accepted features dict 
        if self.species in accepted_inputs and self.feature in accepted_features:
            bool_idx = self.irises["Species"] == species # creates boolean array for which rows match user's inputted species
            species_info = self.irises[bool_idx] # rows containing info only from selected species
            species_feature_info = species_info[feature] # gets information on selected species for a specific feature of data
            min_feature = min(species_feature_info) # minimum value of the feature for selected species
            max_feature = max(species_feature_info) # max value of the feature for selected species
            # print statement outputted describing the mins and maxes
            print(f"\nThe min {feature} for {species} is {min_feature}.\n \nThe max {feature} for {species} is {max_feature}.\n")
        else:
            # raise a value error if species or input is not in the iris dataframe
            raise ValueError("User input for species and feature must be in the iris dataframe." )
   
