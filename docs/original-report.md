# Iris Species Classification 

# Contributors:
- Alexis Pendleton
- Andrew Dorado
- Neira Ibrahimovic
- Lillian Gabrelian
# Project Goal: 
The goal of our project is to classify the iris data set and create a model that takes in measurements as parameters and outputs the correct iris flower classification. 
We use graphs to explore the data and aid us in creating a decision tree, then compare the decision tree we create to a computer-generated decision tree.
We intend to optimize the accuracy of our model so we could predict the correct flower with >90% accuracy.
[Link to demo file video](https://youtu.be/23Iec5DPY7k)
# The Dataset:
The Iris dataset can be found on the UCI Machine Learning Repository.

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

  - **Id:** entry number of the flower
  - **SepalLengthCm:** Length of the sepal (in cm)
  - **SepalWidthCm:** Width of the sepal (in cm)
  - **PetalLengthCm:** Length of the petal (in cm)
  - **PetalWidthCm:** Width of the petal (in cm)
  - **Species:** Iris species name

*It should be noted that our group dropped the Id column in this data set and changed the species names to "versicolor", "setosa", and "virginica".*
  
# Python packages used:
  - numpy 1.21.5
  - matplotlib 3.5.1
  - pandas 1.4.2
  - sklearn 1.0.2
  - sys 3.9.12
  
# Demo File Description:
We suggest having Anaconda installed on your computer to be able to run our project most easily. Please ensure that all of the packages, in the versions that we’ve specified above, are also downloaded on your computer. To run our demo file, download all of the files in this git repository as a .zip file. Then, on your computer, unzip the files into a folder. Once you’ve done this, open Demo_IrisSpeciesClassification.ipynb with JupyterLab or JupyterNotebook and run all of the cells in the file to see how our program works!
 
We imported the iris dataframe from the provided 'iris.csv' file as a pandas dataframe.

## Data Visualization

The first step was exploring the data using scatterplots and histograms. 
We created a function in our class for visualizing both the scatterplots and histograms. 
We created an instance of our class then called the respective functions.

### Scatterplot Comparisons
We first looked at a scatterplot showing the relationship between petal length and petal width for all three flower species.
We observed that setosa had the smallest petal length and width, virginica generally had the largest, and versicolor had 
petal length/widths that was mostly between the other two species. 
![image](https://user-images.githubusercontent.com/119445388/206567426-6d94e7e2-736d-4599-a61a-341bbc861ecf.png)


Next, we looked at a scatterplot showing the relationship between sepal length and sepal width for the three iris species. 
We were not able to find a way to distinguish the species from one another from this scatter plot, as there was a lot of overlap
between the species. 

![image](https://user-images.githubusercontent.com/119445388/206569431-a1db3ed0-89cd-4e25-a605-e1d449f1b667.png)

We then looked at graphs showing the petal length, petal width, sepal length, and sepal width, plotted by species, 
so we could observe trends that differ between the species. Below is an example of one of these graphs. From this graph, it is evident that setosa irises generally have the smallest petal widths, and can be distinguished quite well from the other two species because of a lack of overlap with their petal widths and those for versicolor and virginica species. 
![image](https://user-images.githubusercontent.com/119445388/206569521-3647c4a0-0a98-44bb-b6f3-9436f58d9444.png)

### Histogram Comparisons
After exploring the scatter plots, we further explored our data using histograms. 
When looking at the histograms for sepal length and sepal width, we found there was a lot of overlap
among the species, so the species were not easily distinguishable using these measurements.
Therefore, we decided that these measurements would not be the most useful when creating our own decision tree.

![image](https://user-images.githubusercontent.com/119445388/206569604-46bd05dc-a5ed-4c48-a7d9-769ef4778824.png)
![image](https://user-images.githubusercontent.com/119445388/206569653-b219930b-a9b3-45c7-8289-369f477b49e9.png)

On the other hand, when we observed the histograms for petal length and petal width, we found that the three species       
were easily distinguishable from each other, so these measurements could be used to differentiate between the species. 

In the graph for petal width, we could distinguish setosa from the rest of the flowers by the fact that every setosa flower has a petal width less than 0.8 cm. 
This was the first branch in our decision tree. 
Then, we can distinguish virginica from the rest of the flowers based on the fact that most of its flowers have a petal width above 1.75 cm. 
This was the next branch of our decision tree.

![image](https://user-images.githubusercontent.com/119445388/206569735-f7ec8eaf-38af-4a09-99e7-68447038d39f.png)

In the graph for petal length, we could distinguish versicolor from the flowers we already identified by looking at petal lengths less than or equal to 4.95 cm.

![image](https://user-images.githubusercontent.com/119445388/206569794-d04620ad-68a2-44d3-988c-82ac53ba0d72.png)

### Using Summary Tables
In order to estimate the bounds for the decision tree, two summary plots were made: one on sepal dimensions and the other on petal dimensions. 
We found that there was significant overlap in the sepal dimensions for the three species. On the other hand, the three species had distinct
proportions for petal dimensions. Setosas had very small petals, versicolors had medium sized petals, and virginicas had very large petals.
These means were the basis of our decision tree.

### Using max_min Function
Next, in order to create our decision tree we need to ensure that we are considering all of the values for the features we selected. We did this by running our max_min function on the features “petal width” and “petal length”. We chose these features because we found significant differences for each species in our data visualization section.

The max_min function was run for all species on both features however, only 2 examples of this are shown below.

  **Example 1**

<img width="351" alt="example 1" src="https://user-images.githubusercontent.com/117899786/206571671-2ee448c0-3787-4dca-b6d4-ca2495f2a886.png">

  **Example 2**

<img width="378" alt="example 2" src="https://user-images.githubusercontent.com/117899786/206571707-1824e067-3094-43d9-b390-f851a346b63e.png">

**Example of Exception Handling**

![maxminerror](https://user-images.githubusercontent.com/114278416/206574909-f12bd546-5ec5-46ae-ac95-a828f9e692d3.png)

  **Discussion:**

After running the max_min function, determine which species and features have no overlap. We found that Petal Width for “setosa” had no overlap thus, making it a good feature for our model. In addition, we found that Petal Length for “virignica” had no overlap in it’s data, also making it a good feature for our model.


## Creating Our Decision Tree

The second step was creating our own decision tree. 
<img width="701" alt="Screen Shot 2022-12-08 at 7 58 18 AM" src="https://user-images.githubusercontent.com/119445364/206567680-07310de8-386c-4f08-b37c-426aa9a9b7c2.png">

Our decision tree code is a class method of. It takes in floats or ints, and it will raise type errors if a different data type is inputted. 
Because setosas were much smaller than the other two species, the first if statement checks if the petal length is less than 0.8 cm, which
is greater than the maximum length, and returns 'setosa' if true. Then it differentiates between virginica and versicolors by petal widths.

**Exception Handling Example**

![decisiontreeerror](https://user-images.githubusercontent.com/114278416/206575493-9d5a230c-9f36-4f20-817b-928bbbd58370.png)
 
## Our Decision Tree vs. Computer Generated Decision Tree
Then we compared our decision tree to the computer generated tree.

### Making Computer Generated Decision Tree
We made this function below for the computer decision tree and ran it with a depth of 3.

<img width="699" alt="Screen Shot 2022-12-08 at 1 41 40 PM" src="https://user-images.githubusercontent.com/117899786/206573325-0100c0cc-3d52-4913-bb2e-b9afb7d30f77.png">


Here is the figure of the computer generated decision tree:

<img width="754" alt="Screen Shot 2022-12-08 at 1 42 54 PM" src="https://user-images.githubusercontent.com/117899786/206573526-c73a5222-40b7-4a2d-b378-d91b09b53c9b.png">

### Scoring Computer's Decision Tree
After we split our data into test and train data for the computer's decision tree to see a more accurate view of it's performance.

<img width="399" alt="Screen Shot 2022-12-08 at 1 44 01 PM" src="https://user-images.githubusercontent.com/117899786/206573716-ddbfb503-1ccc-48f9-a455-dbb0325b4623.png">

### Scoring Our Decision Tree
Finally we used a lmabda function to figure out the score of our own decision tree, that code can be seen below.

<img width="474" alt="Screen Shot 2022-12-08 at 1 44 47 PM" src="https://user-images.githubusercontent.com/117899786/206573857-f4ff7001-8240-4d44-ad3e-994789c11d43.png">

After looking at the score's of the computer's decision tree and ours we can conclude that our model didn't show any signs of overfitting.

# Scope and Limitations:
The dataset used is very small and only provides 150 samples' data. Because of this small dataset, it can not be seen as statistically significant data.
This dataset is also from 1936 and may not be representative of today’s irises.

    
# References and Acknowledgement:
This project utilizes the Iris dataset that was originally collected by Edgar Anderson, and is found in the paper “The species problem in Iris”, and was compiled into a dataset by Ronald Fisher in his 1936 paper “The use of multiple measurements in taxonomic problems". This dataset can be acquired from [Kaggle.](https://www.kaggle.com/datasets/uciml/iris) It can also be found in UCI’s Machine Learning Repository [here.](https://archive.ics.uci.edu/ml/datasets/Iris)

Edgar Anderson (1936). "The species problem in Iris". Annals of the Missouri Botanical Garden. 23 (3): 457–509. doi:10.2307/2394164.
R. A. Fisher (1936). "The use of multiple measurements in taxonomic problems". Annals of Eugenics. 7 (2): 179–188. doi:10.1111/j.1469-1809.1936.tb02137.x. 
Code used was derived from lecture and discussions by Dr. Harlin Lee.
