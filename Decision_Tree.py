# ---------------------------------------------------------------------------------------------------------------------
# File Name: main.py
# Author: Kate McAlpine
# Date Created: 17 May 2025
# Last Modified: 19 May 2025
#
# Note:
# Code has been developed in an intentionally naive way for use of reading by people unfamiliar with Machine Learning
# code, or code in general. This aids in understanding, appropriation, and troubleshooting.
#
# Purpose:
# Use the Iris dataset developed by Robert A. Fisher in 1936, to explore Multivariate Analysis in Machine Learning,
# focusing on discriminant analysis. I have used various graphing techniques to represent and compare the Iris dataset,
# the Decision Tree Classification model to classify, then further analysis through confusion matrix and other
# calculations to test the results.
#
# References (NB: Some of the info in the references are incorrect or outdated, but I have updated the theoretical and
# programming issues while developing my own code, so it is correct information with current components):
# Iris dataset: https://archive.ics.uci.edu/dataset/53/iris
# https://www.graphviz.org/
# https://medium.com/analytics-vidhya/mushroom-classification-using-different-classifiers-aa338c1cd0ff
# https://www.kaggle.com/code/louisong97/neural-network-approach-to-iris-dataset
# https://www.tutorialspoint.com/seaborn/seaborn_catplot_method.htm
# https://stackoverflow.com/questions/57047711/how-do-i-display-only-every-nth-axis-label
# https://www.geeksforgeeks.org/how-to-set-axes-labels-limits-in-a-seaborn-plot/
# https://seaborn.pydata.org/generated/seaborn.catplot.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://www.tutorialspoint.com/seaborn/seaborn_plotting_categorical_data.htm
# https://www.geeksforgeeks.org/matplotlib-axes-axes-set_xlim-in-python/
# https://www.geeksforgeeks.org/python-seaborn-catplot/
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html
# https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# ---------------------------------------------------------------------------------------------------------------------
import pandas as pd  # Version 2.2.3
import numpy as np  # Version 2.2.6
import matplotlib.pyplot as plt  # Version 3.10.3
import seaborn as sns  # Version 0.13.2
import os
import graphviz  # Version 0.20.3
# Must install scikit-learn package for the below functions. Version 1.6.1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix

# Investigating iris.data contents, assuming the data file is in the same folder as the project:
irisData = pd.read_csv(r'iris.data')  # Read in the data convert into dataframe (2D)
print("Dataset shape:\n", irisData.shape, "\n")  # Find out how many columns and rows
# Output shows 5 columns, but only 149 rows (ie: sample points). However upon inspection, there are 150 sample points,
# but the first line is not included in the assessment, assuming that is where the column headings are to be located.
print("First 5 rows of dataset:\n", irisData.head(), "\n")  # Print the first 5 rows of the dataset
# Output shows the data captured as indexing from the second row.

# There are 5 columns, but no column headings in iris.data. Let's create some using a list to define attributes:
headings = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)", "Class of Iris"]

# Read in iris.data and convert into dataframe with headings:
irisData = pd.read_csv(r'iris.data', header=None, names=headings)

# Investigating iris.data again, now that there are column headings:
print("Dataset shape:\n", irisData.shape, "\n")  # Find out how many columns and rows
print("First 5 rows of dataset:\n", irisData.head(), "\n")  # Print the first 5 rows of the dataset
print("Dataset info:\n", irisData.info(), "\n")  # Summary of contents, memory usage, index size, data type of dataset
print("Column statistics:\n", irisData.describe(), "\n")  # Print column statistics
print("Types of Iris flowers:\n", irisData["Class of Iris"].unique())  # Prints unique names and data type in column
print("Number of Iris flower samples in each type: \n", irisData["Class of Iris"].value_counts(), "\n")  # Check samples

# Separating the columns of data into 1D vectors (vectors are dynamic with same data type):
sepalLength = irisData["Sepal Length (cm)"].values
sepalWidth = irisData["Sepal Width (cm)"].values
petalLength = irisData["Petal Length (cm)"].values
petalWidth = irisData["Petal Width (cm)"].values
classIris = irisData["Class of Iris"].values
# We won't use these here, but we have them anyway.

# We can generate a correlation heat map to find the variables with the least correlation to use for categorisation, but
# first, we need to assign all non-numerical data to a value because one column has object data types (as we found from
# our investigation), so let's just isolate and convert that column to ordinal numbers:
#classIris = irisData["Class of Iris"].astype("category").cat.codes
irisData["Class of Iris"] = LabelEncoder().fit_transform(irisData["Class of Iris"])  # Accesses classIris and changes it
# Checking that the changes have been applied:
print("First 5 rows of dataset:\n", irisData.head(), "\n")

# NB: Normalisation of the data values is really important when using statistical models. The following methods used
# incorporate normalisation scaling automatically as part of the algorithm.

# Now we can generate the correlation heat map:
plt.figure(figsize=(12, 10))
sns.heatmap(irisData.corr(), linewidths=.1, cmap="Blues", annot=True, annot_kws={"size": 7})
plt.yticks(rotation=0)
plt.savefig("irisDataCorrelation.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

# When classifying data, we look for beneficial linear relationships. The parameters associated sometimes have
# correlation coefficients with the highest magnitude or the greatest range. From the correlation heat map, we can see
# the parameters 'Sepal Width (cm)' vs. 'Class of Iris' and, 'Sepal Width (cm)' vs. 'Petal Length (cm)' share the same
# negative correlation coefficient of -0.42. 'Petal Length' contains the largest range of coefficients (from -0.42 to
# 0.95), so may contain the most complex relationships with the other parameters. Upon further inspection of these
# parameters:

# 'Sepal Width (cm)' vs. 'Class of Iris':
ax = sns.catplot(data=irisData,x="Class of Iris",y="Sepal Width (cm)",hue="Petal Length (cm)",height=5, aspect=.8)
ax.set_titles('Sepal Width vs. Class of Iris')
plt.xticks([0, 1, 2], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.savefig("SepalWidth_vs_ClassofIris.png", format='png', dpi=300, bbox_inches='tight')
plt.show()
# The class groups appear to be separate, reflecting the stronger linear relationship between class and petal length.

# 'Petal Length (cm)' vs. 'Class of Iris':
ax = sns.catplot(data=irisData,x="Class of Iris",y="Petal Length (cm)",hue="Sepal Width (cm)",height=5, aspect=.8)
ax.set_titles('Petal Length vs. Class of Iris')
plt.xticks([0, 1, 2], ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.savefig("PetalLength_vs_ClassofIris.png", format='png', dpi=300, bbox_inches='tight')
plt.show()
# The class groups appear to be separate, but the influence of petal length vs. class can be seen, as they are almost
# completely dependent on each other with a correlation value of 0.95

# 'Sepal Width (cm)' vs. 'Petal Length (cm)':
ax = sns.stripplot(data=irisData,x="Petal Length (cm)",y="Sepal Width (cm)",hue="Class of Iris")
plt.setp(ax.axes.get_xticklabels()[::2], visible=False)
ax.set_title('Sepal Width vs. Petal Length')
handles, labels = ax.get_legend_handles_labels()
labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
ax.legend(handles, labels)
plt.savefig("SepalWidth_vs_PetalLength.png", format='png', dpi=300, bbox_inches='tight')
plt.show()
# It's clear that there is some overlap in class classification between petal length 4.5cm to 5.1cm and sepal width
# 2.5cm to 3.1cm because sepal width vs. petal lengths and sepal width vs. class have the same negative coefficients,
# so the definition is in the positive coefficient between petal length and class, which also has a higher magnitude.
# This graph clearly shows a non-linear decision boundary.

# Through this investigation, it seems the parameter 'Petal Length' may be the most important for classification.

# Since we want to predict the class of Iris flower, we will focus on the "Class of Iris" column to set the x and y-axis
# and split data into seeded (to help with reproducibility) train and test subsets respectively:
x = irisData.drop(["Class of Iris"], axis=1)
y = irisData["Class of Iris"]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)  # 80/20 split
# NB: If we change test_size to 0.3 (meaning 80/20 split) or above, then the Decision Tree Classifier concludes that
# 'Petal Width' is the most important parameter, but the confusion matrix seems more confused.

# Finding out how many samples are in the train and test subsets:
print(len(x_train), len(x_test), len(y_train), len(y_test))

# To find the important parameter, we'll use the Decision Tree Classifier. This model doesn't need normalisation of test
# pairs
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

# Generating the Decision Tree Classifier graphic:
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
#Path may vary according to your Graphviz location
dot_data = export_graphviz(dt, out_file=None, feature_names=x.columns, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render(filename="decision_tree", format='png', view=True)
graph.view()

# Generating a graph showing the level of importance for each parameter. This process identifies the most influential
# parameter in the Decision Tree Classifier system. Result is 'Petal Length (cm)', confirming the above hypothesis.
features_list = x.columns.values
feature_importance = dt.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(12, 10))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color= "red")
plt.yticks(range(len(sorted_idx)), features_list[sorted_idx])
plt.xlabel('Importance')
plt.title('Feature importance')
plt.draw()
plt.savefig("FeatureImportance.png", format='png', dpi=500, bbox_inches='tight')
plt.show()
# However, run it again, and it might come up with 'Petal Width'. Changing the test subset size requires a couple of
# runs to converge this calculation to a stable result.

# Testing the accuracy of the Decision Tree Classifier:
y_pred_dt = dt.predict(x_test)
print(len(y_pred_dt))
print("Decision Tree Classifier report: \n\n", classification_report(y_test, y_pred_dt))
print("Test Accuracy: {}%".format(round(dt.score(x_test, y_test)*100, 2)))
# The calculation of this gives a false positive of 100%, and we'll see why with the generation of the Confusion Matrix
# below, and by counting the terminal leaf nodes in the Decision Tree Classifier.

# Generating a Confusion Matrix for the Decision Tree Classifier:
cm = confusion_matrix(y_test, y_pred_dt)
x_axis_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
y_axis_labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
f, ax = plt.subplots(figsize =(7,7))
sns.heatmap(cm, annot = True, linewidths=0.2, linecolor="black", fmt = ".0f", ax=ax, cmap="Blues", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
plt.xlabel("PREDICTED LABEL")
plt.ylabel("TRUE LABEL")
plt.title('Confusion Matrix for the Decision Tree Classifier')
plt.savefig("ConfusionMatrix.png", format='png', dpi=500, bbox_inches='tight')
plt.show()

# Predicting some of the X_test results and matching it with true (y_test) values using Decision Tree Classifier:
# 0: Iris-setosa
# 1: Iris-versicolor
# 2: Iris-virginica
x_testPredict = dt.predict(x_test)
print(x_testPredict[:10])
print(y_test[:10].values)
