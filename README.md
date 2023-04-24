# Breast Cancer Diagnosis using SVM
This code uses Support Vector Machine (SVM) to diagnose breast cancer based on the given dataset. The dataset contains information on the mean radius, mean texture, mean perimeter, mean area, and mean smoothness of the breast mass. The dataset also includes a binary variable diagnosis that indicates if the mass is malignant (denoted by 1) or benign (denoted by 0).

# Requirements
pandas
numpy
matplotlib
seaborn
sklearn

# Installation
Install the required libraries using pip install pandas numpy matplotlib seaborn sklearn

# Dataset
The dataset is loaded using Google Colab and is saved in a CSV format. Before building the SVM model, the code checks if there are any missing values in the dataset and the data types of each feature.

# Model Building
The code uses SVM with a linear kernel to build the classification model. The dataset is split into training and testing data using the train_test_split function from sklearn.model_selection. The fit function is then used to train the SVM model with the training data, and the predict function is used to predict the class labels of the testing data. The accuracy of the model is calculated using the accuracy_score function from sklearn.metrics.

# Confusion Matrix
The confusion matrix is used to evaluate the performance of the model. The confusion_matrix function from sklearn.metrics is used to calculate the number of true positives, true negatives, false positives, and false negatives. The heatmap function from seaborn is used to create a graphical representation of the confusion matrix.
