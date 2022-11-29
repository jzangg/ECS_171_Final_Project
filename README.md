# ECS_171_Final_Project

## Preprocessing
We're deciding to preprocess the data with min-max normalization, using `MinMaxScaler` from `sklearn.preprocessing`. To determine if the data is normally distributed, we performed the Shapiro-Wilk test. We found that all columns of the dataset had a p-value below 0.05, showing that the data were not normally distributed. This is consistent with our preliminary analysis of the plots; there, we found that some values are skewed or in bimodal distributions. As the data were not normally distributed, we needed to perform normalization using `MinMaxScaler` instead of performing scalarization with tools like `StandardScaler`.
 
As the data did not include categorical labels, we did not need to perform any encoding. Our dataset had no NULL values because they were already replaced with zeros. To test whether or not we should apply a different imputing strategy to our dataset, we determined how significant the null values are. First, we found the amount of zeros found per feature. Zeros were only expected for the Pregnancy column, and thus any zeros found in other features (Insulin, BMI, Age, etc) were previous NULL data points. Since a significant amount of “NULL” values were found (~50% of the Insulin feature), we determined that this would negatively impact our model’s performance. We employed the median imputing strategy to replace the zeros with each feature’s median value (except for pregnancy).

See ECS171_Data_Exploration.ipynb for Data Exploration Milestone (Due November 20)

## Machine Learning Models
We built a variety of classification models on `Outcome` (whether the patient has diabetes) using the attributes from the preprocessed data. The preprocessed data was split into a training and testing set with the ratio of 80:20. The types of models consist of a Logistic Regression model, an Artificial Neural Network, and Support Vector Machines with a linear and RBF kernal. 

Our first model is a `sklearn` Logistic Regression model fit to our training data. Typically, logistic regression models can be made into a classifier using a threshold. However, since `Outcome` only has two possible values, 1 or 0 indicating whether the patient has diabetes, the predicted values are the same with or without a threshold value. The model has an 77% accurancy on the training set and an 76% accuracy on the test set, which is not optimal for classification. The model's performance on predicting the training and testing sets had an log-loss error of 0.49 and 0.46 respectively. The model's low accuracy and high loss indicates that this model may be too simple to represent the Diabetes dataset, resulting in underfitting.

See Preprocessing_First_Model_building.ipynb for Preprocessing & First Model Building and Evaluation Milestone (Due November 28)
