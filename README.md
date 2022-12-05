# ECS_171_Final_Project

## Milestones
|Due Date| Milestone |Notebook|
|---|---|---|
| November 20 | Data Exploration  | [ECS171_Data_Exploration.ipynb](/Milestones/ECS171_Data_Exploration.ipynb)   |
| November 28 | Preprocessing & First Model Building and Evaluation  | [Preprocessing_First_Model_building.ipynb](/Milestones/Preprocessing_First_Model_building.ipynb)  |
|  December 5 | Final Paper | [ECS171_Final_Project.ipynb](ECS171_Final_Project.ipynb) |

## Introduction
**Diabetes** is a metabolic disease where the body is unable to produce enough insulin or use the insulin effectively, causing blood sugar levels to rise. A large portion of the world’s population is at risk of or is affected by this disease, which can lead to problems such as heart and kidney damage or strokes. 

For our project, we wanted to predict if an individual has diabetes based on a variety of factors in order to diagnose the problem as early as possible. Originally part of a larger database from National Institute of Diabetes and Digestive and Kidney Diseases, our Kaggle dataset—[found locally in our repository](/diabetes.csv) or [on Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)—is a subset of observations where all patients are female, at least 21 years old, and of Pima Indian Heritage. It is labeled and consists of 8 additional features. The features are diagnostic measurements contributing to diabetes such as blood pressure, pregnancy frequency, skin thickness, insulin, BMI, age, and etc. The labels represent a diabetes test result where 1 is positive and 0 is negative.

Further, we will be using 3 different predictive models, **Logistic Regression, Support Vector Machines, and a Neural Networks**, and adjust the parameters of those models to see which would be the most accurate in identifying whether or not the patient has diabetes.

This project aims to create a good predictive model, which would provide another way to diagnose patients and enable them to seek treatment early. Additionally, we could understand which features are more likely to increase someone’s risk of having diabetes. 


## Figures
### Heatmap (Figure 1)

![Heatmap](/Figures/Heatmap.jpeg)

### Pairplot (Figure 2)

![Pairplot](/Figures/Pairplot.jpeg)

## Methods

### Data Exploration
In our initial data exploration of the Diabetes dataset, we employed a variety of techniques to visualize the data and understand the patterns and/or relationships between variables. First, we loaded the dataset into a Pandas dataframe. Then, we called the `describe()` method on the data frame to display a summary of important statistics (mean, median, etc.) for each feature. 

```
df = pd.read_csv("/content/diabetes.csv")
df.describe()
```
Next, we inquired whether there were any `NULL` values within our dataset with the following function call on the dataframe. 

```
df[df.isnull().any(axis=1)]
```
We determined that there were no `NULL` values across our dataset as the function call returned 0 data observations—likely because any preexisting `NULL` values were already replaced with zeros. 

Next, we utilized the dataframe to create a heatmap of the different features barring the outcome column (see Figure 1). 

```
df_no_class = df.drop('Outcome', axis=1
corr = df_no_class.corr()
sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, cmap= 'RdBu').set(title='Heatmap')```
```

The heatmap helped us visualize the data to identify broad patterns of correlation. The most significant correlations from the dataset were a 0.54 correlation between ‘Age’ and ‘Pregnancies’, a 0.44 correlation between ‘Insulin’ and ‘SkinThickness,’ and a 0.39 correlation between ‘BMI’ and ‘SkinThickness.’ 

In order to further visualize the data, we created a pairplot (see Figure 2).

```
sns.pairplot(df, hue="Outcome")
```
Upon analyzing the pairplot, we observed that some values were skewed or had bimodal distribution—the data is not normally distributed. Moreover, we also observed that on a general note, observations that are positive for diabetes tend to have numerically larger values in the feature vectors (‘Glucose’, ‘Blood Pressure’, ‘SkinThickness’) compared to observations that are not positive for diabetes. 


### Preprocessing
We decided to preprocess the data with min-max normalization, using `MinMaxScaler` from `sklearn.preprocessing`. To determine if the data is normally distributed, we performed the Shapiro-Wilk test. We found that all columns of the dataset had a p-value below 0.05, showing that the data were not all normally distributed. 

```
for col in diabetes_norm_df:
 p_val = round((shapiro(diabetes_norm_df[col]).pvalue), 2)
```

This is consistent with our preliminary analysis of the plots—there, we found that some values are skewed or in bimodal distributions. As the data were not normally distributed, we needed to perform normalization using `MinMaxScaler` instead of performing scalarization with tools like `StandardScaler`.

```
scaler = MinMaxScaler()
diabetes_norm_df = pd.DataFrame(scaler.fit_transform(df_no_class), columns=df_no_class.columns)
```

As the data did not include categorical labels, we did not need to perform any encoding. Our dataset had no `NULL` values because they were already replaced with zeros. To test whether or not we should apply a different imputing strategy to our dataset, we determined how significant the null values are.

First, we found the amount of zeros found per feature. Zeros were only expected for the Pregnancy column, and thus any zeros found in other features (Insulin, BMI, Age, etc) were previous `NULL` data points. 

```
for col in diabetes_norm_df:
 zero_occurences = len(diabetes_norm_df[diabetes_norm_df[col] == float(0)][col])
```
Since a significant amount of `NULL` values were found (~50% of the Insulin feature), we determined that this would negatively impact our model’s performance. 

We employed the median imputing strategy to replace the zeros with each feature’s median value (except pregnancy). 

```
for col in diabetes_cleaned:
 if (col == "Pregnancies"):
   continue
 
 col_no_zero = diabetes_cleaned[diabetes_cleaned[col] != float(0)][col]
 median = col_no_zero.median()
 diabetes_cleaned.loc[diabetes_cleaned[col] == float(0), col] = median
```


### Machine Learning Models
We built a variety of classification models on `Outcome` (whether the patient has diabetes) using the attributes from the preprocessed data. The preprocessed data was split into a training and testing set with the ratio of 80:20. 

```
X = diabetes_cleaned
Y = df["Outcome"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
```
The types of models consist of a Logistic Regression model, a Support Vector Machine with a RBF kernel, and an Artificial Neural Network.



### Logistic Regression
Our first model is a `sklearn` Logistic Regression model fit to our training data. Typically, logistic regression models can be made into a classifier using a threshold. However, since `Outcome` only has two possible values, 1 or 0 indicating whether the patient has diabetes, the predicted values are the same with or without a threshold value. 

```
log_reg = LogisticRegression(max_iter=200, fit_intercept=True)
log_reg_model = log_reg.fit(x_train, y_train)
```
We used the `predict_proba` function to calculate the log-loss value for predicting the training and testing performance. To find the `yhat` values, we used the `predict` function to classify the training and testing data and analyze the accuracy of the logistic model.

```
logloss_test = metrics.log_loss(y_test, log_reg_model.predict_proba(x_test))
logloss_train = metrics.log_loss(y_train, log_reg_model.predict_proba(x_train))
yhat_train_log_reg = log_reg_model.predict(x_train)
yhat_test_log_reg = log_reg_model.predict(x_test)

```


### SVM
Our second model is a `sklearn` Support Vector Machine. We created a SVM using sklearn’s `SVC` library and made a classifier that utilized a `rbf` kernel.

```
from sklearn.svm import SVC
clf_svm_rbf = SVC(kernel='rbf')
``` 
In order to fit our model, we first passed in the `X_train` and `y_train` values. Then, we predicted a `y_hat` by passing the `X_test` values into our trained model. To compare metrics between the training set and testing set, we printed out classification reports for both sets.

```
clf_svm_rbf.fit(x_train, y_train)
print("Train Metrics")
print(classification_report(y_train, clf_svm_rbf.predict(x_train)))
print("Test Metrics")
print(classification_report(y_test, clf_svm_rbf.predict(x_test)))
```


### Artificial Neural Network
Our final model is an Artificial Neural Network. We utilized a `keras.models` Sequential model to compile and fit our training data. 

```
nn_model = Sequential()
```
Our model has 3 layers—a `relu` activation layer with 15 units, a `tanh` activation layer with 4 units, and a `sigmoid` activation layer with 1 unit. 

```
nn_model.add(Dense(units = 15, activation = "relu", input_dim = data_dim))
nn_model.add(Dense(units = 4, activation = "tanh"))
nn_model.add(Dense(units = 1, activation = 'sigmoid'))
```
We compiled and fit our model using a `rmsprop` optimizer and a `binary_crossentropy` loss—we trained our model with a batch size of 50 and trained for 1000 epochs. 

```
nn_model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy')
nn_model.fit(x_train.astype('float'), y_train, batch_size = 50, epochs = 1000, verbose=0)
```

Finally, we utilized the model to predict our `yhat` and thresholded the output values.
```
yhat_nn_model = nn_model.predict(x_test.astype(float))
yhat_thres_nn_model = [1 if y>=0.5 else 0 for y in yhat_nn_model]
```


### ROC
Additionally, we generated ROC curves for each of our 3 models utilizing the `sklearn.metrics` library and plotted the results using the `matplotlib.pyplot` library. The plots were created to better understand the performance of each model. 

```
from sklearn import metrics
import matplotlib.pyplot as plt

fpr_train, tpr_train, _ = metrics.roc_curve(y_train, yhat_train_log_reg)
fpr_test, tpr_test, _ = metrics.roc_curve(y_test, yhat_test_log_reg)
auc_train = round(metrics.roc_auc_score(y_train, yhat_train_log_reg),3)
auc_test = round(metrics.roc_auc_score(y_test, yhat_test_log_reg), 3)

# plot ROC curve
plt.plot(fpr_train, tpr_train)
plt.plot(fpr_test, tpr_test)
plt.plot([0,1], 'g--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curves of Train vs Test Data')
plt.legend([f'ROC Train (AUC = {auc_train})' , f'ROC Test (AUC = {auc_test})', 'Random Classifier'])
plt.show()
```



## Results

### Logistic Regression

#### Accuracy

##### Training Set
|   | Precision  | Recall | F1-Score  | Support  |
|---|---|---|---|---|
| 0 | 0.78 | 0.90 | 0.84 | 401 |
| 1 | 0.74 | 0.52 | 0.61 | 213 |

##### Testing Set
|   | Precision  | Recall | F1-Score  | Support  |
|---|---|---|---|---|
| 0 | 0.77 | 0.90 | 0.83 | 99 |
| 1 | 0.75 | 0.70 | 0.60 | 55 |

The model had a 0.77 accuracy on the training set and a 0.76 accuracy on the testing set. 

#### ROC Curve

![Logistic ROC Curve](/Figures/Logistic_ROC.jpeg)

### SVM

#### Accuracy

##### Training Set
|   | Precision  | Recall | F1-Score  | Support  |
|---|---|---|---|---|
| 0 | 0.80 | 0.92 | 0.85 | 401 |
| 1 | 0.79 | 0.56 | 0.65 | 213 |

##### Testing Set
|   | Precision  | Recall | F1-Score  | Support  |
|---|---|---|---|---|
| 0 | 0.79 | 0.94 | 0.86 | 99 |
| 1 | 0.84 | 0.56 | 0.67 | 55 |

The model had a 0.79 accuracy on the training set and a 0.81 accuracy on the testing set.

#### ROC Curve

![SVM ROC Curve](/Figures/SVM_ROC.jpeg)

### Neural Network

#### Accuracy

##### Training Set
|   | Precision  | Recall | F1-Score  | Support  |
|---|---|---|---|---|
| 0 | 0.806 | 0.895 | 0.848 | 401 |
| 1 | 0.751 | 0.596 | 0.664 | 213 |

##### Testing Set
|   | Precision  | Recall | F1-Score  | Support  |
|---|---|---|---|---|
| 0 | 0.841 | 0.909 | 0.873 | 99 |
| 1 | 0.808 | 0.690 | 0.745 | 55 |

The model has a 0.79 accuracy on the training set and 0.83 accuracy on the test set.

#### Fitting Graph

![Neural Network Fitting Graph](/Figures/NN_Fitting_Curve.jpeg)

#### ROC Graph

![Neural Network ROC Curve](/Figures/NN_ROC.jpeg)

## Discussion

### Logistic Regression
For the first model, we used a Logistic Regression model, which can easily be a binary classifier by implementing a threshold value. Since the output is categorical, with only two possible values, a threshold is not required.

The Logistic Regression model classified the `Output` of the training set with 77% accuracy and the test set with 76% accuracy, which is not optimal for classification. The model’s performance on predicting the training and testing sets had a log-loss error of 0.49 and 0.46 respectively. The log-loss value indicates how close the prediction probability is to the corresponding binary classification, where a lower log-loss value indicates a better prediction. However, our model had log-loss values that suggested the model only had a 50% probability of predicting the correct value, which is reflected by the model’s lower accuracy. The model’s low accuracy and high loss indicates that this model may be too simple to represent the Diabetes dataset, resulting in underfitting. 

In the ROC curve for both the training and testing data, the Area Under the Curve (AUC) is around 0.711 and 0.704. Random classifiers have an AUC of 0.5, which suggests no discrimination and a predictive ability that is essentially just randomly diagnosing patients. Therefore, an AUC around 0.7 is considered acceptable because it has a slightly better predictive ability than random unbiased classification. However, the optimal value for AUC should be closer to 0.8 to 0.9, so the Logistic Regression model is not an optimal model for diagnosing patients for diabetes.

### SVM
We created an SVM model because we wanted to leverage an SVM’s ability to find an optimal classification line through margin maximization. Additionally, we wanted to use the kernels to help us better separate our data.

Our SVM model utilized an rbf kernel. This had an accuracy of 79% on the training set and an accuracy of 81% on the testing set. These metrics indicated that this model performed marginally better than our logistic regression model. This difference can likely be attributed to two factors: the SVM margin optimization and the behavior of the RBF kernel. As mentioned above, the SVM will maximize the margin between the classification line and the separate classes. This determines an optimal classification and reduces the chances of data points being misclassified. Additionally, the rbf kernel will transform the data, which may initially be difficult to separate into distinct classes. This transformation could allow this model to perform better than a logistic regression classifier.

In the ROC curve for both the training and testing data, the Area Under the Curve (AUC) is around 0.739 and 0.752, respectively. These metrics are acceptable but not optimal, as they perform better than a random classifier with an AUC of 0.5, but do not fall within the optimal range of 0.8 to 0.9. Using more training data or better adjusting the model parameters could potentially yield more accurate results in the future.

### Artificial Neural Network
Our dataset consists of 8 features, tabular data relating to various health metrics, which are not linearly correlated as shown in our initial exploration of the data. Hence, predicting whether diabetes is likely or not, a binary classification problem, requires a machine learning model capable of learning a nonlinear function. We determined that an artificial Neural Network is an ideal approach as it maps tabular input to a binary output by updating weights. Other types of Neural Networks were not considered due to the nature of our data. For instance, a CNN is typically used when the dataset consists of images and thus was not considered for this task. 

We utilized the Keras’s Sequential class with Dense layers to build our ANN. After experimenting with various parameters, our final model had 3 hidden layers. The first hidden layer consisted of 15 nodes and a relu activation function. When we increased or decreased the number of nodes, the accuracy decreased. 15 nodes appeared to capture enough information from the input features without the model being too complex or simple. The relu function was chosen since the data was already in a 0-1 distribution to due our MinMax preprocessing. Next, the second hidden layer contained 4 nodes and a tanh activation function. The tanh function is typically used for classification purposes and thus, when even experimenting with different functions, tanh yielded the highest accuracy. With another layer of weights being adjusted from the 15 node layer to 4 node layer, this second hidden layer captures the most important features relevant to the output and thus, improves the overall accuracy of the model. Lastly, the output layer had 1 node and a sigmoid activation function. Since, this is a binary classification problem a sigmoid function with 1 node was used to determine the probability of the true class.

The Neural Network model was the best performing at 0.83 accuracy. When comparing the ROC curves of all models, only the Neural Network curve had an AUC value in the optimal range (0.809). Thus, this indicates it has the best performance. We mentioned above that the logistic model showed underfitting. For the Neural Network fitting graph, the optimally fitted space was found when training for 1000 epochs as the testing set error was below the training set error. Thus, we trained our model for 1000 epochs to minimize underfitting and overfitting. More optimal fitting of the Neural Network could contribute to its better performance over the logistic regression model. In comparison with the SVM, its accuracy improved from the logistic model but was still worse than the Neural Network. Since our data was nonlinear, the structure of a Neural Network which updates its weights to map the input to an output value could have been simpler than the SVM’s approach of finding an optimal classification line by choosing a kernel. These differences between models could support the high accuracy of the Neural Network. 

## Conclusion
Overall, all of the models resulted in over 75% accuracy for the testing data, with the Neural Network being the best model with a 83% test data accuracy. For future work, we would want to see which model contains more false positives and which has more false negatives. This is because we would want to avoid false negatives when dealing with diseases such as diabetes, when identification is important. A way that we could reduce false negatives in our results would be to reduce the threshold value, and determine a value that would decrease false negatives while maintaining the accuracy that we have obtained with our existing models.

Our current dataset is limited to a set of observations where the patients are female, 21 years or older, and of Pima Indian heritage. While we can still make observations based off of the results of our current predictive models, other important features, such as ethnicity, sex, or region, may affect an individual’s risk of diabetes. Our current model does a good job for predicting if a female of Pima Indian heritage has diabetes, but this could lead to bias when trying to predict someone of a different background. In the future, we would like to expand the dataset to produce even more accurate predictive models that can be applied to a broader population. Gathering and testing with more data from different demographics would help reduce the bias in our current models. With this, we can compare models and see how different features impact people of different regions/genders/etc, and our models would be more beneficial in identifying diabetes in any patient.

Additionally, we could determine which features contribute more to diabetes and create newer models based on those features. Some features have some correlation to each other, and we believe that by reducing the number of features the model contains, we could create a more accurate model and acknowledge which features are the most relevant to predicting if an individual has diabetes.

We hope the supervised approaches we implemented made progress in diagnosing people of Pima Indian heritage, and any future work could aim to have a larger-scale impact on identifying diabetes on a patient of any background.

## Collaboration

### Chelsea Huffman
We all worked on the first proposal together. Then I helped on the logistic model and worked the ROC curves there. For the final submission, I worked on the introduction, conclusion, and helped come up with ways we can add to our project in the future. We also proofread our submission together.

### Emily Wen
We all worked together on the proposal. For the milestones, I helped with the preprocessing and worked on the logistic model. For the final submission, I compiled the results from the logistic regression model and wrote the discussion material for that model. I also helped with writing and organizing the final submission material.

### Jeffrey Zang
We worked together on the proposal and the data preprocessing. For the final submission, I trained and compared several SVM models, and chose one that had the best performance. I also gathered the resulting metrics, and contributed to writing the SVM sections in the methods, results, and discussion portions of the paper. As a team, we also proofread and formatted the final paper. Throughout the project, I helped coordinate the meetings and plan the work distribution.

### Kajal Patel
We all collaborated equally for the first milestone. For the second milestone, I assisted the group by programming parts of the preprocessing and logistic model. Lastly, for the final submission, I programmed the Neural Network, gathered the results, created figures, and analyzed the findings in the discussion section. Overall, I contributed my ideas and helped organize the team!

### Sanjana Aithal
We all worked on the first proposal together. For the milestones, I also assisted the group by programming the data preprocessing which included normalizing and cleaning the data. For the final submission, I wrote the Methods section and formatted the text for the final paper. We proofread our report together. Overall, we all worked together collaboratively to address challenges, and were there to always help each other throughout the quarter.

