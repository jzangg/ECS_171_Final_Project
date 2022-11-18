# ECS_171_Final_Project

## Preprocessing
We're deciding to preprocess the data with min-max normalization, using `MinMaxScaler` from `sklearn.preprocessing`. To determine if the data is normally distributed, we performed the Shapiro-Wilk test. We found that all columns of the dataset had a p-value below 0.05, showing that the data were not normally distributed. This is consistent with our preliminary analysis of the plots; there, we found that some values are skewed or in bimodal distributions. As the data were not normally distributed, we needed to perform normalization using `MinMaxScaler` instead of performing scalarization with tools like `StandardScaler`.
 
As the data did not include categorical labels, we did not need to perform any encoding. Our dataset had no NULL values because they were already replaced with zeros. To test whether or not we should apply a different imputing strategy to our dataset, we determined how significant the null values are. First, we found the amount of zeros found per feature. Zeros were only expected for the Pregnancy column, and thus any zeros found in other features (Insulin, BMI, Age, etc) were previous NULL data points. Since a significant amount of “NULL” values were found (~50% of the Insulin feature), we determined that this would negatively impact our model’s performance. We employed the median imputing strategy to replace the zeros with each feature’s median value (except for pregnancy).
