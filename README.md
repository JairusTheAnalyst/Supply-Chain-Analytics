# Prediction of House Prices using Linear Regression.

## Table of Contents
- [Project overview](Project-overview)
  - [Goals and Objectives](Goals-and-Objectives)
  - [Data and Methods](Data-and-Methods)
  - [Tools](Tools)
- [Importing Relevant libraries](Importing-Relevant-libraries)
- [Data attributes](Data-attributes)
- [Data Cleaning](Data-Cleaning)
  - [visualizing missing values in the dataset](visualizing-missing-values-in-the-dataset)
  - [Dealing with missing values in the data](Dealing-with-missing-values-in-the-data)
  - [Outlier Detection and Treatment](Outlier-Detection-and-Treatment)
- [Feature Encoding](Feature-Encoding)
- [Exploratory Data Analysis (EDA)](Exploratory-Data-Analysis-(EDA))
- [Modeling Techniques](Modeling-Techniques)
  - [Train and Test Sets](Train-and-Test-Sets)
  - [Fitting Linear Regression](Fitting-Linear-Regression)
  - [Model Evaluation](Model-Evaluation)
- [Key Outcomes](Key-Outcomes)
- [Limitations](Limitations)
- [Future Scope](Future-Scope)
- [References](References)

## Project overview
This project focused on predicting house prices in Perth, Australia, leveraging Machine Learning algorithms to identify key price determinants and forecast values with accuracy. By analyzing a range of property features, neighborhood data, and market trends, this model aims to assist potential buyers, real estate agents, and investors with data-driven insights for better decision-making.
---

### Goals and Objectives
- Develop a predictive model that accurately estimates house prices based on various attributes.
- Explore feature engineering techniques to enhance the model’s ability to interpret Perth’s unique housing market.
- Evaluate and compare the performance of multiple machine learning algorithms to identify the most suitable one for this use case.
- Provide actionable insights for real estate stakeholders in Perth.

### Data and Methods
- The dataset used for this project can be downloaded from here [data](https://www.kaggle.com/datasets/syuzai/perth-house-prices?select=all_perth_310121.csv) 
- Data Collection: The dataset included property features (e.g., size, number of bedrooms and bathrooms, location), neighborhood information, and recent sales prices.

### Tools
- Excel- Data Cleaning [Download here](https://microsoft.com)
- Google Colab
- Python- Data analysis
- SQL
- Tableau- Creating a Report

## Importing Relevant libraries
```
import numpy as np
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import confusion_matrix
from scipy import stats
%matplotlib inline
sns.set()
```
### Data attributes
The Perth dataset which was used for this analysis has the following 19 attributes:
1. ADDRESS: Property address (string/object).
2. SUBURB: Suburb location of the property (string/object).
3. PRICE: Sale price of the property (integer).
4. BEDROOMS: Number of bedrooms in the property (integer).
5. BATHROOMS: Number of bathrooms in the property (integer).
6. GARAGE: Number of garage spaces (float, some missing values).
7. LAND_AREA: Total land area of the property in square meters (integer).
8. FLOOR_AREA: Total floor area in square meters (integer).
9. BUILD_YEAR: Year the property was built (float, some missing values).
10. CBD_DIST: Distance to the central business district (CBD) in kilometers (integer).
11. NEAREST_STN: Name of the nearest train station (string/object).
12. NEAREST_STN_DIST: Distance to the nearest train station in kilometers (integer).
13. DATE_SOLD: Date the property was sold (string/object, date format).
14. POSTCODE: Postal code of the property (integer).
15. LATITUDE: Latitude coordinate of the property (float).
16. LONGITUDE: Longitude coordinate of the property (float).
17. NEAREST_SCH: Name of the nearest school (string/object).
18. NEAREST_SCH_DIST: Distance to the nearest school in kilometers (float).
19. NEAREST_SCH_RANK: Ranking of the nearest school (float, many missing values).

## Data Cleaning
- Handling Missing Values:

  - Identification: First, checked for any missing values across all features.
Treatment:
For numerical features like Lot Size, Square Footage, or Year Built, used techniques like mean or median imputation based on the distribution of the data.
For categorical features such as Property Type or Neighborhood, applied mode imputation or "Unknown" category assignment, ensuring no essential information was lost.

```
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/0d8360ff-f956-44f6-9337-bd8e9c3decf4)

- Percentage missing
```
df.isnull().mean()*100
```
![image](https://github.com/user-attachments/assets/d327c849-c3ed-4704-bac1-12a436401142)

### visualizing missing values in the dataset
```
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull())
plt.show()
```
![image](https://github.com/user-attachments/assets/7bad74ee-8bf7-42ef-8a6c-4ee34980b0c1)

### Dealing with missing values in the data
We then Drop columns which have many null values and those that are not so important, therefore from our dataset we drop the NEAREST_SCH_RANK column.We don't require the column NEAREST_SCH_RANK since it has too many null values. We can also drop the ADDRESS columns. The missing values in Garage can be replaced by the median while 75% of the values will replace the BUILD_YEAR column

```
df= Allperthdataset.drop(['NEAREST_SCH_RANK','ADDRESS'],axis=1)
df['GARAGE'] = Allperthdataset['GARAGE'].fillna(Allperthdataset['GARAGE'].median())
df['BUILD_YEAR'] = Allperthdataset['BUILD_YEAR'].fillna(Allperthdataset['BUILD_YEAR'].quantile(0.75))

```
### Outlier Detection and Treatment
  - Univariate Outliers: Identified outliers in numerical features (e.g., extremely high/low Price or Square Footage) using z-scores or IQR (Interquartile Range).
  - Treatment: For outliers that could skew the model, either capped extreme values or removed them based on business rules.

- Checking for Outliers in our dataset
  ```
  sns.scatterplot(x='LAND_AREA',y='PRICE_Log',data=df)
  ```
- removing outliers in LAND_AREA variable
  
```
df = df[df.LAND_AREA < 200000]
sns.scatterplot(x=df.LAND_AREA, y=df.PRICE_Log)
```
-  removing outliers in Bedrooms variable
    
```
df = df[df.BEDROOMS <8]
sns.scatterplot(x=df.BEDROOMS, y=df.PRICE_Log)
```


### Feature Encoding
 - Categorical Variables: Converted categorical features (e.g., Property Type, Condition) into numerical formats using one-hot encoding or label encoding, depending on the feature and the algorithm’s requirements.
 - Replaced String variables with dummies
   
  ```
  for i in ['SUBURB', 'NEAREST_STN', 'NEAREST_SCH', 'POSTCODE']:
    dummies = pd.get_dummies(df[i])
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(i, axis=1)
  ```
### Data Transformation:
 - Checking for skewness and kurtosis in target variable Price

  ```
  sns.distplot(df['PRICE']);
print("Skewness: %f" % df['PRICE'].skew())
print("Kurtosis: %f" % df['PRICE'].kurt())
  ```
![image](https://github.com/user-attachments/assets/a5e4699f-1bce-4077-bc8e-bda2036974f1)

### Log transformation on skewed features like Price .

  ```
  df['PRICE_Log'] = np.log(df['PRICE'])
sns.distplot(df['PRICE_Log'])
print("Skewness: %f" % df['PRICE_Log'].skew())
print("Kurtosis: %f" % df['PRICE_Log'].kurt())
 ```
![image](https://github.com/user-attachments/assets/79e2ccb9-2cab-4a17-b470-cdf415873cb4)


   - Scaled numerical features using standardization or min-max scaling to ensure that all values were on a comparable scale for model training.


- Duplicate Removal:
 - Removed any duplicate entries to ensure each record was unique and relevant to the model.

## Exploratory Data Analysis (EDA)
We aimed to answer the following questions:
   1. How do the Age of a House affect its Price?
   2. What is the relationship between house price and Square Footage?
   3. Which numerical features are most correlated with Price?


- Descriptive Statistics
Descriptive statistics was computed for numeric variables in the dataset to determine the distribution.
```Python
df.describe().T
```
![image](https://github.com/user-attachments/assets/42e4f666-07d3-4d82-8af4-f0d455df01e4)

-  The count of bedrooms
```
plt.figure(figsize=(12,6))
sns.countplot(df.BEDROOMS)
```
![image](https://github.com/user-attachments/assets/70195f08-6783-43b7-9963-5721d16f60e7)

- The count of bathrooms
```
![image](https://github.com/user-attachments/assets/3b1acae2-07f8-4cd9-aa05-5c13df1c54b5)

plt.figure(figsize=(12,6))
sns.countplot(df.BATHROOMS)
```
- Relationship between House Age and Price
```
plt.figure(figsize=(15,4))
fig = sns.lineplot(x=df['HOUSE_AGE'], y=df['PRICE'])
```
![image](https://github.com/user-attachments/assets/94224200-e972-41b1-a2b1-7c5e0a49c6aa)

- correlation heatmap
```
sns.heatmap(df.corr())
#correlations sorting
#top correlated variables
df.corr()['PRICE_Log'].sort_values(ascending=False)
```
![image](https://github.com/user-attachments/assets/bd7618ce-1747-40ba-8c01-71211a42f238)

## Modeling Techniques
- Machine Learning algorithms such as Linear Regression was used for predictive analysis.

###  Train and Test Sets
We then split the dataset into the test set and train set
```
from sklearn.model_selection import train_test_split


X = df.drop(["PRICE_Log","PRICE","DATE_SOLD"], axis=1)
y = df["PRICE_Log"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)

```
### Fitting Linear Regression

```
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
mlr = LinearRegression()
mlr.fit(x_train,y_train)
mlr_score = mlr.score(x_test,y_test)
pred_mlr = mlr.predict(x_test)
expl_mlr = explained_variance_score(pred_mlr,y_test)
```

### Model Evaluation 
- Performance will be evaluated using metrics like RMSE, MAE, and R² to assess predictive accuracy.
```
print("The intercept for the Multiple Linear regression: ", mlr.intercept_)

print("Linear Regression R^2 Score: ", mlr.score(x_train, y_train))
print("Linear Regression Test R^2 Score: ", mlr.score(x_test, y_test))
print("Mean Squared Error: ", mean_squared_error(pred_mlr, y_test))
print("Mean Absolute Error: ", metrics.mean_absolute_error(pred_mlr, y_test))
```
![image](https://github.com/user-attachments/assets/0f7cb837-e96c-4b93-b121-4a4e42fdde30)


## Key Outcomes
- Accurate House Price Predictions:

The model achieved a high R² score of 0.76 on the training data and 0.75 on the test data, indicating that it explains approximately 76% of the variance in house prices. This level of accuracy makes the model a reliable tool for estimating house prices.

- Minimal Overfitting: With similar R² values between the training and test datasets, the model demonstrates good generalization, meaning it is not overly complex and performs well on new data.
- Low Prediction Error:
  - The Mean Absolute Error (MAE) of 0.172 suggests that, on average, the model’s predictions are within approximately 17% of the actual prices, which is acceptable for real estate applications. The Mean Squared Error (MSE) of 0.062 reinforces this, showing that the average squared difference between predicted and actual prices is relatively low.
- Identification of Key Price Drivers:
  - Through feature analysis, the model highlights the most influential factors affecting house prices in Perth. These insights can help stakeholders understand what drives property values, providing a foundation for data-driven decision-making.
- Potential for Real Estate Stakeholders:
   - This model serves as a useful tool for real estate agents, property investors, and home buyers by providing accurate price estimations and insights into market trends. It can assist stakeholders in setting competitive prices and making informed purchasing decisions.

## Limitations
- The model uses historical data without dynamically adjusting for current market conditions, such as economic changes or seasonal trends. This limits the model’s applicability in volatile markets where prices may fluctuate due to interest rates, inflation, or housing supply and demand.
- The model does not consider external economic factors like employment rates, interest rates, or GDP growth, all of which can influence the housing market. Integrating these factors could make predictions more resilient to economic shifts.
- The model was trained specifically on data from Perth. Applying it to predict house prices in other regions would require additional retraining and fine-tuning, limiting its direct scalability to different markets.

## Future Scope
- Include macroeconomic variables like interest rates, inflation, and employment data to account for broader economic conditions impacting house prices. This could help the model adjust to changing market conditions and improve predictive accuracy.
- Use geospatial data and mapping techniques to provide a visual representation of price trends across different neighborhoods in Perth. Geospatial analysis could reveal clusters or "hot spots" of high or low prices, enabling users to make location-based decisions.
- Enable real-time predictions by incorporating live data feeds (e.g., recent sales, interest rates, or property listings). This would make the model practical for dynamic, on-the-fly predictions suitable for fast-paced real estate markets.

### References
[Kaggle Datasets](https://www.kaggle.com/datasets)
