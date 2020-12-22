# Predicting the Air Quality Index 

## Introduction

Air Pollution is described as the presence of harmful pollutants in the air. Pollutant can be solid particles, liquid aerosols or gases. Major contributors to air pollution are vehicle emissions, industrial exhaust, wildfires etc.
One of the key metrics to measure the degree of air pollution is Air Quality Index (AQI). It is calculated on a scale of 0 – 500 and above, with a higher AQI indicating higher level of air pollution.

Data set for this project has been sourced from the United States Environment Protection Agency website.
Apart from the dependent variable AQI, we also have following features:
Ozone (O3); PM 2.5; Sulphur Dioxide; Carbon Monoxide; PM 10; Nitrogen Dioxide
<img src="/assets/plots/Introduction.png" width="100%" height="100%"/>

## Description of Dataset
### Plot of the dependent variable versus time
<img src="/assets/plots/Description_1.png" width="100%" height="100%"/>   
Above is the plot of dependent variable Air Quality Index (AQI) vs Time for the date range 24th July 2000 to 31st December 2019. We have highlighted the buckets for AQI to understand the severity of air pollution in this region.

### ACF of the dependent variable
<img src="/assets/plots/ACF_AQI.png" width="100%" height="100%"/>    
Above is the ACF plot for 250 lags. We can observe that AQI values are highly correlated to past values, indicating the presence of non-stationarity.

### Correlation Matrix with seaborn heatmap and Pearson’s correlation coefficient
<img src="/assets/plots/CorrPlot.png" width="100%" height="100%"/>     
Following are the observations on AQI from the correlation plot above:  
• All features except CO are positively correlated to AQI.  
• Very highly correlated to presence of Ozone.  
• Moderately correlated to presence of PM 2.5 and PM 10.  
• Presence of NO2, SO2 and CO has very low correlation to AQI.  

### Preprocessing procedures
Following pre-processing steps were taken to normalize the dataset:  
• Outlier values of AQI for certain dates such as 21st October 2007 and 26th March 2014 were 1108 and 537 respectively. Hence, to cater such anomalies, an upper
limit of 250 (covering 99.9% of observations) was set to treat outliers going above them.  
• Missing values for PM 10 and PM 2.5 were replaced with the median of the corresponding feature.  

### Split the dataset into train set (80%) and test set (20%)
To measure the performance of the different models, we split the data set into train (80%) and test (20%).
<img src="/assets/plots/Description_2.png" width="100%" height="100%"/>
All further training and data diagnostics will be performed using the train data set, and test data set will only be used to evaluate the model forecasts.

## Stationarity
<img src="/assets/plots/Stationarity_1.png" width="100%" height="100%"/>
Even though the Augmented Dickey Fuller Test and the Rolling Stats plot give an indication that the data set is stationary, but the ACF plot does provide evidence of non-stationarity. ADF Test results are subject to Type I and Type II errors, and hence, cannot be relied upon for this case.

<img src="/assets/plots/Stationarity_2.png" width="100%" height="100%"/>
To make our data set stationary, we applied 1st Order Differencing, and diagnosed it again for stationarity. We can observe that the differenced data set is situated around a mean of 0. ACF Plot after 1st Order Differencing does give evidence that the data set has now become stationary. 
<img src="/assets/plots/Stationarity_3.png" width="100%" height="100%"/>
Augmented Dickey Fuller Test and the Rolling Stats plot for the differenced data set also indicate that it is stationary. Rolling mean and variance become static at values of 0 and ~1000 respectively. Hence, we can now move ahead with our model building process.

## Time Series Decomposition
<img src="/assets/plots/STL_AQI.png" width="100%" height="100%"/>    
We now decompose our data to understand the trend and seasonality components individually. Strength of Trend is 0.845, and Strength of Seasonality is 0.398. Hence, we can conclude that our data set has strong trend and moderate seasonality.  
Since, we know our data set has high trend and moderate seasonal components, its good to comprehend whether they are additive or multiplicative.
<img src="/assets/plots/STL_2.png" width="100%" height="100%"/>
For Additive Time Series Decomposition of the AQI, we can notice high variance in the residuals here ranging from -100 to + 100.  
For Multiplicative Time Series Decomposition of the AQI, we can notice low variance in the residuals here.  
Based on the variance in the residuals of both types of decomposition, we can conclude that Multiplicative Decomposition is most suitable for our data set.
<img src="/assets/plots/STL_3.png" width="100%" height="100%"/>

## Base Models
### Average Model
<img src="/assets/plots/AVG.png" width="100%" height="100%"/> 

### Naive Model
<img src="/assets/plots/Naive.png" width="100%" height="100%"/>   

### Drift Model
<img src="/assets/plots/Drift.png" width="100%" height="100%"/>

### Simple Exponential Smoothening Model
<img src="/assets/plots/SES.png" width="100%" height="100%"/>

## Holt-Winters Model
<img src="/assets/plots/HW.png" width="100%" height="100%"/>

## Feature Selection
We performed feature selection based on the following attributes of the predictive accuracy metrics:    
• Adjusted R Square: Higher the better   
• AIC Value: Lower the better    
• BIC Value: Lower the better    
### Forward Selection
Approach towards Feature Selection:
<img src="/assets/plots/Selection_1.png" width="100%" height="100%"/>    
<img src="/assets/plots/Selection_2.png" width="100%" height="100%"/>     
### Backward Selection
Approach towards Feature Selection:
<img src="/assets/plots/Selection_3.png" width="100%" height="100%"/>    
<img src="/assets/plots/Selection_4.png" width="100%" height="100%"/>    
### Results
Following are the observations:    
o Irrespective of the approach i.e., forward selection or backward elimination, we get same features for individual performance metrics.     
o Adjusted R2 and AIC Value end with same list of features i.e., 'CO', 'Ozone', 'SO2', 'PM10' and 'PM25'.      
o BIC Value leads to a model with 4 features i.e., 'Ozone', 'PM25', 'PM10' and 'CO'.     
Therefore, we have decided to eliminate NO2 as it is not adding much value to the model in terms of any of the performance metric.     
The final set of features we have decided to build a Multiple Regression Model are 'CO', 'Ozone', 'SO2', 'PM10', 'PM25'.     

## Multiple Linear Regression
<img src="/assets/plots/Regression_1.png" width="100%" height="100%"/>    
Therefore, as per the T-Test Results, we can conclude that all of the selected features are statistically significant.
<img src="/assets/plots/Regression_2.png" width="35%" height="35%"/>    
Based on the Adjusted R2 value, we can say that the 91% of the variance in the value of “AQI” can be explained by this model.
<img src="/assets/plots/Regression_3.png" width="100%" height="100%"/>    

## ARMA Model
### GPAC & Order Estimation
<img src="/assets/plots/GPAC.png" width="100%" height="100%"/>    
As per our understanding of the GPAC table, we can list the following orders that can be potentially used for ARMA model configuration:    
(3,7); (6,6); (8,5); (9,9)

However, after the performing the Chi-Squared Test for Whiteness of Residuals, none of the above listed orders could pass the same. Hence, we had to look for potential orders through Brute Force methodology, where in we evaluate each combination of (na , nb) for the whiteness test.
Based on this approach, we were able to find out two combination of orders i.e., (4, 9) & (3, 9).    

### Parameter Estimation
Results for Order (na , nb): (4, 9)     
<img src="/assets/plots/Order1_1.png" width="100%" height="100%"/>    

Results for Order (na , nb): (3, 9)     
<img src="/assets/plots/Order2_1.png" width="100%" height="100%"/>    

### Parameter Diagnostic Outcomes
#### Confidence Intervals
Confidence intervals are calculated for the estimated parameters to check if they are statistically important. If zero includes inside the confidence interval, this means that the corresponding parameter is not important.     
Order (4, 9): For b3 , b8 and b9 we have zero included in the confidence interval, hence they are not statistically important.     
Order (3, 9): For a3 , b7 and b8 we have zero included in the confidence interval, hence they are not statistically important.     

#### Zero/Pole Cancellation
<img src="/assets/plots/Order12_2.png" width="100%" height="100%"/>    
After zero/pole cancellation, the ARMA (4, 9) got converted to ARMA (3, 9) due to reduced order. Hence, we will be going forward with the ARMA (3, 9) model.

### Chi Squared Test for Whiteness of Residuals
Q-value is calculated for the residuals and chi-squared whiteness test is performed to check if the residuals were white.    
<img src="/assets/plots/Order12_3.png" width="50%" height="50%"/>    

### Other Performance Measures
<img src="/assets/plots/Order12_4.png" width="50%" height="50%"/>    

## ARMA Results
<img src="/assets/plots/ARMA_1.png" width="100%" height="100%"/>    

<img src="/assets/plots/ARMA.png" width="100%" height="100%"/>    


## Final Model Selection
<img src="/assets/plots/AllTable.png" width="100%" height="100%"/>    
o Regression, Holt Winter, Naive and Drift Models have mean ~0, which means they are unbiased estimator.    
o ARMA model has the lowest Q-Value, which means they have been able to extract the most information from the data.    
o Regression model has the lowest Residual MSE and Variance. Also, the ratio of Variance of Forecast Errors and Residuals is very close to 0. Which means it is not overfitting.    

## Conclusion
o 1st Order differencing was done to make the raw data stationary.    
o Time Series decomposition showed that data has strong trend and moderate seasonality.     
o Feature selection process showed the most important attributes for AQI are 'CO', 'Ozone', 'SO2', 'PM10', 'PM25'.     
o ARMA Model (3,9) passed the diagnostic test but did not capture the seasonal attributes.     
o Best performing model is Regression model and Holt Winters Model. o Next steps to apply SARIMA in order to incorporate seasonality.     
