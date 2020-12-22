# Predicting the Air Quality Index 

## Introduction

Air Pollution is described as the presence of harmful pollutants in the air. Pollutant can be solid particles, liquid aerosols or gases. Major contributors to air pollution are vehicle emissions, industrial exhaust, wildfires etc.
One of the key metrics to measure the degree of air pollution is Air Quality Index (AQI). It is calculated on a scale of 0 – 500 and above, with a higher AQI indicating higher level of air pollution.

Data set for this project has been sourced from the United States Environment Protection Agency website.
Apart from the dependent variable AQI, we also have following features:
Ozone (O3); PM 2.5; Sulphur Dioxide; Carbon Monoxide; PM 10; Nitrogen Dioxide
![Alt text](/assets/plots/Introduction.png?raw=true "")

## Description of Dataset
### Plot of the dependent variable versus time
![Alt text](/assets/plots/Description_1.png?raw=true "")
Above is the plot of dependent variable Air Quality Index (AQI) vs Time for the date range 24th July 2000 to 31st December 2019. We have highlighted the buckets for AQI to understand the severity of air pollution in this region.

### ACF of the dependent variable
![Alt text](/assets/plots/ACF_AQI.png?raw=true "")
Above is the ACF plot for 250 lags. We can observe that AQI values are highly correlated to past values, indicating the presence of non-stationarity.

### Correlation Matrix with seaborn heatmap and Pearson’s correlation coefficient
![Alt text](/assets/plots/CorrPlot.png?raw=true "")
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
![Alt text](/assets/plots/Description_2.png?raw=true "")
All further training and data diagnostics will be performed using the train data set, and test data set will only be used to evaluate the model forecasts.

## Stationarity
![Alt text](/assets/plots/Stationarity_1.png?raw=true "")
Even though the Augmented Dickey Fuller Test and the Rolling Stats plot give an indication that the data set is stationary, but the ACF plot does provide evidence of non-stationarity. ADF Test results are subject to Type I and Type II errors, and hence, cannot be relied upon for this case.

![Alt text](/assets/plots/Stationarity_2.png?raw=true "")
To make our data set stationary, we applied 1st Order Differencing, and diagnosed it again for stationarity. We can observe that the differenced data set is situated around a mean of 0. ACF Plot after 1st Order Differencing does give evidence that the data set has now become stationary. 
![Alt text](/assets/plots/Stationarity_3.png?raw=true "")
Augmented Dickey Fuller Test and the Rolling Stats plot for the differenced data set also indicate that it is stationary. Rolling mean and variance become static at values of 0 and ~1000 respectively. Hence, we can now move ahead with our model building process.

## Time Series Decomposition
![Alt text](/assets/plots/STL_AQI.png?raw=true "")    
We now decompose our data to understand the trend and seasonality components individually. Strength of Trend is 0.845, and Strength of Seasonality is 0.398. Hence, we can conclude that our data set has strong trend and moderate seasonality.  
Since, we know our data set has high trend and moderate seasonal components, its good to comprehend whether they are additive or multiplicative.
![Alt text](/assets/plots/STL_2.png?raw=true "")
For Additive Time Series Decomposition of the AQI, we can notice high variance in the residuals here ranging from -100 to + 100.  
For Multiplicative Time Series Decomposition of the AQI, we can notice low variance in the residuals here.  
Based on the variance in the residuals of both types of decomposition, we can conclude that Multiplicative Decomposition is most suitable for our data set.
![Alt text](/assets/plots/STL_3.png?raw=true "")

## Base Models
### Average Model
![Alt text](/assets/plots/AVG.png?raw=true "") 

### Naive Model
![Alt text](/assets/plots/Naive.png?raw=true "")   

### Drift Model
![Alt text](/assets/plots/Drift.png?raw=true "")

### Simple Exponential Smoothening Model
![Alt text](/assets/plots/SES.png?raw=true "")

## Holt-Winters Model
![Alt text](/assets/plots/HW.png?raw=true "")

## Feature Selection
We performed feature selection based on the following attributes of the predictive accuracy metrics:
• Adjusted R Square: Higher the better   
• AIC Value: Lower the better    
• BIC Value: Lower the better    



