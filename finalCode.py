import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import toolkit.time_series_toolkit as tst
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.graphics import tsaplots
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir("/Users/utkarshvirendranigam/Desktop/GitHub Projects/Forecast-Air-Quality-Index")

seed = 10

data = pd.read_csv("CAData.csv")
data["Date"] = pd.to_datetime(data["Date"])
data["AQI_Normalized"]=data["AQI"]
data.loc[data["AQI"] >250, 'AQI_Normalized'] = 250
data = data.drop('AQI', 1)
data = data.rename(columns={'AQI_Normalized': 'AQI'})
data=data.set_index("Date")
temp=data[data["PM10"]!="."]
temp=temp["PM10"].astype(int)
data.loc[data["PM10"] ==".", 'PM10'] = np.median(temp)
temp=data[data["PM25"]!="."]
temp=temp["PM25"].astype(int)
data.loc[data["PM25"] ==".", 'PM25'] = np.median(temp)
data["PM10"]=data["PM10"].astype(int)
data["PM25"]=data["PM25"].astype(int)
y=data["AQI"]



# Plot of the dependent variable versus time

plt.figure()
plt.plot(data.index, data["AQI"])
plt.title("Dependant Variable AQI vs Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.savefig("Plots/AQI_Time.png")
plt.show()


# ACF of the dependent variable

lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(data["AQI"], i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF[::-1]
y_ACF_plot = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Air Quality Index (New Delhi)")
plt.savefig("Plots/ACF_AQI.png")
plt.show()



# Correlation Matrix with seaborn heatmap and Pearson’s correlation coefficient

plt.figure()
plt.rcParams.update({'font.size': 6})
sns.heatmap(data.corr(), cmap='coolwarm_r', annot=True, fmt=".2f", vmin=-1, vmax=1)
plt.rcParams.update({'font.size': 8})
plt.title("Correlation Matrix")
plt.savefig("Plots/CorrPlot.png")
plt.show()


# Split the dataset into train set (80%) and test set (20%)

y = data["AQI"]
independantVariables=list(data.columns)
independantVariables.remove("AQI")
x=data[independantVariables]
# for colValue in independantVariables:
#     x[colValue].fillna(value=x[colValue].mean(),inplace=True)

x_train,x_test,y_train, y_test= train_test_split(x,y, test_size=0.2, shuffle=False)

hSteps=len(y_test)
trainingCount=len(y_train)


# Stationarity: Check for a need to make the dependent variable stationary.
# If the dependent variable is not stationary, you need to use the techniques
# discussed in class to make it stationary. You need to make sure that ADF-test
# is not passed with 95% confidence.

tst.ADF_Cal(y_train)

dataMean1=[]
dataVariance1=[]
for i in range(len(y_train)):
    dataMean1.append(y_train[:i].mean())
    dataVariance1.append(y_train[:i].var())

plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(1,len(dataMean1)+1),dataMean1,label="Rolling Mean")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.title("AQI: Rolling Mean")
plt.legend(loc="upper right")

plt.subplot(2,1,2)
plt.plot(np.arange(1,len(dataVariance1)+1),dataVariance1,label="Rolling Variance")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.title("AQI: Rolling Variance")
plt.legend(loc="upper right")
plt.savefig("Plots/RollingStats_AQI.png")
plt.show()

f, axs = plt.subplots(2,1, figsize=(10,5))
tsaplots.plot_acf(y_train.dropna(), alpha=0.05,ax=axs[0], lags=50)
# plt.show()
tsaplots.plot_pacf(y_train.dropna(), alpha=0.05,ax=axs[1], lags=50)
plt.savefig("Plots/PACF_ACF_AQI.png")
plt.show()


yStationary=y_train.diff()
yStationary=yStationary[1:]

plt.figure()
plt.plot(data.index[1:trainingCount], yStationary)
plt.title("AQI (1st Order Differencing) vs Time")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.savefig("Plots/AQI_1st_Time.png")
plt.show()


lagsCount = 50
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF_Stationary = []
for i in range(lagsCount + 1):
    y_ACF_Stationary.append(tst.acf_eqn(yStationary, i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF_Stationary[::-1]
y_ACF_Stationary_plot = np.concatenate((tempArray[:-1], y_ACF_Stationary), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_Stationary_plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: AQI (1st Order Differencing)")
plt.savefig("Plots/ACF_AQI_1st.png")
plt.show()


dataMean2=[]
dataVariance2=[]

for i in range(1,len(y)):
    dataMean2.append(yStationary[:i].mean())
    dataVariance2.append(yStationary[:i].var())
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(1,len(dataMean2)+1),dataMean2,label="Rolling Mean")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.title("AQI (1st Order Differencing): Rolling Mean")
plt.legend(loc="upper right")

plt.subplot(2,1,2)
plt.plot(np.arange(1,len(dataVariance2)+1),dataVariance2,label="Rolling Variance")
plt.ylabel("Magnitude")
plt.xlabel("Samples")
plt.title("AQI (1st Order Differencing): Rolling Variance")
plt.legend(loc="upper right")
plt.savefig("Plots/RollingStats_AQI_1st.png")
plt.show()

tst.ADF_Cal(yStationary)

f, axs = plt.subplots(2,1, figsize=(10,5))
tsaplots.plot_acf(yStationary.dropna(), alpha=0.05,ax=axs[0], lags=50)
# plt.show()
tsaplots.plot_pacf(yStationary.dropna(), alpha=0.05,ax=axs[1], lags=50)
plt.savefig("Plots/PACF_ACF_AQI_1st.png")
plt.show()


# Time series Decomposition: Approximate the trend and the seasonality and plot
# the detrended the seasonally adjusted data set. Find the out the strength of
# the trend and seasonality. Refer to the lecture notes for different type of
# time series decomposition techniques.


STL = STL(y_train)
res = STL.fit()
plt.rcParams.update({'font.size': 8})
fig=res.plot()
plt.savefig("Plots/STL_AQI.png")
plt.show()

T=res.trend
S=res.seasonal
R=res.resid


adjustedSeasonal = y_train-S

plt.figure()
plt.rcParams.update({'font.size': 8})
plt.plot(data.index[:trainingCount], y_train, color="orange", label="Original")
plt.plot(data.index[:trainingCount], np.array(adjustedSeasonal), color="blue",linestyle='--',dashes=(2, 5), label="Seasonal Adjusted")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("Original Data with Seasonal Adjusted")
plt.rcParams.update({'font.size': 8})
plt.legend(loc="upper left")
plt.savefig("Plots/SeasonalAdjusted_AQI.png")
plt.show()


FTrend = np.maximum(0,1-(np.var(np.array(R))/(np.var(np.array(T+R)))))
print("\nThe strength of trend for this data set is ",FTrend)

FSeasonal = np.maximum(0,1-(np.var(np.array(R))/(np.var(np.array(S+R)))))
print("\nThe strength of seasonality for this data set is ",FSeasonal)


# Holt-Winters method: Using the Holt-Winters method try to find the best
# fit using the train dataset and make a prediction using the test set.

HoltWinterSeasonalModel = ets.ExponentialSmoothing(y_train,trend="additive", seasonal="additive",seasonal_periods=365,freq=y_train.index.inferred_freq).fit()
predictedData = HoltWinterSeasonalModel.fittedvalues
forecastedData = HoltWinterSeasonalModel.forecast(steps=hSteps)

plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[trainingCount:], forecastedData, color="g", label="h-step forecast")
plt.xlabel("t")
plt.ylabel("y")
plt.rcParams.update({'font.size': 8})
plt.title("Data: Holt-Winter Seasonal Method & Forecast")
plt.legend(loc="upper left")
plt.savefig("Plots/Holts_Winter_AQI.png")
plt.show()


# Feature selection: You need to have a section in your report that explains how the
# feature selection was performed. Forward and backward stepwise regression is needed.
# You must explain that which feature(s) need to be eliminated and why.

pValueThreshold=0.05
totalFeatures=independantVariables.copy()
forwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}
previousValue=0
while len(forwardStepwiseList) < len(independantVariables):
    adjRSquaredDict={}
    aicDict={}
    bicDict={}
    for featureValue in totalFeatures:
        currentFeatureList=forwardStepwiseList+[featureValue]
        X = x_train[currentFeatureList]
        X = sm.add_constant(X)
        tempModel = sm.OLS(y_train, X).fit()
        if (tempModel.f_pvalue < pValueThreshold):
            checkPValueDict = dict(tempModel.pvalues)
            if (checkPValueDict[featureValue] < pValueThreshold):
                adjRSquaredDict[featureValue]=tempModel.rsquared_adj
                aicDict[featureValue] = tempModel.aic
                bicDict[featureValue] = tempModel.bic
    if (len(adjRSquaredDict.keys())>0):
        featureSelected=max(adjRSquaredDict, key=adjRSquaredDict.get)
        if(previousValue<=adjRSquaredDict[featureSelected]):
            forwardStepwiseList.append(featureSelected)
            totalFeatures.remove(featureSelected)
            featuresFinal[' '.join([str(elem) for elem in forwardStepwiseList])]=adjRSquaredDict[featureSelected]
            dataframeDict["Features"].append(tuple(forwardStepwiseList))
            dataframeDict["Adj R-Squared"].append(adjRSquaredDict[featureSelected])
            dataframeDict["AIC"].append(aicDict[featureSelected])
            dataframeDict["BIC"].append(bicDict[featureSelected])
            previousValue=adjRSquaredDict[featureSelected]
        else:
            break
    else:
        break

dfForward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Forward Selection Regression:\n",dfForward)

totalFeatures=independantVariables.copy()
backwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}

previousValue=0
while len(totalFeatures)>0:
    adjRSquaredDict={}
    # aicDict={}
    # bicDict={}
    X = x_train[totalFeatures]
    X = sm.add_constant(X)
    tempModel = sm.OLS(y_train, X).fit()
    checkPValueDict = dict(tempModel.pvalues)
    checkPValueDict.pop("const")
    if (np.all(np.array(list(checkPValueDict.values())) < pValueThreshold)):
        if (previousValue <= tempModel.rsquared_adj):
            previousValue=tempModel.rsquared_adj
            featuresFinal[' '.join([str(elem) for elem in totalFeatures])] = tempModel.rsquared_adj
            dataframeDict["Features"].append(tuple(totalFeatures))
            dataframeDict["Adj R-Squared"].append(tempModel.rsquared_adj)
            dataframeDict["AIC"].append(tempModel.aic)
            dataframeDict["BIC"].append(tempModel.bic)
            for featureValue in totalFeatures:
                tempList=totalFeatures.copy()
                tempList.remove(featureValue)
                X = x_train[totalFeatures]
                X = sm.add_constant(X)
                tempModel = sm.OLS(y_train, X).fit()
                if (tempModel.f_pvalue < pValueThreshold):
                    adjRSquaredDict[featureValue] = tempModel.rsquared_adj
            if (len(adjRSquaredDict.keys()) > 0):
                removeFeature = max(adjRSquaredDict, key=adjRSquaredDict.get)
            else:
                break
        else:
            break
    else:
        featuresFinal[' '.join([str(elem) for elem in totalFeatures])] = tempModel.rsquared_adj
        dataframeDict["Features"].append(tuple(totalFeatures))
        dataframeDict["Adj R-Squared"].append(tempModel.rsquared_adj)
        dataframeDict["AIC"].append(tempModel.aic)
        dataframeDict["BIC"].append(tempModel.bic)
        removeFeature=max(checkPValueDict, key=checkPValueDict.get)
    if (removeFeature not in totalFeatures):
        print(removeFeature)
    totalFeatures.remove(removeFeature)

dfBackward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Backward Elimination Regression:\n",dfBackward)
dfForward.to_csv("forward_Adjr2.csv")
dfBackward.to_csv("backward_Adjr2.csv")
print("\nFeatures Selected after Forward Selection Regression:\n",np.sort(dfForward.iloc[-1,0]))
print("\nFeatures Selected after Backward Elimination Regression:\n",np.sort(dfBackward.iloc[-1,0]))

print(50*"*")
print(50*"*","\n\n")

print(50*"*","\nFeature Selection based on AIC Values")
print(50*"*")

pValueThreshold=0.05
totalFeatures=independantVariables.copy()
forwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}
previousValue=10000000
while len(forwardStepwiseList) < len(independantVariables):
    adjRSquaredDict={}
    aicDict={}
    bicDict={}
    for featureValue in totalFeatures:
        currentFeatureList=forwardStepwiseList+[featureValue]
        X = x_train[currentFeatureList]
        X = sm.add_constant(X)
        tempModel = sm.OLS(y_train, X).fit()
        if (tempModel.f_pvalue < pValueThreshold):
            checkPValueDict = dict(tempModel.pvalues)
            if (checkPValueDict[featureValue] < pValueThreshold):
                adjRSquaredDict[featureValue]=tempModel.rsquared_adj
                aicDict[featureValue] = tempModel.aic
                bicDict[featureValue] = tempModel.bic
    if (len(aicDict.keys())>0):
        featureSelected=min(aicDict, key=aicDict.get)
        if(previousValue>=aicDict[featureSelected]):
            forwardStepwiseList.append(featureSelected)
            totalFeatures.remove(featureSelected)
            featuresFinal[' '.join([str(elem) for elem in forwardStepwiseList])]=aicDict[featureSelected]
            dataframeDict["Features"].append(tuple(forwardStepwiseList))
            dataframeDict["Adj R-Squared"].append(adjRSquaredDict[featureSelected])
            dataframeDict["AIC"].append(aicDict[featureSelected])
            dataframeDict["BIC"].append(bicDict[featureSelected])
            previousValue=aicDict[featureSelected]
        else:
            break
    else:
        break

dfForward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Forward Selection Regression:\n",dfForward)

totalFeatures=independantVariables.copy()
backwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}

previousValue=10000000
while len(totalFeatures)>0:
    aicDict={}
    X = x_train[totalFeatures]
    X = sm.add_constant(X)
    tempModel = sm.OLS(y_train, X).fit()
    checkPValueDict = dict(tempModel.pvalues)
    checkPValueDict.pop("const")
    if (np.all(np.array(list(checkPValueDict.values())) < pValueThreshold)):
        if (previousValue >= tempModel.aic):
            previousValue=tempModel.aic
            featuresFinal[' '.join([str(elem) for elem in totalFeatures])] = tempModel.aic
            dataframeDict["Features"].append(tuple(totalFeatures))
            dataframeDict["Adj R-Squared"].append(tempModel.rsquared_adj)
            dataframeDict["AIC"].append(tempModel.aic)
            dataframeDict["BIC"].append(tempModel.bic)
            for featureValue in totalFeatures:
                tempList=totalFeatures.copy()
                tempList.remove(featureValue)
                X = x_train[totalFeatures]
                X = sm.add_constant(X)
                tempModel = sm.OLS(y_train, X).fit()
                if (tempModel.f_pvalue < pValueThreshold):
                    aicDict[featureValue] = tempModel.aic
            if (len(aicDict.keys()) > 0):
                removeFeature = min(aicDict, key=aicDict.get)
            else:
                break
        else:
            break
    else:
        featuresFinal[' '.join([str(elem) for elem in totalFeatures])] = tempModel.aic
        dataframeDict["Features"].append(tuple(totalFeatures))
        dataframeDict["Adj R-Squared"].append(tempModel.rsquared_adj)
        dataframeDict["AIC"].append(tempModel.aic)
        dataframeDict["BIC"].append(tempModel.bic)
        removeFeature=max(checkPValueDict, key=checkPValueDict.get)

    totalFeatures.remove(removeFeature)

dfBackward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Backward Elimination Regression:\n",dfBackward)
dfForward.to_csv("forward_AIC.csv")
dfBackward.to_csv("backward_AIC.csv")
print("\nFeatures Selected after Forward Selection Regression:\n",np.sort(dfForward.iloc[-1,0]))
print("\nFeatures Selected after Backward Elimination Regression:\n",np.sort(dfBackward.iloc[-1,0]))

print(50*"*")
print(50*"*","\n\n")

print(50*"*","\nFeature Selection based on BIC Values")
print(50*"*")

pValueThreshold=0.05
totalFeatures=independantVariables.copy()
forwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}
previousValue=10000000
while len(forwardStepwiseList) < len(independantVariables):
    adjRSquaredDict={}
    aicDict={}
    bicDict={}
    for featureValue in totalFeatures:
        currentFeatureList=forwardStepwiseList+[featureValue]
        X = x_train[currentFeatureList]
        X = sm.add_constant(X)
        tempModel = sm.OLS(y_train, X).fit()
        if (tempModel.f_pvalue < pValueThreshold):
            checkPValueDict = dict(tempModel.pvalues)
            if (checkPValueDict[featureValue] < pValueThreshold):
                adjRSquaredDict[featureValue]=tempModel.rsquared_adj
                aicDict[featureValue] = tempModel.aic
                bicDict[featureValue] = tempModel.bic
    if (len(bicDict.keys())>0):
        featureSelected=min(bicDict, key=bicDict.get)
        if(previousValue>=bicDict[featureSelected]):
            forwardStepwiseList.append(featureSelected)
            totalFeatures.remove(featureSelected)
            featuresFinal[' '.join([str(elem) for elem in forwardStepwiseList])]=bicDict[featureSelected]
            dataframeDict["Features"].append(tuple(forwardStepwiseList))
            dataframeDict["Adj R-Squared"].append(adjRSquaredDict[featureSelected])
            dataframeDict["AIC"].append(aicDict[featureSelected])
            dataframeDict["BIC"].append(bicDict[featureSelected])
            previousValue=bicDict[featureSelected]
        else:
            break
    else:
        break

dfForward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Forward Selection Regression:\n",dfForward)

totalFeatures=independantVariables.copy()
backwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}
previousValue=10000000
while len(totalFeatures)>0:
    bicDict={}
    # aicDict={}
    # bicDict={}
    X = x_train[totalFeatures]
    X = sm.add_constant(X)
    tempModel = sm.OLS(y_train, X).fit()
    checkPValueDict = dict(tempModel.pvalues)
    checkPValueDict.pop("const")
    if (np.all(np.array(list(checkPValueDict.values())) < pValueThreshold)):
        if (previousValue >= tempModel.bic):
            previousValue=tempModel.bic
            featuresFinal[' '.join([str(elem) for elem in totalFeatures])] = tempModel.bic
            dataframeDict["Features"].append(tuple(totalFeatures))
            dataframeDict["Adj R-Squared"].append(tempModel.rsquared_adj)
            dataframeDict["AIC"].append(tempModel.aic)
            dataframeDict["BIC"].append(tempModel.bic)
            for featureValue in totalFeatures:
                tempList=totalFeatures.copy()
                tempList.remove(featureValue)
                X = x_train[totalFeatures]
                X = sm.add_constant(X)
                tempModel = sm.OLS(y_train, X).fit()
                if (tempModel.f_pvalue < pValueThreshold):
                    bicDict[featureValue] = tempModel.bic
            if (len(bicDict.keys()) > 0):
                removeFeature = min(bicDict, key=bicDict.get)
            else:
                break
        else:
            break
    else:
        featuresFinal[' '.join([str(elem) for elem in totalFeatures])] = tempModel.bic
        dataframeDict["Features"].append(tuple(totalFeatures))
        dataframeDict["Adj R-Squared"].append(tempModel.rsquared_adj)
        dataframeDict["AIC"].append(tempModel.aic)
        dataframeDict["BIC"].append(tempModel.bic)
        removeFeature=max(checkPValueDict, key=checkPValueDict.get)

    totalFeatures.remove(removeFeature)

dfBackward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Backward Elimination Regression:\n",dfBackward)
dfForward.to_csv("forward_BIC.csv")
dfBackward.to_csv("backward_BIC.csv")
print("\nFeatures Selected after Forward Selection Regression:\n",np.sort(dfForward.iloc[-1,0]))
print("\nFeatures Selected after Backward Elimination Regression:\n",np.sort(dfBackward.iloc[-1,0]))

print(50*"*")
print(50*"*","\n\n")


# Develop the multiple linear regression model that represent the dataset. Check the accuracy of the developed model.
# a. You need to include the complete regression analysis into your report. Perform one-step ahead prediction and compare the performance versus the test set.
# b. Hypothesis tests: F-test, t-test.
# c. AIC, BIC, RMSE, R-squared and Adjusted R-squared
# d. ACF of residuals.
# e. Q-value
# f. Variance and mean of the residuals.

print(50*"*","\n\t\t\t\tFinal Model")
print(50*"*")
featuresPrint=' '.join([str(elem) for elem in np.sort(dfForward.iloc[-1,0])])
print("\nFeatures Selected: ",featuresPrint)
X = x_train[list(dfForward.iloc[-1,0])]
X = sm.add_constant(X)
tempModel = sm.OLS(y_train, X).fit()
errorsPred=y_train-np.array(tempModel.fittedvalues).flatten()
SSE_Pred=np.sum(np.square(errorsPred))
print("\n\n",tempModel.summary())

print("\n\nF-Test Results:")
fTestPValue=tempModel.f_pvalue
print("\np-Value of F-Test: ",fTestPValue)
if(fTestPValue<=pValueThreshold):
        print("Since, p-value < = ",pValueThreshold,
              ", therefore we reject the null-hypothesis and\nconclude "
              "that our model provides a better fit than the intercept-only model.")
else:
    print("Since, p-value > ", pValueThreshold,
          ", therefore we fail to reject the null-hypothesis and\ncannot conclude that"
          " our model provides a better fit than the intercept-only model.")

print("\n\nT-Test Results:")
checkPValueDict = dict(tempModel.pvalues)
for valueDict in checkPValueDict.keys():
    print(50*"-")
    currPValue=checkPValueDict[valueDict]
    print("Feature: ",valueDict,"\tp-Value: ",currPValue)
    if (currPValue <= pValueThreshold):
        print("Since, p-value < = ", pValueThreshold,
              ", therefore we reject the null-hypothesis and\nconclude "
              "that the coefficient of this feature is significant and not equal to 0.")
    else:
        print("Since, p-value > ", pValueThreshold,
              ", therefore we fail to reject the null-hypothesis and\ncannot conclude that"
              "the coefficient of this feature is significant and hence is equal to 0.")


print("\nBelow are the model performance measures:")
print(50*"-")
print("\nBased on Fitted Values:\n")
print("AIC: ",tempModel.aic)
print("BIC: ",tempModel.bic)
print("RMSE: ",np.sqrt(SSE_Pred))
print("R-Squared: ",tempModel.rsquared)
print("Adjusted R-Squared: ",tempModel.rsquared_adj)

lagsCount = 25
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(errorsPred, i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF[::-1]
y_ACF = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Training Data Set Errors (Residuals)")
plt.savefig("Plots/ACF_Regression_Train_Errors.png")
plt.show()

qValue = tst.Q_cal(y_ACF[1:], len(y_train))
print("\nQ-Value - Residuals= ", round(qValue, 2))

print("\nMean of Residuals: ",np.mean(errorsPred))
print("Variance of Residuals: ",np.var(errorsPred))

X = x_test[list(dfForward.iloc[-1,0])]
X = sm.add_constant(X)
predictedResults=tempModel.predict(X)
plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[trainingCount:], predictedResults, color="g", label="h-step forecast")
plt.xlabel("t")
plt.ylabel("y")
plt.rcParams.update({'font.size': 8})
plt.title("Data: Regression Forecast")
plt.legend(loc="upper left")
plt.savefig("Plots/Regression_AQI.png")
plt.show()


errorsFore=y_test- predictedResults
SSE_Fore=np.sum(np.square(errorsFore))
rSquaredFore = np.square(tst.correlation_coefficent_cal(predictedResults,y_test))
T=len(y_test)
k=len(list(X.columns))-1
adjRSquaredFore=((1-(1-rSquaredFore))*(T-1))/(T-k-1)
AIC_Fore=(T*np.log((SSE_Fore/T)))+(2*(k+2))
BIC_Fore=(T*np.log((SSE_Fore/T)))+((k+2)*np.log(T))

print(50*"-")
print("\nBased on Predicted Values:\n")
print("AIC: ",AIC_Fore)
print("BIC: ",BIC_Fore)
print("RMSE: ",np.sqrt(SSE_Fore))
print("R-Squared: ",rSquaredFore)
print("Adjusted R-Squared: ",adjRSquaredFore)

lagsCount = 25
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(errorsFore, i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF[::-1]
y_ACF = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Test Data Set Errors (Residuals)")
plt.savefig("Plots/ACF_Regression_Test_Errors.png")
plt.show()

qValue = tst.Q_cal(y_ACF[1:], len(y_test))
print("\nQ-Value - Residuals= ", round(qValue, 2))

print("\nMean of Residuals: ",np.mean(errorsFore))
print("Variance of Residuals: ",np.var(errorsFore))


# ARMA (ARIMA or SARIMA) model order determination: Develop an ARMA (ARIMA or SARIMA) model that represent the dataset.
# a. Preliminary model development procedures and results. (ARMA model order determination). Pick at least two orders using GPAC table.
# b. Should include discussion of the autocorrelation function and the GPAC. Include a plot of the autocorrelation function and the GPAC table within this section).
# c. Include the GPAC table in your report and highlight the estimated order.

# lagsCount = 25
# x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
# y_ACF = []
# for i in range(lagsCount + 1):
#     y_ACF.append(tst.acf_eqn(y_train, i))
#
# # print("\nACF (20 Lags): ", np.round(y_ACF,4))
# tempArray = y_ACF[::-1]
# y_ACF = np.concatenate((tempArray[:-1], y_ACF), axis=None)
#

# plt.figure()
# plt.stem(x_ACF, y_ACF)
# plt.ylabel("Magnitude")
# plt.xlabel("Lags")
# plt.title("ACF: Train Data Set")
# plt.show()

titlePlot = "GPAC Table: AQI (1st Order Differencing)"
tst.GPAC_Cal(y_ACF_Stationary, 8, 8, titlePlot,figSize=(12,12),minSize=1)


# Estimate ARMA model parameters using the Levenberg Marquardt algorithm.
# Display the parameter estimates, the standard deviation of the parameter estimates
# and confidence intervals.


ar_order=3
ma_order=3

# ar_order=6
# ma_order=5
print("\nEstimated AR Order: ",ar_order)
print("\nEstimated MA Order: ",ma_order)


runLM=tst.Levenberg_Marquardt(y_train,ar_order,ma_order,iterations=150,rateMax=1000000)
results=runLM.calculateCoefficients()

model=sm.tsa.ARIMA(y_train,(ar_order,0,ma_order)).fit(disp=0)
# print(model.params)
# print(model.summary())
results=[model.params.values,model.cov_params().values]

model_name="ARMA("+str(ar_order)+", "+str(ma_order)+")"
ar_Array=[]
ma_Array=[]
for i in range(ar_order):
    currCoefficent=results[0][i]
    ar_Array.append(currCoefficent)
    currStdValue=np.sqrt(results[1][i,i])
    print("\na", i + 1, ":",np.round(currCoefficent,2),"    Standard Deviation: ",np.round(currStdValue,2))
    print("Confidence Intervals: ",np.round(currCoefficent-2*currStdValue,2)," < a", i + 1," < ",np.round(currCoefficent+2*currStdValue,2))

for i in range(ma_order):
    currCoefficent = results[0][i+ar_order]
    ma_Array.append(currCoefficent)
    currStdValue = np.sqrt(results[1][i+ar_order, i+ar_order])
    print("\nb", i + 1, ":",np.round(currCoefficent,2),"    Standard Deviation: ",np.round(currStdValue,2))
    print("Confidence Intervals: ", np.round(currCoefficent - 2 * currStdValue,2)," < b", i + 1," < ", np.round(currCoefficent + 2 * currStdValue,2))




armaForecast = tst.Basic_Forecasting_Methods(y_train)
y_pred = armaForecast.ARMAMethodPredict(ar_Array, ma_Array)
plt.figure()
# plt.plot(np.arange(1, 51), y_train[:50], color="blue", label="Train Data")
# plt.plot(np.arange(2,51), y_pred[1:50], color="g", label="1-Step Prediction")
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[:trainingCount], model.fitted, color="g", label="h-step forecast")
plt.xlabel("t")
plt.ylabel("y")
plt.rcParams.update({'font.size': 8})
plt.title("Model Prediction")
plt.legend()
plt.show()



predictionError=y_train[1:]-y_pred[1:]
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(predictionError, i))
tempArray = y_ACF[::-1]
y_ACF_plot = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF Plot of Residuals")
plt.show()

qValue = tst.Q_cal(y_ACF[1:], len(y_train))
print("\nQ-Value - Prediction Error = ", round(qValue, 2))

DOF = lagsCount - ar_order - ma_order
alpha = 0.01
chi_critical = chi2.ppf(1 - alpha, DOF)
print("For Degrees of Freedom = ",DOF, "& Alpha = ",alpha,", Chi-Critical = ",np.round(chi_critical,2))
if qValue < chi_critical:
    print("\nSince, Q-Value < Chi-Critical, the residual is white!")
else:
    print("\nSince, Q-Value > Chi-Critical, the residual is NOT white!")



print("\nMean of Residuals = ", round(predictionError.mean(), 2))
print("\nVariance of Residuals = ", round(predictionError.var(), 2))
msePredError=np.square(predictionError).mean(axis=0)
print("\nMean Squared Error of Residuals = ", round(msePredError, 2))

if (np.absolute(predictionError.mean())<0.05):
    print("\nSince mean is 0 (or <0.05), therefore it is an unbiased estimator")
else:
    print("\nSince mean is not equal to 0 (or <0.05), therefore it is not an unbiased estimator")

corrY_Y_Hat=tst.correlation_coefficent_cal(y_train[1:],y_pred[1:])
print("\nCorrelation coefficient between y(t) and yˆ (1): ",np.round(corrY_Y_Hat,2))
plt.figure()
plt.scatter(y_train[1:],y_pred[1:], color="b")
plt.xlabel("y(t)")
plt.ylabel("yˆ(1)")
plt.title("Scatter plot of y(t) and yˆ(1) with r = {}".format(corrY_Y_Hat))
plt.show()


corrY_Hat_Res=tst.correlation_coefficent_cal(y_pred[1:],predictionError)
print("\nCorrelation coefficient between yˆ (1) and residuals: ",np.round(corrY_Hat_Res,2))
plt.figure()
plt.scatter(y_pred[1:],predictionError, color="g")
plt.xlabel("yˆ(1)")
plt.ylabel("Residuals")
plt.title("Scatter plot of yˆ(1) and residuals with r = {}".format(corrY_Hat_Res))
plt.show()







def calculateModelResults(train,test,pred,fore,modelName="Not provided",lagsCount=5,returnResults=0):
    print("\n\n")
    print(50 * "*")
    print("\t\t",modelName," Model Results")
    print(50 * "*")
    print("\n")
    # lagsCount = 5
    predictionError=train[1:]-pred[1:]
    forecastError=test-fore
    x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
    y_ACF = []
    for i in range(lagsCount + 1):
        y_ACF.append(tst.acf_eqn(predictionError, i))
    tempArray = y_ACF[::-1]
    y_ACF_plot = np.concatenate((tempArray[:-1], y_ACF), axis=None)
    plt.figure()
    plt.stem(x_ACF, y_ACF_plot)
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.title("ACF Plot: "+ modelName+" Model")
    plt.savefig("Plots/ACFPlot_"+ modelName+"_Model.png")
    plt.show()
    qValue = tst.Q_cal(y_ACF[1:], len(predictionError))
    print("\nQ-Value - Prediction Error = ", round(qValue, 2))
    print("\nVariance:\nPrediction Error = ", round(predictionError.var(), 2))
    print("Forecasting Error = ", round(forecastError.var(), 2))
    trainCount=len(train)
    testCount=len(test)
    plt.figure()
    plt.plot(data.index[: trainCount], y_train, color="blue", label="Train Data")
    plt.plot(data.index[trainCount:], y_test, color="orange", label="Test Data")
    plt.plot(data.index[1:trainCount], pred[1:], color="g", label="1-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title(modelName+" Model Prediction")
    plt.legend(loc="upper left")
    plt.savefig("Plots/" + modelName + "_Model_Prediction.png")
    plt.show()

    plt.plot(data.index[: trainCount], y_train, color="blue", label="Train Data")
    plt.plot(data.index[trainCount:], y_test, color="orange", label="Test Data")
    plt.plot(data.index[trainCount:], fore, color="g", label="h-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title(modelName+" Model Forecast")
    plt.legend(loc="upper left")
    plt.savefig("Plots/" + modelName + "_Model_Forecast.png")
    plt.show()

    if returnResults==1:
        return {"predictionError":predictionError,"forecastError":forecastError,"qValue":qValue}

runTSTModels=tst.Basic_Forecasting_Methods(y_train)


# Average Method
model_name="Average"
prediction=runTSTModels.averageMethodPredict()
forecast=runTSTModels.averageMethodForecast(steps=hSteps)
calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25)


# Naive Method
model_name="Naive"
prediction=runTSTModels.naiveMethodPredict()
forecast=runTSTModels.naiveMethodForecast(steps=hSteps)
calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25)

# Drift Method
model_name="Drift"
prediction=runTSTModels.driftMethodPredict()
forecast=runTSTModels.driftMethodForecast(steps=hSteps)
calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25)


# SES Method
alpha=0.5
model_name="SES"
prediction=runTSTModels.SESMethodPredict(alpha=alpha)
forecast=runTSTModels.SESMethodForecast(steps=hSteps,alpha=alpha)
calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25)

