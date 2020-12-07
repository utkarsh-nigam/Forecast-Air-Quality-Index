import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import toolkit.time_series_toolkit as tst
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir("/Users/utkarshvirendranigam/Desktop/GitHub Projects/Forecast-Air-Quality-Index")

seed = 10

filterRows = np.arange(23368, 48192)
data = pd.read_csv("Data/Raw Data.csv")
data = data.iloc[filterRows, :]
data["Datetime"] = pd.to_datetime(data["Datetime"])
data=data.set_index("Datetime")
y=data["AQI"]

# Plot of the dependent variable versus time

plt.figure()
plt.plot(data["Datetime"], data["AQI"])
plt.title("Dependant Variable AQI vs Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()


# ACF of the dependent variable

lagsCount = 25
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(data["AQI"], i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF[::-1]
y_ACF = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Air Quality Index (New Delhi)")
plt.show()


# Correlation Matrix with seaborn heatmap and Pearson’s correlation coefficient

plt.figure()
plt.rcParams.update({'font.size': 6})
sns.heatmap(data.corr(), cmap='coolwarm_r', annot=True, fmt=".2f", vmin=-1, vmax=1)
plt.rcParams.update({'font.size': 8})
plt.title("Correlation Matrix")
plt.show()


# Split the dataset into train set (80%) and test set (20%)

y = data["AQI"]
y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
hSteps=len(y_test)
trainingCount=len(y_train)

# Stationarity: Check for a need to make the dependent variable stationary.
# If the dependent variable is not stationary, you need to use the techniques
# discussed in class to make it stationary. You need to make sure that ADF-test
# is not passed with 95% confidence.

tst.ADF_Cal(y)

# Time series Decomposition: Approximate the trend and the seasonality and plot
# the detrended the seasonally adjusted data set. Find the out the strength of
# the trend and seasonality. Refer to the lecture notes for different type of
# time series decomposition techniques.


STL = STL(y)
res = STL.fit()
plt.rcParams.update({'font.size': 8})
fig=res.plot()
plt.show()

T=res.trend
S=res.seasonal
R=res.resid


adjustedSeasonal = y-S

plt.figure()
plt.rcParams.update({'font.size': 8})
plt.plot(data.index, y, color="orange", label="Original")
plt.plot(data.index, np.array(adjustedSeasonal), color="blue",linestyle='--',dashes=(2, 5), label="Seasonal Adjusted")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("Original Data with Seasonal Adjusted")
plt.rcParams.update({'font.size': 8})
plt.legend(loc="upper left")
plt.show()


FTrend = np.maximum(0,1-(np.var(np.array(R))/(np.var(np.array(T+R)))))
print("The strength of trend for this data set is ",FTrend)

FSeasonal = np.maximum(0,1-(np.var(np.array(R))/(np.var(np.array(S+R)))))
print("\nThe strength of seasonality for this data set is ",FSeasonal)


# Holt-Winters method: Using the Holt-Winters method try to find the best
# fit using the train dataset and make a prediction using the test set.

HoltWinterSeasonalModel = ets.ExponentialSmoothing(y_train, trend="additive", damped=True, seasonal="additive").fit()
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
plt.show()


# Feature selection: You need to have a section in your report that explains how the
# feature selection was performed. Forward and backward stepwise regression is needed.
# You must explain that which feature(s) need to be eliminated and why.

independantVariables=list(data.columns)
independantVariables.remove("AQI")
for colValue in independantVariables:
    data[colValue].fillna(value=data[colValue].mean(),inplace=True)

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
        X = data[currentFeatureList]
        X = sm.add_constant(X)
        tempModel = sm.OLS(y, X).fit()
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
    X = data[totalFeatures]
    X = sm.add_constant(X)
    tempModel = sm.OLS(y, X).fit()
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
                X = data[totalFeatures]
                X = sm.add_constant(X)
                tempModel = sm.OLS(y, X).fit()
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
previousValue=10000
while len(forwardStepwiseList) < len(independantVariables):
    print("Here")
    adjRSquaredDict={}
    aicDict={}
    bicDict={}
    for featureValue in totalFeatures:
        print(featureValue)
        currentFeatureList=forwardStepwiseList+[featureValue]
        X = data[currentFeatureList]
        X = sm.add_constant(X)
        tempModel = sm.OLS(y, X).fit()
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
print(dataframeDict)
dfForward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Forward Selection Regression:\n",dfForward)

totalFeatures=independantVariables.copy()
backwardStepwiseList=[]
featuresFinal={}
dataframeDict={"Features":[],"Adj R-Squared":[],"AIC":[],"BIC":[]}

previousValue=10000
while len(totalFeatures)>0:
    aicDict={}
    X = data[totalFeatures]
    X = sm.add_constant(X)
    tempModel = sm.OLS(y, X).fit()
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
                X = data[totalFeatures]
                X = sm.add_constant(X)
                tempModel = sm.OLS(y, X).fit()
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
print(dataframeDict)
dfBackward=pd.DataFrame(dataframeDict,index=np.arange(1,len(featuresFinal.keys())+1))
pd.set_option('display.max_columns', None)
print("\n\nResults after Backward Elimination Regression:\n",dfBackward)

# print("\nFeatures Selected after Forward Selection Regression:\n",np.sort(dfForward.iloc[-1,0]))
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
previousValue=10000
while len(forwardStepwiseList) < len(independantVariables):
    adjRSquaredDict={}
    aicDict={}
    bicDict={}
    for featureValue in totalFeatures:
        currentFeatureList=forwardStepwiseList+[featureValue]
        X = data[currentFeatureList]
        X = sm.add_constant(X)
        tempModel = sm.OLS(y, X).fit()
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
previousValue=10000
while len(totalFeatures)>0:
    bicDict={}
    # aicDict={}
    # bicDict={}
    X = data[totalFeatures]
    X = sm.add_constant(X)
    tempModel = sm.OLS(y, X).fit()
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
                X = data[totalFeatures]
                X = sm.add_constant(X)
                tempModel = sm.OLS(y, X).fit()
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
dfForward.to_csv("forward.csv")
dfBackward.to_csv("backward.csv")
print("\nFeatures Selected after Forward Selection Regression:\n",np.sort(dfForward.iloc[-1,0]))
print("\nFeatures Selected after Backward Elimination Regression:\n",np.sort(dfBackward.iloc[-1,0]))

print(50*"*")
print(50*"*","\n\n")


print(50*"*","\n\t\t\t\tFinal Model")
print(50*"*")
featuresPrint=' '.join([str(elem) for elem in np.sort(dfBackward.iloc[-1,0])])
print("\nFeatures Selected: ",featuresPrint)
X = data[list(dfBackward.iloc[-1,0])]
X = sm.add_constant(X)
tempModel = sm.OLS(y, X).fit()
print("\n\n",tempModel.summary())



titlePlot = "GPAC Table: Air Quality Index (New Delhi)"
tst.GPAC_Cal(y_ACF, 8, 8, titlePlot)
