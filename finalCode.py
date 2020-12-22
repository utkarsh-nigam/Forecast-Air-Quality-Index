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
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")

seed = 10

data = pd.read_csv("CAData.csv")
data["Date"] = pd.to_datetime(data["Date"])
data=data.tail(7100)
print(data["Date"].min())
print(data["Date"].max())
data=data.set_index("Date")

plt.figure()
plt.plot(data.index, data["AQI"])
plt.title("Dependant Variable AQI vs Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()


data["AQI_Normalized"]=data["AQI"]
data.loc[data["AQI"] >250, 'AQI_Normalized'] = 250
data = data.drop('AQI', 1)
data = data.rename(columns={'AQI_Normalized': 'AQI'})

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
plt.title("AQI (Pre Processed) vs Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()


# ACF of the dependent variable

lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(data["AQI"], i))
tempArray = y_ACF[::-1]
y_ACF_plot = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_plot)
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
independantVariables=list(data.columns)
independantVariables.remove("AQI")
x=data[independantVariables]

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
##plt.savefig("Plots/RollingStats_AQI.png")
plt.show()

f, axs = plt.subplots(2,1, figsize=(10,5))
tsaplots.plot_acf(y_train.dropna(), alpha=0.05,ax=axs[0], lags=50)
tsaplots.plot_pacf(y_train.dropna(), alpha=0.05,ax=axs[1], lags=50)
plt.show()


yStationary=y_train.diff()
yStationary=yStationary[1:]

plt.figure()
plt.plot(data.index[1:trainingCount], yStationary)
plt.title("AQI (1st Order Differencing) vs Time")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.show()


lagsCount = 50
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF_Stationary = []
for i in range(lagsCount + 1):
    y_ACF_Stationary.append(tst.acf_eqn(yStationary, i))
tempArray = y_ACF_Stationary[::-1]
y_ACF_Stationary_plot = np.concatenate((tempArray[:-1], y_ACF_Stationary), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_Stationary_plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: AQI (1st Order Differencing)")
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
plt.show()

tst.ADF_Cal(yStationary)

f, axs = plt.subplots(2,1, figsize=(10,5))
tsaplots.plot_acf(yStationary.dropna(), alpha=0.05,ax=axs[0], lags=50)
tsaplots.plot_pacf(yStationary.dropna(), alpha=0.05,ax=axs[1], lags=50)
plt.show()


# Time series Decomposition: Approximate the trend and the seasonality and plot
# the detrended the seasonally adjusted data set. Find the out the strength of
# the trend and seasonality. Refer to the lecture notes for different type of
# time series decomposition techniques.

STL = STL(y_train)
res = STL.fit()
plt.rcParams.update({'font.size': 8})
fig=res.plot()
plt.show()

T=res.trend
S=res.seasonal
R=res.resid

adjustedSeasonal = y_train-S

plt.figure()
plt.rcParams.update({'font.size': 8})
plt.plot(data.index[:trainingCount], y_train, label="Original")
plt.plot(data.index[:trainingCount], np.array(adjustedSeasonal),linewidth=0.7,linestyle='--',dashes=(2, 5), label="Seasonal Adjusted")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("Original Data with Seasonal Adjusted")
plt.rcParams.update({'font.size': 8})
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.rcParams.update({'font.size': 8})
plt.plot(data.index[:trainingCount], y_train, label="Original")
plt.plot(data.index[:trainingCount], np.array(T),linewidth=0.7,linestyle='--',dashes=(2, 5), label="Trend")
plt.plot(data.index[:trainingCount], np.array(S),linewidth=0.7, label="Seasonal")
plt.plot(data.index[:trainingCount], np.array(R),linewidth=0.7,linestyle='--',dashes=(2, 5), label="Residual")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.title("Original Data & Components")
plt.rcParams.update({'font.size': 8})
plt.legend(loc="upper left")
plt.show()


decomposed_data = seasonal_decompose(y_train, "additive")
decomposed_data.plot()
plt.title("Seasonal Decomposition: Additive Residuals")
plt.show()

decomposed_data = seasonal_decompose(y_train, "multiplicative")
decomposed_data.plot()
plt.title("Seasonal Decomposition: Multiplicative Residuals")
plt.show()

FTrend = np.maximum(0,1-(np.var(np.array(R))/(np.var(np.array(T+R)))))
print("\nThe strength of trend for this data set is ",FTrend)

FSeasonal = np.maximum(0,1-(np.var(np.array(R))/(np.var(np.array(S+R)))))
print("\nThe strength of seasonality for this data set is ",FSeasonal)


# Holt-Winters method: Using the Holt-Winters method try to find the best
# fit using the train dataset and make a prediction using the test set.

HoltWinterSeasonalModel = ets.ExponentialSmoothing(y_train,trend="multiplicative",damped=True, seasonal="multiplicative",seasonal_periods=365,freq=y_train.index.inferred_freq).fit()
predictedData = HoltWinterSeasonalModel.fittedvalues
forecastedData = HoltWinterSeasonalModel.forecast(steps=hSteps)

plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[1: trainingCount], predictedData[1:], color="g", label="1-step prediction")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.rcParams.update({'font.size': 8})
plt.title("Data: Holt-Winter Seasonal Method & Prediction")
plt.legend(loc="upper left")
plt.show()

plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[trainingCount:], forecastedData, color="g", label="h-step forecast")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.rcParams.update({'font.size': 8})
plt.title("Data: Holt-Winter Seasonal Method & Forecast")
plt.legend(loc="upper left")
plt.show()

predictionError=y_train[1:]-predictedData[1:]
forecastingError=y_test-forecastedData
msePredError=np.square(predictionError).mean(axis=0)
mseForeError=np.square(forecastingError).mean(axis=0)

lagsCount=25
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF_PE=[]
for i in range(lagsCount+1):
    y_ACF_PE.append(tst.acf_eqn(predictionError,i))
qValuePE = tst.Q_cal(y_ACF_PE[1:], len(y_train))

tempArray = y_ACF_PE[::-1]
qValuePE_Plot = np.concatenate((tempArray[:-1], y_ACF_PE), axis=None)

plt.figure()
plt.stem(x_ACF, qValuePE_Plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF Plot: Holt's Winter Model")
plt.show()

y_ACF_FE = []
for i in range(lagsCount + 1):
    y_ACF_FE.append(tst.acf_eqn(forecastingError, i))
qValueFE = tst.Q_cal(y_ACF_FE[1:],len(y_test))

tableData={"Model":"Holt Winter","Q Value (PE)":qValuePE,"Q Value (FE)":qValueFE,"MSE (PE)":msePredError,"MSE (FE)":mseForeError,"RMSE (PE)":np.sqrt(msePredError),"RMSE (FE)":np.sqrt(mseForeError),"Var (PE)":predictionError.var(),"Var (FE)":forecastingError.var(),"Mean (PE)":np.mean(predictionError),"Mean (FE)":np.mean(forecastingError)}
compareTable=pd.DataFrame(tableData,index=[0])


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
pValueThreshold=0.05
featuresSelected=["CO", "Ozone", "SO2", "PM10", "PM25"]
featuresPrint=' '.join([str(elem) for elem in featuresSelected])
print("\nFeatures Selected: ",featuresPrint)
X = x_train[featuresSelected]
X = sm.add_constant(X)
tempModel = sm.OLS(y_train, X).fit()

predictionRegression=np.array(tempModel.fittedvalues).flatten()
predictionRegression[predictionRegression>250]=250
errorsPred=y_train-predictionRegression
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
y_ACF_Regression = []
for i in range(lagsCount + 1):
    y_ACF_Regression.append(tst.acf_eqn(errorsPred, i))

tempArray = y_ACF_Regression[::-1]
y_ACF_Regression_Plot = np.concatenate((tempArray[:-1], y_ACF_Regression), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_Regression_Plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Training Data Set Errors (Residuals)")
plt.show()

qValue = tst.Q_cal(y_ACF_Regression[1:], len(y_train))
print("\nQ-Value - Residuals= ", round(qValue, 2))

print("\nMean of Residuals: ",np.mean(errorsPred))
print("Variance of Residuals: ",np.var(errorsPred))

X = x_test[featuresSelected]
X = sm.add_constant(X)
predictedResults=tempModel.predict(X)
predictedResults[predictedResults>250]=250
plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[: trainingCount], predictionRegression, color="g", label="1-step prediction")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.rcParams.update({'font.size': 8})
plt.title("Data: Regression Prediction")
plt.legend(loc="upper left")
plt.show()


plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[trainingCount:], predictedResults, color="g", label="h-step forecast")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.rcParams.update({'font.size': 8})
plt.title("Data: Regression Forecast")
plt.legend(loc="upper left")
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
print("\nBased on Forecasted Values:\n")
print("AIC: ",AIC_Fore)
print("BIC: ",BIC_Fore)
print("RMSE: ",np.sqrt(SSE_Fore))
print("R-Squared: ",rSquaredFore)
print("Adjusted R-Squared: ",adjRSquaredFore)

lagsCount = 25
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF_Regression = []
for i in range(lagsCount + 1):
    y_ACF_Regression.append(tst.acf_eqn(errorsFore, i))

tempArray = y_ACF_Regression[::-1]
y_ACF_Regression_Plot = np.concatenate((tempArray[:-1], y_ACF_Regression), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF_Regression_Plot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Test Data Set Errors (Residuals)")
plt.show()

qValue = tst.Q_cal(y_ACF_Regression[1:], len(y_test))
print("\nQ-Value - Residuals= ", round(qValue, 2))

print("\nMean of Residuals: ",np.mean(errorsFore))
print("Variance of Residuals: ",np.var(errorsFore))

predictionError=y_train-np.array(tempModel.fittedvalues).flatten()
forecastingError=y_test- predictedResults
msePredError=np.square(predictionError).mean(axis=0)
mseForeError=np.square(forecastingError).mean(axis=0)

lagsCount=25
y_ACF_PE=[]
for i in range(lagsCount+1):
    y_ACF_PE.append(tst.acf_eqn(predictionError,i))
qValuePE = tst.Q_cal(y_ACF_PE[1:], len(y_train))

y_ACF_FE = []
for i in range(lagsCount + 1):
    y_ACF_FE.append(tst.acf_eqn(forecastingError, i))
qValueFE = tst.Q_cal(y_ACF_FE[1:],len(y_test))

tableData={"Model":"Regression","Q Value (PE)":qValuePE,"Q Value (FE)":qValueFE,"MSE (PE)":msePredError,"MSE (FE)":mseForeError,"RMSE (PE)":np.sqrt(msePredError),"RMSE (FE)":np.sqrt(mseForeError),"Var (PE)":predictionError.var(),"Var (FE)":forecastingError.var(),"Mean (PE)":np.mean(predictionError),"Mean (FE)":np.mean(forecastingError)}
compareTable=compareTable.append(tableData,ignore_index=True)



# ARMA (ARIMA or SARIMA) model order determination: Develop an ARMA (ARIMA or SARIMA) model that represent the dataset.
# a. Preliminary model development procedures and results. (ARMA model order determination). Pick at least two orders using GPAC table.
# b. Should include discussion of the autocorrelation function and the GPAC. Include a plot of the autocorrelation function and the GPAC table within this section).
# c. Include the GPAC table in your report and highlight the estimated order.

titlePlot = "GPAC Table: AQI (1st Order Differencing)"
tst.GPAC_Cal(y_ACF_Stationary, 12, 12, titlePlot,figSize=(12,12),minSize=1)

f, axs = plt.subplots(2,1, figsize=(10,5))
tsaplots.plot_acf(yStationary.dropna(), alpha=0.05,ax=axs[0], lags=50)
tsaplots.plot_pacf(yStationary.dropna(), alpha=0.05,ax=axs[1], lags=50)
plt.show()


# Estimate ARMA model parameters using the Levenberg Marquardt algorithm.
# Display the parameter estimates, the standard deviation of the parameter estimates
# and confidence intervals.

def transformFirstOrderPrediction(actualData, predictedData):
    finalPred = np.array([np.NaN])
    yTemp = actualData[1:-1] + predictedData[1:]
    finalPred = np.concatenate((finalPred, yTemp), axis=None)
    return finalPred


def transformFirstOrderForecast(actualData, forecastedData):
    finalFore = [actualData[-1]]
    for i in range(len(forecastedData)):
        finalFore.append(finalFore[-1] + forecastedData[i])
    return np.array(finalFore[1:])

arOrderArray=[3,4]
maOrderArray=[9,9]
for flag in range(len(arOrderArray)):
    ar_order=arOrderArray[flag]
    ma_order=maOrderArray[flag]
    model=sm.tsa.ARMA(yStationary,order=(ar_order,ma_order)).fit(trend="nc",disp=0)
    results = [model.params.values, model.cov_params().values]
    model_name = "ARMA(" + str(ar_order) + ", " + str(ma_order) + ")"
    print("\n")
    print(50 * "*")
    print("\t\t", model_name)
    print(50 * "*")
    ar_Array = []
    ma_Array = []
    print(model.summary())
    for i in range(ar_order):
        currCoefficent = -1*results[0][i]
        ar_Array.append(currCoefficent)
        currStdValue = np.sqrt(results[1][i, i])
        minLimit=currCoefficent - 2 * currStdValue
        maxLimit=currCoefficent + 2 * currStdValue
        print("\na", i + 1, ":", np.round(currCoefficent, 2), "; Standard Deviation: ", np.round(currStdValue, 2),
              "; Covariance: ", np.round(results[1][i, i], 2))
        print("Confidence Intervals: ", np.round(currCoefficent - 2 * currStdValue, 2), " < a", i + 1, " < ",
              np.round(currCoefficent + 2 * currStdValue, 2))

    for i in range(ma_order):
            currCoefficent = results[0][i + ar_order]
            ma_Array.append(currCoefficent)
            currStdValue = np.sqrt(results[1][i + ar_order, i + ar_order])
            minLimit = currCoefficent - 2 * currStdValue
            maxLimit = currCoefficent + 2 * currStdValue
            print("\nb", i + 1, ":", np.round(currCoefficent, 2), "; Standard Deviation: ", np.round(currStdValue, 2),
                  "; Covariance: ", np.round(results[1][i + ar_order, i + ar_order], 2))
            print("Confidence Intervals: ", np.round(currCoefficent - 2 * currStdValue, 2), " < b", i + 1, " < ",
                  np.round(currCoefficent + 2 * currStdValue, 2))

    print("\n\nRoots of Numerator/Zeros: ", np.roots([1] + ma_Array))
    print("\nRoots of Denominator/Poles: ", np.roots([1] + ar_Array))

    armaForecast = tst.Basic_Forecasting_Methods(yStationary)
    y_pred = armaForecast.ARMAMethodPredict(ar_Array, ma_Array)
    y_fore = armaForecast.ARMAMethodForecast(ar_Array, ma_Array, hSteps)

    print(50*"-")
    print("\tResults on Stationary Data")
    print(50 * "-")

    errorsPrediction = yStationary[1:] - y_pred[1:]

    acf_pred = []
    lagsCount = 25
    x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
    for i in range(lagsCount + 1):
        acf_pred.append(tst.acf_eqn(errorsPrediction, i))
    tempArray = acf_pred[::-1]
    acf_pred_plot = np.concatenate((tempArray[:-1], acf_pred), axis=None)
    plt.figure()
    plt.stem(x_ACF, acf_pred_plot)
    plt.ylabel("Magnitude")
    plt.xlabel("Lags")
    plt.title("ACF Plot: " + model_name + " Model")
    plt.show()

    acf_pred = []
    lagsCount=25
    for i in range(lagsCount + 1):
        acf_pred.append(tst.acf_eqn(errorsPrediction, i))
    qPred = tst.Q_cal(acf_pred[1:], len(yStationary))


    meanPredError = np.mean(errorsPrediction)
    varPredError = errorsPrediction.var()
    msePredError = np.square(errorsPrediction).mean(axis=0)
    rmsePredError = np.sqrt(msePredError)


    plt.figure()
    plt.plot(data.index[1: trainingCount], yStationary, color="blue", label="Train Data")
    plt.plot(data.index[2:trainingCount], y_pred[1:], color="g", label="1-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title("Data (1st_Order_Differencing)"+model_name+" Prediction")
    plt.legend()
    plt.show()


    predictions = transformFirstOrderPrediction(y_train, y_pred)
    forecasts = transformFirstOrderForecast(y_train, y_fore)

    plt.figure()
    plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
    plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
    plt.plot(data.index[2:trainingCount], predictions[1:], color="g", label="1-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title("Actual Data " + model_name + " Prediction")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
    plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
    plt.plot(data.index[trainingCount:], forecasts, color="g", label="h-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title("Actual Data " + model_name + " Forecast")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(data.index[trainingCount:trainingCount+50], y_test[:50], color="orange", label="Test Data")
    plt.plot(data.index[trainingCount:trainingCount+50], forecasts[:50], color="g", label="h-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title("Actual Data " + model_name + " Forecast (First 50)")
    plt.legend()
    plt.show()

    forecastingError = y_test - forecasts
    meanForeError = np.mean(forecastingError)
    varForeError = forecastingError.var()
    mseForeError = np.square(forecastingError).mean(axis=0)
    rmseForeError = np.sqrt(mseForeError)

    lagsCount = 25
    qValue = qPred.copy()

    y_ACF_FE = []
    for i in range(lagsCount + 1):
        y_ACF_FE.append(tst.acf_eqn(forecastingError, i))
    qValueFE = tst.Q_cal(y_ACF_FE[1:], len(y_test))

    print("\nResidual Stats:")
    print("Mean: ", round(meanPredError, 2))
    if (np.absolute(errorsPrediction.mean()) < 0.05):
        print("Since mean is 0 (or <0.05), therefore it is an unbiased estimator")
    else:
        print("Since mean is not equal to 0 (or <0.05), therefore it is not an unbiased estimator")
    print("\nMSE: ", round(msePredError, 2))
    print("RMSE: ", round(rmsePredError, 2))
    print("Variance: ", round(varPredError, 2))

    print("\nQ-Value: ", round(qValue, 2))
    DOF = lagsCount - ar_order - ma_order
    alpha = 0.01
    chi_critical = chi2.ppf(1 - alpha, DOF)
    print("For Degrees of Freedom = ", DOF, "& Alpha = ", alpha, ", Chi-Critical = ", np.round(chi_critical, 2))
    if qValue < chi_critical:
        print("Since, Q-Value < Chi-Critical, the residual is white!")
    else:
        print("Since, Q-Value > Chi-Critical, the residual is NOT white!")
    print("\n\nForecast Errors Stats:")
    print("Mean: ", round(meanForeError, 2))
    print("MSE: ", round(mseForeError, 2))
    print("RMSE: ", round(rmseForeError, 2))
    print("Variance: ", round(varForeError, 2))

    tableData={"Model":model_name,"Q Value (PE)":qValue,"Q Value (FE)":qValueFE,"MSE (PE)":msePredError,"MSE (FE)":mseForeError,"RMSE (PE)":rmsePredError,"RMSE (FE)":rmseForeError,"Var (PE)":varPredError,"Var (FE)":varForeError,"Mean (PE)":meanPredError,"Mean (FE)":meanForeError}
    compareTable = compareTable.append(tableData, ignore_index=True)


# Base-models: average, naïve, drift, simple and exponential smoothing

def calculateModelResults(train,test,pred,fore,modelName="Not provided",lagsCount=5,returnResults=0):
    print("\n\n")
    print(50 * "*")
    print("\t\t",modelName," Model Results")
    print(50 * "*")
    print("\n")

    predictionError=train[1:]-pred[1:]
    forecastError=test-fore

    meanPredError=np.mean(predictionError)
    meanForeError =np.mean(forecastError)

    varPredError = predictionError.var()
    varForeError = forecastError.var()

    msePredError = np.square(predictionError).mean(axis=0)
    mseForeError = np.square(forecastError).mean(axis=0)

    rmsePredError=np.sqrt(msePredError)
    rmseForeError = np.sqrt(mseForeError)


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
    plt.show()
    qValue = tst.Q_cal(y_ACF[1:], len(predictionError))
    y_ACF_FE = []
    for i in range(lagsCount + 1):
        y_ACF_FE.append(tst.acf_eqn(forecastingError, i))
    qValueFE = tst.Q_cal(y_ACF_FE[1:], len(y_test))
    print("\nResidual Stats:")
    print("Mean: ",round(meanPredError, 2))
    print("MSE: ", round(msePredError, 2))
    print("RMSE: ", round(rmsePredError, 2))
    print("Variance: ", round(varPredError, 2))

    print("\nQ-Value: ", round(qValue, 2))
    print("\n\nForecast Errors Stats:")
    print("Mean: ", round(meanForeError, 2))
    print("MSE: ", round(mseForeError, 2))
    print("RMSE: ", round(rmseForeError, 2))
    print("Variance: ", round(varForeError, 2))
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
    plt.show()

    plt.plot(data.index[: trainCount], y_train, color="blue", label="Train Data")
    plt.plot(data.index[trainCount:], y_test, color="orange", label="Test Data")
    plt.plot(data.index[trainCount:], fore, color="g", label="h-step prediction")
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.rcParams.update({'font.size': 8})
    plt.title(modelName+" Model Forecast")
    plt.legend(loc="upper left")
    plt.show()

    if returnResults==1:
        return {"predictionError":predictionError,"forecastError":forecastError,"qValue_PE":qValue,"Mean_PE":meanPredError,"MSE_PE":msePredError,"RMSE_PE":rmsePredError,"Variance_PE":varPredError,"qValue_FE":qValueFE,"Mean_FE":meanForeError,"MSE_FE":mseForeError,"RMSE_FE":rmseForeError,"Variance_FE":varForeError}

runTSTModels=tst.Basic_Forecasting_Methods(y_train)


# Average Method
model_name="Average"
prediction=runTSTModels.averageMethodPredict()
forecast=runTSTModels.averageMethodForecast(steps=hSteps)
results=calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25,returnResults=1)


tableData={"Model":"Average","Q Value (PE)":results["qValue_PE"],"Q Value (FE)":results["qValue_FE"],"MSE (PE)":results["MSE_PE"],"MSE (FE)":results["MSE_FE"],"RMSE (PE)":results["RMSE_PE"],"RMSE (FE)":results["RMSE_FE"],"Var (PE)":results["Variance_PE"],"Var (FE)":results["Variance_FE"],"Mean (PE)":results["Mean_PE"],"Mean (FE)":results["Mean_FE"]}
compareTable=compareTable.append(tableData,ignore_index=True)


# Naive Method
model_name="Naive"
prediction=runTSTModels.naiveMethodPredict()
forecast=runTSTModels.naiveMethodForecast(steps=hSteps)
results=calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25,returnResults=1)


tableData={"Model":"Naive","Q Value (PE)":results["qValue_PE"],"Q Value (FE)":results["qValue_FE"],"MSE (PE)":results["MSE_PE"],"MSE (FE)":results["MSE_FE"],"RMSE (PE)":results["RMSE_PE"],"RMSE (FE)":results["RMSE_FE"],"Var (PE)":results["Variance_PE"],"Var (FE)":results["Variance_FE"],"Mean (PE)":results["Mean_PE"],"Mean (FE)":results["Mean_FE"]}
compareTable=compareTable.append(tableData,ignore_index=True)


# Drift Method
model_name="Drift"
prediction=runTSTModels.driftMethodPredict()
forecast=runTSTModels.driftMethodForecast(steps=hSteps)
results=calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25,returnResults=1)

tableData={"Model":"Drift","Q Value (PE)":results["qValue_PE"],"Q Value (FE)":results["qValue_FE"],"MSE (PE)":results["MSE_PE"],"MSE (FE)":results["MSE_FE"],"RMSE (PE)":results["RMSE_PE"],"RMSE (FE)":results["RMSE_FE"],"Var (PE)":results["Variance_PE"],"Var (FE)":results["Variance_FE"],"Mean (PE)":results["Mean_PE"],"Mean (FE)":results["Mean_FE"]}
compareTable=compareTable.append(tableData,ignore_index=True)

# SES Method
alpha=0.5
model_name="SES"
prediction=runTSTModels.SESMethodPredict(alpha=alpha)
forecast=runTSTModels.SESMethodForecast(steps=hSteps,alpha=alpha)
results=calculateModelResults(np.array(y_train),np.array(y_test),np.array(prediction),np.array(forecast),model_name,lagsCount=25,returnResults=1)


tableData={"Model":"SES","Q Value (PE)":results["qValue_PE"],"Q Value (FE)":results["qValue_FE"],"MSE (PE)":results["MSE_PE"],"MSE (FE)":results["MSE_FE"],"RMSE (PE)":results["RMSE_PE"],"RMSE (FE)":results["RMSE_FE"],"Var (PE)":results["Variance_PE"],"Var (FE)":results["Variance_FE"],"Mean (PE)":results["Mean_PE"],"Mean (FE)":results["Mean_FE"]}
compareTable=compareTable.append(tableData,ignore_index=True)
print(compareTable.round(2) )