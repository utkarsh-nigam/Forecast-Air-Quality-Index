import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import toolkit.time_series_toolkit as tst
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from scipy.stats import chi2
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir("/Users/utkarshvirendranigam/Desktop/GitHub Projects/Forecast-Air-Quality-Index")

seed = 10

# filterRows = np.arange(23368, 48192)
data = pd.read_csv("CAData.csv")
# data = data.iloc[filterRows, :]
data["Date"] = pd.to_datetime(data["Date"])
data["AQI_Normalized"]=data["AQI"]
data.loc[data["AQI"] >250, 'AQI_Normalized'] = 250
data = data.drop('AQI', 1)
data = data.rename(columns={'AQI_Normalized': 'AQI'})
print(data["Date"].min())
print(data["Date"].max())

data=data.set_index("Date")

y=data["AQI"]

# Plot of the dependent variable versus time

plt.figure()
plt.plot(data.index, data["AQI"])
plt.title("Dependant Variable AQI vs Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()


# ACF of the dependent variable

lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(data["AQI"], i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF[::-1]
y_ACF_Plot = np.concatenate((tempArray[:-1], y_ACF), axis=None)





dataMean1=[]
dataVariance1=[]
for i in range(len(y)):
    dataMean1.append(y[:i].mean())
    dataVariance1.append(y[:i].var())
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
plt.show()

tst.ADF_Cal(y)

# Stationary data

yStationary=y.diff()

plt.figure()
plt.plot(data.index[1:], yStationary[1:])
plt.title("AQI (1st Order Differencing) vs Time")
plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.show()

lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_Stationary_ACF = []
for i in range(lagsCount + 1):
    y_Stationary_ACF.append(tst.acf_eqn(yStationary[1:], i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_Stationary_ACF[::-1]
y_Stationary_ACF_PLot = np.concatenate((tempArray[:-1], y_Stationary_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_Stationary_ACF_PLot)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: AQI (1st Order Differencing)")
plt.show()


dataMean2=[]
dataVariance2=[]

for i in range(1,len(y)):
    dataMean2.append(yStationary[1:i].mean())
    dataVariance2.append(yStationary[1:i].var())
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

tst.ADF_Cal(yStationary[1:])



# colorSeq=sns.color_palette("pastel")
# columnSeq=['NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
# i=0
# j=1
# for val in columnSeq:
#     plt.subplot(1,3,j)
#     sns.violinplot(data[val],showmeans=True, showmedians=True,color=colorSeq[i], bw_method=None,orient="v",inner="quartile")
#     plt.title(val)
#     i+=1
#     j+=1
#     if j>3:
#         plt.show()
#         j=1




# Split the dataset into train set (80%) and test set (20%)


independantVariables=list(data.columns)
independantVariables.remove("AQI")
x=data[independantVariables]
# for colValue in independantVariables:
#     x[colValue].fillna(value=x[colValue].mean(),inplace=True)

x_train,x_test,y_train, y_test,yStationary_train,yStationary_test = train_test_split(x,y,yStationary, test_size=0.2, shuffle=False)
# x_train,x_test,y_train, y_test = train_test_split(x,y, test_size=0.2, shuffle=False)

hSteps=len(y_test)
trainingCount=len(y_train)

def transformFirstOrderPrediction(actualData,predictedData):
    finalPred=np.array([np.NaN])
    yTemp=actualData[1:-1]+predictedData[1:]
    finalPred=np.concatenate((finalPred, yTemp), axis=None)
    return finalPred

def transformFirstOrderForecast(actualData,forecastedData):
    finalFore=[actualData[-1]]
    for i in range(len(forecastedData)):
        finalFore.append(finalFore[-1]+forecastedData[i])
    return np.array(finalFore[1:])

HoltWinterSeasonalModel = ets.ExponentialSmoothing(y_train, seasonal="additive",seasonal_periods=365,freq=y_train.index.inferred_freq).fit()
predictedData = HoltWinterSeasonalModel.fittedvalues
forecastedData = HoltWinterSeasonalModel.forecast(steps=hSteps)

# predictions=transformFirstOrderPrediction(y_train,predictedData)
# forecasts=transformFirstOrderForecast(y_train,forecastedData)


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

#
#
lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_train_ACF = []
for i in range(lagsCount + 1):
    y_train_ACF.append(tst.acf_eqn(y_train, i))


lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_Stationary_train_ACF = []
for i in range(lagsCount + 1):
    y_Stationary_train_ACF.append(tst.acf_eqn(yStationary_train[1:], i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
# tempArray = y_Stationary_train_ACF[::-1]
# y_Stationary_train_ACF = np.concatenate((tempArray[:-1], y_Stationary_train_ACF), axis=None)


titlePlot = "GPAC Table: AQI_Train"
tst.GPAC_Cal(y_train_ACF, 20, 20, titlePlot,figSize=(12,12))

titlePlot = "GPAC Table: AQI_Train_Stationary"
tst.GPAC_Cal(y_Stationary_train_ACF, 20, 20, titlePlot,figSize=(12,12),minSize=1)



arOrderArray=[3]
maOrderArray=[3]

for a in range(0,len(arOrderArray)):


    ar_order=arOrderArray[a]
    ma_order=maOrderArray[a]
    model_name="ARMA("+str(ar_order)+", "+str(ma_order)+")"

    print("\n")
    print(50 * "*")
    print("\t\t",model_name)
    print(50 * "*")

    # print("\nEstimated AR Order: ",ar_order)
    # print("\nEstimated MA Order: ",ma_order)

    runLM=tst.Levenberg_Marquardt(yStationary,ar_order,ma_order,iterations=150,rateMax=1000000)
    results=runLM.calculateCoefficients()

    try:
        ar_Array=[]
        ma_Array=[]
        for i in range(ar_order):
            currCoefficent=results[0][i][0]
            ar_Array.append(currCoefficent)
            currStdValue=np.sqrt(results[2][i,i])
            print("\na", i + 1, ":",np.round(currCoefficent,2))
            print("Confidence Intervals: ",np.round(currCoefficent-2*currStdValue,2)," < a", i + 1," < ",np.round(currCoefficent+2*currStdValue,2))

        for i in range(ma_order):
            currCoefficent = results[0][i+ar_order][0]
            ma_Array.append(currCoefficent)
            currStdValue = np.sqrt(results[2][i+ar_order, i+ar_order])
            print("\nb", i + 1, ":",np.round(currCoefficent,2))
            print("Confidence Intervals: ", np.round(currCoefficent - 2 * currStdValue,2)," < b", i + 1," < ", np.round(currCoefficent + 2 * currStdValue,2))
    except:
        continue



armaForecast = tst.Basic_Forecasting_Methods(y_train)
predictions = armaForecast.ARMAMethodPredict(ar_Array, ma_Array)
forecasts = armaForecast.ARMAMethodForecast(ar_Array, ma_Array,hSteps)
# predictions=transformFirstOrderPrediction(y_train,y_pred)
# forecasts=transformFirstOrderForecast(y_train,y_fore)



plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[1:trainingCount], predictions, color="g", label="h-step forecast")
plt.xlabel("t")
plt.ylabel("y")
plt.rcParams.update({'font.size': 8})
plt.title("Model Prediction")
plt.legend()
plt.show()



plt.figure()
plt.plot(data.index[: trainingCount], y_train, color="blue", label="Train Data")
plt.plot(data.index[trainingCount:], y_test, color="orange", label="Test Data")
plt.plot(data.index[trainingCount:], forecasts, color="g", label="h-step forecast")
plt.xlabel("t")
plt.ylabel("y")
plt.rcParams.update({'font.size': 8})
plt.title("Model Forecasts")
plt.legend()
plt.show()


from statsmodels.graphics import tsaplots

f, axs = plt.subplots(2,1, figsize=(20,10))
tsaplots.plot_acf(yStationary_train.dropna(), alpha=0.05,ax=axs[0], lags=50)
# plt.show()
tsaplots.plot_pacf(yStationary_train.dropna(), alpha=0.05,ax=axs[1], lags=50)
plt.show()



tsa.ARIMA(y_train['close'], order=(2,1,1), freq=y_train.index.inferred_freq).fit(disp=0)