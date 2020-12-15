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

filterRows = np.arange(23368, 48192)
data = pd.read_csv("Data/Raw Data.csv")
data = data.iloc[filterRows, :]
data["Datetime"] = pd.to_datetime(data["Datetime"])
data=data.set_index("Datetime")

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
y_ACF = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: Air Quality Index (New Delhi)")
plt.show()


tst.ADF_Cal(y)



yDiff=y.diff()

# Plot of the dependent variable versus time
plt.figure()
plt.plot(data.index, yDiff)
plt.title("Dependant Variable AQI vs Time")
plt.xlabel("Time")
plt.ylabel("AQI")
plt.show()


# ACF of the dependent variable
lagsCount = 250
x_ACF = np.arange(-1 * (lagsCount), lagsCount + 1)
y_ACF = []
for i in range(lagsCount + 1):
    y_ACF.append(tst.acf_eqn(yDiff[1:], i))
# print("\nACF (20 Lags): ", np.round(y_ACF,4))
tempArray = y_ACF[::-1]
y_ACF = np.concatenate((tempArray[:-1], y_ACF), axis=None)

plt.figure()
plt.stem(x_ACF, y_ACF)
plt.ylabel("Magnitude")
plt.xlabel("Lags")
plt.title("ACF: AQI First Order Diff")
plt.show()


titlePlot = "GPAC Table: Air Quality Index (New Delhi)"
tst.GPAC_Cal(y_ACF, 20, 20, titlePlot,figSize=(12,12))



ar_order=0
ma_order=15

# ar_order=6
# ma_order=5
print("\nEstimated AR Order: ",ar_order)
print("\nEstimated MA Order: ",ma_order)


runLM=tst.Levenberg_Marquardt(yDiff[1:],ar_order,ma_order,iterations=150,rateMax=1000000)
results=runLM.calculateCoefficients()

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


