import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import os
os.chdir("/Users/utkarshvirendranigam/Desktop/GWU_course/Time Series/Project")

def ADF_Cal(tempData):
    results = adfuller(tempData)
    print("ADF Statistic: ", results[0])
    print("p-value: ", results[1])
    print("Critical Values:")
    for key, value in results[4].items():
        print("\t", key, ": ", value)


data1=pd.read_csv("city_hour.csv")
data1=data1[data1["City"]=="Delhi"]
data1["Datetime"]=pd.to_datetime(data1["Datetime"])

data2=pd.read_csv("delhi.csv")
# data2=data1[data1["City"]=="Delhi"]
data2["Datetime"]=pd.to_datetime(data2["date_time"])



# result = pd.merge(left, right, on=['key1', 'key2'])



# data["Date"]=data["Date"]
print(data2.dtypes)
result = pd.merge(data1, data2, how="left", on=["Datetime"])
print(result.head())
# print(data["Date"])

# filteredData=data[data["Date"].dt.year >=2006]
plt.figure()
plt.plot(result["Datetime"],result["AQI"])
plt.show()



# ADF_Cal(result["AQI"])
print(result["AQI"].head(20))
temp=result[["AQI"]]# print(result.head())


print(result.columns)

finalData=result[['Datetime', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO',
       'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI', 'cloudcover', 'humidity', 'precipMM',
       'pressure', 'tempC', 'visibility', 'windspeedKmph']]

finalData.to_csv("FinalProjectData.csv")




rows_with_nan = []
for index, row in temp.iterrows():
    is_nan_series = row.isnull()
    if is_nan_series.any():
        rows_with_nan.append(index)

print(rows_with_nan)


# filteredDataFirstOrder=filteredData["Close"].diff()
# ADF_Cal(filteredDataFirstOrder[1:])

plt.figure()
plt.plot(np.arange(1,len(filteredDataFirstOrder)),filteredDataFirstOrder[1:])
plt.show()

plt.figure()
plt.rcParams.update({'font.size': 6})
sns.heatmap(finalData.corr(),cmap='coolwarm_r', annot=True, fmt=".2f", vmin=-1, vmax=1)
plt.rcParams.update({'font.size': 8})
plt.title("Correlation Matrix")
plt.show()




# data=pd.read_csv("online_retail_II.csv")
#
#
# print(data.head())
# print(data.dtypes)
# print(data["InvoiceDate"].head())
# data["Date"]=pd.to_datetime(data["InvoiceDate"])
# data["Date"]=data["Date"].dt.date
# print(data["Date"])
# print(data["Country"].unique())
#
# print(data["Country"].value_counts())
#
#
# my_pt = pd.pivot_table(data, index=["Date"], values=["Quantity","Price"], aggfunc=np.sum)
# pivotData = pd.DataFrame(my_pt.to_records())
# print(pivotData)
#
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(pivotData["Date"],pivotData["Price"])
# plt.show()
#
#
# from statsmodels.tsa.stattools import adfuller
# def ADF_Cal(tempData):
#     results = adfuller(tempData)
#     print("ADF Statistic: ", results[0])
#     print("p-value: ", results[1])
#     print("Critical Values:")
#     for key, value in results[4].items():
#         print("\t", key, ": ", value)
#
#
# ADF_Cal(pivotData["Price"])
