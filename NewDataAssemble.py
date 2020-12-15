import os
import pandas as pd

import numpy as np

os.chdir("/Users/utkarshvirendranigam/Desktop/GitHub Projects/Forecast-Air-Quality-Index")
files=os.listdir("CAData/")
DATA_DIR="RawData/"

print(files)
# for path in [f for f in files if f[-4:] == ".zip"]:
#     print (path)
#     commandOS="unzip "+str(DATA_DIR)+str(path)
#     os.system(commandOS)

# currData=pd.read_csv("RawData/daily_88101_2006.csv")
#
# featureName={"88101":"PM2.5","44201":"Ozone","42401":"SO2","42101":"CO","42602":"NO2","WIND":"Wind","TEMP":"Temperature","PRESS":"Pressure"}
# colNames=["Date Local","Arithmetic Mean","AQI"]
#
# count2=0
# for currFeature in featureName.keys():
#     # tempData=pd.DataFrame()
#     count1 = 0
#     for i in range(1999,2021):
#         fileName="RawData/daily_"+str(currFeature)+"_"+str(i)+".csv"
#         try:
#             currData=pd.read_csv(fileName)
#             currData=currData[currData["City Name"]=="Pittsburgh"]
#             currData=currData[colNames]
#             currData = currData.rename(columns={'Date Local': 'Date', 'Arithmetic Mean':featureName[currFeature]})
#             currData["Date"] = pd.to_datetime(currData["Date"]).dt.date
#             currData.sort_values(by=featureName[currFeature], ascending=False,inplace=True)
#             currData.drop_duplicates(subset="Date",keep="first", inplace=True)
#             if count1==0:
#                 tempData=currData.copy()
#                 count1=1
#             else:
#                 tempData=pd.concat([tempData,currData])
#         except:
#             print(fileName)
#     if count2 == 0:
#         finalData = tempData.copy()
#         colNames = ["Date Local", "Arithmetic Mean"]
#         count2 = 1
#     else:
#         finalData = pd.merge(finalData, tempData, on="Date",how="outer")
#
# print(finalData.shape)
# print(finalData["Date"].min())
# print(finalData["Date"].max())
#
# finalData.sort_values(by="Date", ascending=True,inplace=True)
# finalData.to_csv("finalData.csv")




# seriesCheck=pd.date_range(start='1-1-1999', end='10-1-2020', freq='D')



#
# currData=pd.read_csv("RawData/daily_42101_2019.csv")
# currData=currData[currData["City Name"]=="Pittsburgh"]
# currData=currData[colNames]
# currData = currData.rename(columns={'Date Local': 'Date', 'Arithmetic Mean':featureName[currFeature]})

count1=0
for i in range(1999, 2020):
    fileName = "CAData/aqidaily" + str(i) + ".csv"
    try:
        currData = pd.read_csv(fileName)
        # currData = currData[currData["City Name"] == "Pittsburgh"]
        # currData = currData[colNames]
        # currData = currData.rename(columns={'Date Local': 'Date', 'Arithmetic Mean': featureName[currFeature]})
        currData["Date"] = pd.to_datetime(currData["Date"]).dt.date
        # currData.sort_values(by=featureName[currFeature], ascending=False, inplace=True)
        # currData.drop_duplicates(subset="Date", keep="first", inplace=True)
        if count1 == 0:
            tempData = currData.copy()
            count1 = 1
        else:
            tempData = pd.concat([tempData, currData])
    except:
        print(fileName)

print(tempData.shape)
print(tempData["Date"].min())
print(tempData["Date"].max())

tempData.sort_values(by="Date", ascending=True,inplace=True)
tempData = tempData.rename(columns={'Overall AQI Value': 'AQI'})
columnsKeep=['Date', 'AQI', 'CO', 'Ozone', 'SO2', 'PM10', 'PM25', 'NO2']
tempData=tempData[columnsKeep]
tempData.to_csv("CAData.csv")
print("Missing Values:\n",tempData.isna().sum())


