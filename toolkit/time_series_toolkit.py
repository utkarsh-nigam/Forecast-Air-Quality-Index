import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

def ADF_Cal(tempData,thresholdValue=0.05):
    results = adfuller(tempData)
    print("ADF Statistic: ", results[0])
    print("p-value: ", results[1])
    print("Critical Values:")
    for key, value in results[4].items():
        print("\t", key, ": ", value)
    if(results[1]<=thresholdValue):
        print("\nSince, p-value is less than equal to ",thresholdValue,", therefore, dataset is STATIONARY!")
    else:
        print("\nSince, p-value is greater than ", thresholdValue, ", therefore, dataset is NOT STATIONARY!")


def correlation_coefficent_cal(x, y):
    r = np.sum((x - np.mean(x)) * (y - np.mean(y))) / (
                (np.sqrt(np.sum(np.square(x - np.mean(x))))) * (np.sqrt(np.sum(np.square(y - np.mean(y))))))
    return r


def partialCorrelation(r_ab, r_ac, r_bc):
    return ((r_ab - (r_ac * r_bc)) / (np.sqrt(1 - np.square(r_ac)) * np.sqrt(1 - np.square(r_bc))))


def Q_cal(acfSeries, T):
    return (T * np.sum(np.square(acfSeries)))


def acf_eqn(tempData, tempCount):
    tempData = np.array(tempData)
    numerator = 0
    denominator = 0
    if tempCount == 0:
        return 1
    else:
        for i in range(tempCount + 1, len(tempData) + 1):
            numerator += (tempData[i - 1] - np.mean(tempData)) * (tempData[i - tempCount - 1] - np.mean(tempData))
        for j in range(0, len(tempData)):
            denominator += np.square(tempData[j] - np.mean(tempData))

        if denominator != 0:
            return round(numerator / denominator, 2)
        else:
            return 1


def movingAverageCalculator(y, m, fold=0):
    if m % 2 == 1:
        length = len(y) - m + 1
        movingAverage = np.zeros(length, )
        for k in range(0, m):
            movingAverage += y[k:k + length]
        movingAverage = movingAverage / m
        if fold != 0:
            length = len(movingAverage) - fold + 1
            movingAverage2 = np.zeros(length, )
            for k in range(0, fold):
                movingAverage2 += movingAverage[k:k + length]
            movingAverage2 = movingAverage2 / fold
            tempArray = np.empty((int((m - 1) / 2) + int((fold - 1) / 2),))
            tempArray[:] = np.NaN
            return (np.concatenate((tempArray, movingAverage2)))
        else:
            tempArray = np.empty((int((m - 1) / 2),))
            tempArray[:] = np.NaN
            return (np.concatenate((tempArray, movingAverage)))
    else:
        length = len(y) - m + 1
        movingAverage = np.zeros(length, )
        for k in range(0, m):
            movingAverage += y[k:k + length]
        movingAverage = movingAverage / m
        if fold != 0:
            length = len(movingAverage) - fold + 1
            movingAverage2 = np.zeros(length, )
            for k in range(0, fold):
                movingAverage2 += movingAverage[k:k + length]
            movingAverage2 = movingAverage2 / fold
            if fold == 2:
                tempArray = np.empty(((int(((m / 2) - 1)) + int((fold / 2))),))
            else:
                tempArray = np.empty(((int(((m / 2) - 1)) + int((fold / 2) - 1)),))
            tempArray[:] = np.NaN
            return (np.concatenate((tempArray, movingAverage2)))
        else:
            tempArray = np.empty((int((m / 2) - 1),))
            tempArray[:] = np.NaN
            return (np.concatenate((tempArray, movingAverage)))


def GPAC_Cal(acf_series, rows, columns, plotTitle="GPAC Table",figSize=(8, 8),minSize=0):
    tableData = np.empty((rows + 1, columns))
    tableData[:] = np.NaN
    for j in range(0, rows + 1):
        for k in range(1, columns + 1):
            tempDenominatorData = np.empty((k, k))
            tempDenominatorData[:] = np.NaN
            for kValue in range(1, k + 1):
                for kValue2 in range(1, k + 1):
                    currentValue = kValue2 - kValue - j
                    tempDenominatorData[kValue - 1, kValue2 - 1] = acf_series[np.absolute(currentValue)]

            tempNumeratorData = tempDenominatorData.copy()
            for kValue in range(1, k + 1):
                tempNumeratorData[kValue - 1, k - 1] = acf_series[j + kValue]

            currentPACValue = (np.linalg.det(tempNumeratorData) / np.linalg.det(tempDenominatorData))
            tableData[j, k - 1] = np.round(currentPACValue, 3)

    yticks = np.arange(0, rows + 1, dtype=np.int)
    xticks = np.arange(1, columns + 1, dtype=np.int)
    plt.figure(figsize=figSize)
    if minSize==0:
        plot = sns.heatmap(tableData, annot=True, fmt=".3f", yticklabels=yticks, xticklabels=xticks)
    else:
        plot = sns.heatmap(tableData, annot=True, fmt=".3f", yticklabels=yticks, xticklabels=xticks, vmin=-1*minSize, vmax=minSize)
    plot.set_yticklabels(plot.get_yticklabels(), rotation=0, fontsize=8, va="center")
    plt.rcParams.update({'font.size': 8})
    plt.title(plotTitle)
    plt.show()


def standard_error_cal(series, features):
    T = len(series)
    return (np.sqrt((1 / (T - features - 1)) * (np.sum(np.square(series)))))


class Levenberg_Marquardt():
    def __init__(self, y_series, order_ar, order_ma, iterations=50, rate=0.01, rateMax=100):
        super(Levenberg_Marquardt).__init__()
        self.y = y_series
        self.N = len(self.y)
        self.na = order_ar
        self.nb = order_ma
        self.n_orders = self.na + self.nb
        # print(self.n_orders)
        self.iterationCount = iterations
        self.mu = rate
        self.muMax = rateMax
        self.orderCompleteStack = []
        for i in range(np.absolute(self.na - self.nb)):
            self.orderCompleteStack.append(0)

    def step1(self):
        self.thetaTemp = self.theta.flatten()
        if self.na == 0:
            den = [1]
        else:
            den = [1] + self.thetaTemp[0:self.na].tolist()
            # den = [1]+np.reshape(self.theta[0:self.na],(self.na,-1)).tolist()
        if self.nb == 0:
            num = [1]
        else:
            num = [1] + self.thetaTemp[self.na:self.n_orders].tolist()
            # num = [1]+np.reshape(self.theta[self.na:self.n],(self.nb,-1)).tolist()
        if (self.na > self.nb):
            num += self.orderCompleteStack
        elif (self.na < self.nb):
            den += self.orderCompleteStack
        sys = (den, num, 1)
        _, self.e = signal.dlsim(sys, self.y)
        self.SSE_theta = np.dot(np.transpose(self.e), self.e)
        self.SSE_Array2.append(self.SSE_theta[0])
        for k in range(self.n_orders):
            numTemp = num.copy()
            denTemp = den.copy()
            if k < self.na:
                denTemp[k + 1] = denTemp[k + 1] + np.power(0.1, 6)
            else:
                numTemp[k - self.na + 1] = numTemp[k - self.na + 1] + np.power(0.1, 6)
            sys = (denTemp, numTemp, 1)
            _, self.e_Temp = signal.dlsim(sys, self.y)
            self.e_Temp = (self.e - self.e_Temp) / np.power(0.1, 6)
            if k == 0:
                self.e_TempArray = self.e_Temp
            else:
                self.e_TempArray = np.hstack((self.e_TempArray, self.e_Temp))

        self.X = self.e_TempArray.copy()
        self.A = np.dot(np.transpose(self.X), self.X)
        self.g = np.dot(np.transpose(self.X), self.e)

    def step2(self):
        self.check = self.mu * np.identity(self.n_orders)
        self.check = self.A + self.check

        self.deltaTheta = np.dot(np.linalg.inv(self.check), self.g)
        self.thetaNew = self.theta + self.deltaTheta
        self.thetaNewTemp = self.thetaNew.flatten()
        if self.na == 0:
            den = [1]
        else:
            den = [1] + self.thetaNewTemp[0:self.na].tolist()
        if self.nb == 0:
            num = [1]
        else:
            num = [1] + self.thetaNewTemp[self.na:self.n_orders].tolist()
        if (self.na > self.nb):
            num += self.orderCompleteStack
        elif (self.na < self.nb):
            den += self.orderCompleteStack

        sys = (den, num, 1)
        _, self.e_new = signal.dlsim(sys, self.y)
        self.SSE_thetaNew = np.dot(np.transpose(self.e_new), self.e_new)
        if (np.isnan(self.SSE_thetaNew) or np.isinf(self.SSE_thetaNew)):
            self.SSE_thetaNew = 2.71828 ** 10
            self.SSE_Array2.append(self.SSE_thetaNew)
        else:
            self.SSE_Array2.append(self.SSE_thetaNew[0])

    def calculateCoefficients(self):
        self.theta = np.zeros((self.n_orders, 1))
        self.SSE_Array2 = []
        self.step1()
        self.step2()
        self.SSE_Array = []
        # self.SSE_Array2 = []
        self.count = 0
        while self.count < self.iterationCount:
            if self.SSE_thetaNew < self.SSE_theta:
                if np.linalg.norm(self.deltaTheta) < 0.001:
                    self.theta = self.thetaNew
                    self.var_e = self.SSE_thetaNew / (self.N - self.n_orders)
                    self.covar_theta = self.var_e[0] * np.linalg.inv(self.A)
                    return self.theta, self.var_e, self.covar_theta, self.SSE_Array,self.SSE_Array2

                else:
                    self.theta = self.thetaNew
                    self.mu = self.mu / 10

            while self.SSE_thetaNew >= self.SSE_theta:
                self.mu = self.mu * 10
                if self.mu > self.muMax:
                    print(self.theta)  # , self.var_e, self.covar_theta)
                    print("Coefficients did not converge!")
                    return 0
                # self.step1()
                self.step2()
            self.count += 1
            self.SSE_Array.append(self.SSE_theta[0])
            if self.count >= self.iterationCount:
                print(self.theta)  # ,self.var_e,self.covar_theta)
                print("Coefficients did not converge!")
                return 0
            self.theta = self.thetaNew
            self.step1()
            self.step2()



class Basic_Forecasting_Methods():
    def __init__(self, y_series):
        super(Basic_Forecasting_Methods).__init__()
        self.time_series = y_series

    def averageMethodPredict(self):
        self.predictedData = [np.NaN]
        for i in range(1, len(self.time_series)):
            self.predictedData.append(np.mean(self.time_series[0:i]))
        return self.predictedData

    def averageMethodForecast(self, steps=1):
        self.forecastedData = []
        for i in range(0, steps):
            self.forecastedData.append(np.mean(self.time_series))
        return self.forecastedData

    def naiveMethodPredict(self):
        self.predictedData = [np.NaN]
        for i in range(1, len(self.time_series)):
            self.predictedData.append(self.time_series[i - 1])
        return self.predictedData

    def naiveMethodForecast(self, steps=1):
        self.forecastedData = []
        for i in range(0, steps):
            self.forecastedData.append(self.time_series[-1])
        return self.forecastedData

    def driftMethodPredict(self):
        self.predictedData = [np.NaN, self.time_series[0]]
        for i in range(2, len(self.time_series)):
            tempPred = self.time_series[i - 1] + ((self.time_series[i - 1] - self.time_series[0]) / (i - 1))
            self.predictedData.append(tempPred)
        return self.predictedData

    def driftMethodForecast(self, steps=1):
        self.forecastedData = []
        for i in range(1, steps + 1):
            tempFore = (self.time_series[-1] + (
                        i * ((self.time_series[-1] - self.time_series[0]) / (len(self.time_series) - 1))))
            self.forecastedData.append(tempFore)
        return self.forecastedData

    def SESMethodPredict(self, alpha=0.5):
        self.predictedData = [np.NaN, self.time_series[0]]
        for i in range(2, len(self.time_series)):
            tempPred = (alpha * self.time_series[i - 1]) + (1 - alpha) * (self.predictedData[i - 1])
            self.predictedData.append(tempPred)
        return self.predictedData

    def SESMethodForecast(self, steps=1, alpha=0.5):
        self.predictedData = [self.time_series[0]]
        for i in range(1, len(self.time_series)):
            tempPred = (alpha * self.time_series[i - 1]) + (1 - alpha) * (self.predictedData[i - 1])
            self.predictedData.append(tempPred)
        self.forecastedData = []
        tempPred = (alpha * self.time_series[- 1]) + (1 - alpha) * (self.predictedData[- 1])
        for i in range(0, steps):
            self.forecastedData.append(tempPred)
        return self.forecastedData

    def ARMAMethodPredict(self, ar_coeff, ma_coeff):
        self.coeff_ar = ar_coeff
        self.coeff_ma = ma_coeff
        self.predictedData = [0]
        for t in range(2, len(self.time_series) + 1):
            tempPred = 0
            for j in range(1, len(self.coeff_ar) + 1):
                currARCoeff = -1 * self.coeff_ar[j - 1]
                if (t - j) > 0:
                    tempPred += currARCoeff * self.time_series[t - j - 1]
            for k in range(1, len(self.coeff_ma) + 1):
                currMACoeff = self.coeff_ma[k - 1]
                if (t - k) > 0:
                    tempPred += currMACoeff * (self.time_series[t - k - 1] - self.predictedData[t - k - 1])
            self.predictedData.append(tempPred)
        return (self.predictedData)

    def ARMAMethodForecast(self, ar_coeff, ma_coeff, steps=1):
        self.coeff_ar = ar_coeff
        self.coeff_ma = ma_coeff
        self.predictedData = [0]
        for t in range(2, len(self.time_series) + 2):
            tempPred = 0
            for j in range(1, len(self.coeff_ar) + 1):
                currARCoeff = -1 * self.coeff_ar[j - 1]
                if (t - j) > 0:
                    tempPred += currARCoeff * self.time_series[t - j - 1]
            for k in range(1, len(self.coeff_ma) + 1):
                currMACoeff = self.coeff_ma[k - 1]
                if (t - k) > 0:
                    tempPred += currMACoeff * (self.time_series[t - k - 1] - self.predictedData[t - k - 1])
            self.predictedData.append(tempPred)

        self.forecastedData = [self.predictedData[-1]]
        t = len(self.time_series)
        for h in range(2, steps + 1):
            tempFore = 0
            for j in range(1, len(self.coeff_ar) + 1):
                currARCoeff = -1 * self.coeff_ar[j - 1]
                if (j - h) >= 0:
                    tempFore += currARCoeff * self.time_series[t - j + h - 1]
                else:
                    tempFore += currARCoeff * self.forecastedData[h - j - 1]

            for k in range(1, len(self.coeff_ma) + 1):
                currMACoeff = self.coeff_ma[k - 1]
                # previousValue=i-1
                if (k - h) >= 0:
                    tempFore += currMACoeff * (self.time_series[t - k + h - 1] - self.predictedData[t - k + h - 1])
            self.forecastedData.append(tempFore)
        return self.forecastedData