install.packages(c("TTR", "forecast", "tseries"))

# call libraries
library(TTR)
library(forecast)
library(tseries)

# read data
wine <- read.csv("./dataset/timedata/AustralianWines.csv")
head(wine)
# 1980년 1월부터의 자료인 것을 확인할 수 있다.

#======================================================
# Problem 1. Fortified, Red, Rose, Sparkling, Sweet white, 그리고
# Dry white 와인의 매출 자료의 시간변수를 생성하고 시계열 그림을 그려라.
# make time series variables
fortified <- wine$Fortified
fortified.ts <- ts(fortified, frequency=12, start=c(1980, 1))

red <- wine$Red
red.ts <- ts(red, frequency=12, start=c(1980, 1))

rose <- wine$Rose
rose.ts <- ts(rose, frequency=12, start=c(1980, 1))

sparkling <- wine$sparkling
sparkling.ts <- ts(sparkling, frequency=12, start=c(1980, 1))

sweetwhite <- wine$Sweet.white
sweetwhite.ts <- ts(sweetwhite, frequency=12, start=c(1980, 1))

# timeseries plots
layout(1:5)
plot.ts(fortified.ts, main="Fortified wine")
plot.ts(red.ts, main="Red wine")
plot.ts(rose.ts, main="Rose wine")
plot.ts(sparkling.ts, main="Sparkling wine")
plot.ts(sweetwhite.ts, main="Dry white wine")

#======================================================
# Problem 2. Red 와인 자료를 가지고 단순지수평활법과 Hold-Winter의
# 지수평활법을 적용하여 향후 1년의 매출을 예측해보아라.
# 시계열 그림도 그려보아라.
# Red wine: 단순지수평활법
red.ts.simple <- HoltWinters(red.ts, beta=FALSE, gamma=FALSE)
red.ts.simple.forecasts <- forecast(red.ts.simple, h=12)
red.ts.simple.forecasts  # 1995년 매출 예측값

# Red wine: Holt-Winter 지수평활법
red.ts.hw <- HoltWinters(red.ts, gamma=FALSE)
red.ts.hw.forecasts <- forecast(red.ts.hw, h=12)
red.ts.hw.forecasts  # 1995년 매출 예측값

# plots
layout(1:2)
plot(red.ts.simple.forecasts, main="단순지수평활법")
plot(red.ts.hw.forecasts, main="Holt-Winter 지수평활법")

#======================================================
# Problem 3. Sweet.white의 시계열 그림을 보고 정상성 여부를 확인하시오.
# 또한 로그변환 후 정상성 여부를 확인해보시오.
layout(1)
plot.ts(sweetwhite.ts)
adf.test(sweetwhite.ts)
# p-value = 0.094, 유의수준 0.05에서 비정상적이다.

plot.ts(log(sweetwhite.ts))
adf.test(log(sweetwhite.ts))
# p-value = 0.081, 유의수준 0.05에서 비정상적이다.

#======================================================
# Problem 4. Sweet.white 자료를 1차 차분한 뒤 단위근 검정을 실애해보시오.
sweetwhite.diff <- diff(sweetwhite.ts)
plot.ts(sweetwhite.diff)
adf.test(sweetwhite.diff)
# p-value < 0.01, 정상성을 갖는 데이터로 변환되었다.

#======================================================
# Problem 5. Sweet.white 자료에 ARIMA모형을 적합해 보아라.
# ARMA(p, d, q) fitting
# Problem 3-4에서 확인했듯이 정상성을 보장하기 인해 d=1로 놓는다.

# 표본상관도표
layout(1:2)
acf(sweetwhite.diff)
pacf(sweetwhite.diff)
# SACF, PACF 모두 첫 시차에서 시작해서 소멸하는 싸인함수 형태이므로 
# p=q 라고 할 수 있다.
# 첫 두 시차가 유의함을 볼 수 있으므로 p=q=2 라 하고,
# ARMA(2, 1, 2)을 적합하자.

sweetwhite.arima = arima(sweetwhite, order=c(2, 1, 2))
sweetwhite.arima
sweetwhite.arima.forecasts <- forecast(sweetwhite.arima, h=12)
layout(1)
plot(sweetwhite.arima.forecasts)
