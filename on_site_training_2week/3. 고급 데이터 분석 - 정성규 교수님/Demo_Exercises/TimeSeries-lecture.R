# time-series plot intro
kings <- scan("./dataset/timedata/kings.dat",skip=3) # http:// url 주소를 줘도 된다
kingstimeseries <- ts(kings); # ts: 순서있는 시계열이다
plot.ts(kingstimeseries)

souvenir <- scan("./dataset/timedata/fancy.dat")
souvenirtimeseries <- ts(souvenir, frequency=12, start=c(1987, 1)) # 매달 관측된 자료 (계정성분 지정방법), 1987년1월부터
plot.ts(souvenirtimeseries)

# log-transformation
logsouvenirtimeseries <- log(souvenirtimeseries) # 로그변환하면 분산이 안정화 된다.
plot.ts(logsouvenirtimeseries)

## without seasonal component
# simple moving average
library(TTR)
kingstimeseriesSMA3 <- SMA(kingstimeseries, n=3) # Simple Moving Average로최근 3년
plot.ts(kingstimeseriesSMA3)

kingstimeseriesSMA8 <- SMA(kingstimeseries, n=8)
plot.ts(kingstimeseriesSMA8)

## with seasonal component: decomposition
births <- scan("./dataset/timedata/nybirths.dat")
birthstimeseries <- ts(births, frequency=12, start=c(1946, 1))
plot.ts(birthstimeseries)
birthstimeseriescomponents <- decompose(birthstimeseries)
plot(birthstimeseriescomponents)

## seasonal adjustment
TS_components <- decompose(birthstimeseries)
TS_seasonally_adjusted <- birthstimeseries - TS_components$seasonal
plot(TS_seasonally_adjusted)


## exponential smoothing method
rain=scan("./dataset/timedata/precip1.dat",skip=1)
rainseries <- ts(rain,start=c(1813))
plot.ts(rainseries)
rainforecasts <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE)
rainforecasts
rainforecasts$fitted
plot(rainforecasts)
rainforecasts$SSE
par(mfrow=c(3,1))
plot(HoltWinters(rainseries, alpha=0.3, beta=FALSE, gamma=FALSE), main="Alpha=0.3")
plot(HoltWinters(rainseries, alpha=0.7, beta=FALSE, gamma=FALSE), main="Alpha=0.7")
plot(HoltWinters(rainseries, alpha=1, beta=FALSE, gamma=FALSE), main="Alpha=1")

alpha03 <- HoltWinters(rainseries, alpha=0.3, beta=FALSE, gamma=FALSE)
alpha07 <- HoltWinters(rainseries, alpha=0.7, beta=FALSE, gamma=FALSE)
alpha1 <- HoltWinters(rainseries, alpha=1, beta=FALSE, gamma=FALSE)

alpha03$SSE
alpha07$SSE
alpha1$SSE

rainforecasts35 <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE, l.start=35)
plot(rainforecasts35)

library(forecast)
rainforecasts2 <- forecast(rainforecasts, h=5)
rainforecasts2
plot(rainforecasts2)

## Holt's double exponential smoothing
skirts <- scan("./dataset/timedata/skirts.dat", skip=5)
skirtsseries <- ts(skirts, start=c(1866))
plot.ts(skirtsseries)
skirtsforecasts <- HoltWinters(skirtsseries, gamma=FALSE)
skirtsforecasts
plot(skirtsforecasts)

skirtsforecasts2 <- forecast(skirtsforecasts, h=10)
plot(skirtsforecasts2)

## HW-exponential smoothing
plot.ts(logsouvenirtimeseries)
souvenirforecasts <- HoltWinters(logsouvenirtimeseries)
souvenirforecasts
plot(souvenirforecasts)
souvenirforecasts2 <- forecast(souvenirforecasts, h=36)
plot(souvenirforecasts2)


## unit root test: checking whether stationarity holds or not
par(mfrow=c(1, 2))
plot.ts(skirtsseries)
plot.ts(rainseries)
library(tseries)
adf.test(skirtsseries)
adf.test(rainseries)


## ARMA(p, q)
volcanodust <- scan("./dataset/timedata/dvi.dat", skip=1)
volcanodustseries <- ts(volcanodust, start=c(1500))
plot.ts(volcanodustseries)
adf.test(volcanodustseries)  # p-value < 0.01

layout(1:2)
acf(volcanodustseries)
pacf(volcanodustseries)

volcano.arima <- arima(volcanodustseries, order=c(2, 0, 0))
volcano.arima
volcano.forecast <- forecast(volcano.arima, h=30)
plot(volcano.forecast)

volcano.auto <- auto.arima(volcanodustseries)
volcano.auto
volcano.auto.forecast<- forecast(volcano.auto, h=30)
plot(volcano.auto.forecast)

## normalizing time series data
layout(1)
plot.ts(birthstimeseries)
birthdiff1 <- diff(birthstimeseries)
plot.ts(birthdiff1)
adf.test(birthdiff1)

## ARMA(p, d, q)
tooth <- read.csv("./dataset/timedata/tooth.csv", header=TRUE)  # read a data
tooth.ts <- ts(tooth$colgate, start=c(1958, 1), frequency=52)
plot.ts(tooth.ts, main="colgate market share 1958 to 1963",
        ylab="market share")
tooth.diff1 <- diff(tooth.ts)
plot.ts(tooth.diff1,
        main="1st differenced colgate market share 1958 to 1963",
        ylab="market share")
adf.test(tooth.diff1)
layout(1:2)
acf(tooth.diff1)
pacf(tooth.diff1)
layout(1)
tooth.arima <- arima(tooth.ts, order=c(0, 1, 1))
tooth.arima
tooth.arima.forecasts <- forecast(tooth.arima, h=30)
plot(tooth.arima.forecasts)
