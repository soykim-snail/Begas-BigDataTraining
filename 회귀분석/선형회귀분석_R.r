#============================================================
#title: "선형회귀분석"
#subtitle: "Linear Regression"
#============================================================

# Setting
setwd(paste0(getwd(), "/dataset"))

#============================================================

## 예제 데이터 
# 유사한 제품을 생산하는 12개 기업에 대해 
# 1년 광고비(독립변수, x)와 매출액(종속변수, y)
x <- c(11, 19, 23, 26, 56, 62, 29, 30, 38, 39, 46, 49)
y <- c(23, 32, 36, 46, 93, 99, 49, 50, 65, 70, 71, 89)

plot(y~x, xlab="adver", ylab="sales")

adsales.lm <- lm(y ~ x)
anova(adsales.lm)

summary(adsales.lm)

# 잔차분석
windows()
par(mfrow=c(2,2))
plot(adsales.lm)

# 등분산성 검정
resid(adsales.lm)
rstandard(adsales.lm)

# 정규성 검정
par(mfrow=c(1,2))
hist(rstandard(adsales.lm))
boxplot(rstandard(adsales.lm))

# 독립성 검정(Durbin-Watson 통계량)
install.packages("car")
library(car)
durbinWatsonTest(adsales.lm)
lmtest::dwtest(adsales.lm)

# 이상점, 영향점 분석
# 1년 광고비(독립변수, ad)와 매출액(종속변수, sales)

infludata<-data.frame(ad=x, sales=y)
infludata.lm<-lm(sales~ad, data=infludata)
summary(infludata.lm)

# 잔차의 절대값이 큰 2개의 자료 (10, 12번) 를 제외
# 10번만 제외하는 경우 
fit.10<-lm(sales~ad, infludata[-10,])
summary(fit.10)

# 12번만 제외하는 경우
fit.12<-lm(sales~ad, infludata[-12,])
summary(fit.12)

# 10, 12번 모두를 제외하는 경우
fit.1012<-lm(sales~ad, infludata[c(-10, -12),])
summary(fit.1012)

# 영향력 측도 분석
influence.measures(infludata.lm)

# Cook's D
cooks.distance(infludata.lm)
plot(infludata.lm, which=4)

# DFFITS
dffits(infludata.lm)
plot(dffits(infludata.lm), type="h")

# DFBETAS
dfbetas(infludata.lm)
dfbetaPlots(infludata.lm)

# Boston dataset
insatll.packages("MASS")
library(MASS)
data(Boston)
str(Boston)

medv.lm<-lm(medv~., data=Boston)
summary(medv.lm)

# Correlation
Boston.cor<-cor(Boston[1:13])
Boston.cor

# VIF
summary(vif(medv.lm))

# Condition number
eigenval<-eigen(Boston.cor)$values
sqrt(max(eigenval)/eigenval)

# 모형 적합 - 단계별 회귀
fit.model0 <- lm(medv ~ 1, data=Boston)
add1(fit.model0, scope = ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, test = "F")

fit.model1 <-lm(medv ~ crim, data=Boston)
add1(fit.model1, scope = ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, test = "F")

fit.model2 <- lm(medv ~ crim + zn, data=Boston)
add1(fit.model2, scope = ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, test = "F")

fit.model3 <- lm(medv ~ crim + zn + indus, data=Boston)
add1(fit.model3, scope = ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, test = "F")
summary(fit.model3)

fit.model4 <- lm(medv ~ crim + zn + indus + nox, data=Boston)
drop1(fit.model4, test="F")

# Stepwise()
fit.model <-step(fit.model0, scope = ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, direction = "both" )
summary(fit.model)

# Forward Selection
fit.model0 <-lm(medv ~ 1., data=Boston)
fit.forward <- step(fit.model0, scope = ~ crim + zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat, direction = "forward" )
summary(fit.forward)

# Backward Elimination
fit.fullmodel <-lm(medv ~ ., data=Boston)
fit.backward <- step(fit.fullmodel, direction = "backward" )
summary(fit.backward)







