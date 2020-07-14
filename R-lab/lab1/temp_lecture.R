# 회귀분석, 상관분석 강의중 메모 자료료

eng <- sample(1:10, 22)
kor <-1:22
fit1 <- lm(eng~kor)
plot(kor, eng)
abline(fit1, col="blue")
coef(fit1)

# y 값
fitted(fit1)
round(residuals(fit1),2)

fitted(fit1) + residuals(fit1)

summary(fit1)
par(mfrow=c(2,2))
plot(fit1)

attach(iris)
iris_lml <-  lm(Sepal.Width ~ Species)
summary(iris_lml)


head(anscombe)
summary(anscombe)
attach(anscombe)
plot(x1, y1)
plot(x2, y2)
plot(x3, y3)
plot(x4, y4)

a <- as.factor(1:10)
b <- factor(1:10)

