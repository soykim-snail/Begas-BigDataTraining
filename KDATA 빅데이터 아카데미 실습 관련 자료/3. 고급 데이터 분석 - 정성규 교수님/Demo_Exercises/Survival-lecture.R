library(survival)
library(survminer)
library(tidyverse) 

## Kaplan-Meier example
data(ovarian)
rx1ov <- ovarian %>% filter(rx == 1)
rx1ov

with(rx1ov, Surv(futime, fustat))

# Kaplan-Meier fitting
ov.fit <- survfit(Surv(futime, fustat) ~ 1, data=rx1ov)
summary(ov.fit)

# Kaplan-Meier Survival Curve
ggsurvplot(ov.fit, legend='none')

# 치료법 rx=1, rx=2 에 따른 생존함수 추정
ov.fit2 <- survfit(Surv(futime, fustat) ~ rx, data=ovarian)
ggsurvplot(ov.fit2)


## 직접해보기(p.25)
# load a data
data(lung)

# data information
?survival::lung 
# we can check that
# - sex: Male=1, Female=2
# - time: Survival time in days
# - status: censoring status: 1=censored, 2=dead

lung$sex[lung$sex==1] <- 'Male'
lung$sex[lung$sex==2] <- 'Female'

lung.fit <- survfit(Surv(time, status) ~ sex, data=lung)
ggsurvplot(lung.fit, conf.int=TRUE, pval=TRUE, 
           legend.title="Sex", legend.labs=c("Female", "Male"),
           risk.table=TRUE, # show risk table
           ggtheme=theme_bw(), # 격자무늬 추가
           palette = c("#E7B800", "#2E9FDF"), # color change
           size=1) # line size 
# you can modify more and more ...


## log-rank test
ggsurvplot(ov.fit2, pval = T, pval.method = T, conf.int = T)

#Gehan-Breslow-Wilcoxon
ggsurvplot(ov.fit2, pval = T, log.rank.weights = "n")
#Tharone-Ware
ggsurvplot(ov.fit2, pval = T, log.rank.weights = "sqrtN")
#Peto & Peto
ggsurvplot(ov.fit2, pval = T, log.rank.weights = "S1")

survdiff(Surv(futime, fustat) ~ rx, data = ovarian, rho=0) # log-rank
survdiff(Surv(futime, fustat) ~ rx, data = ovarian, rho=1) # Peto-Peto

## 직접해보기(p.37)
data(lung)
#log-rank test
survdiff(Surv(time, status) ~ sex, data=lung, rho=0)
# p-value = 0.001 -> 귀무가설 기각 -> 성별에 따른 폐암 생존율 차이는 유의미하다.

# Peto-Peto test
survdiff(Surv(time, status) ~ sex, data=lung, rho=1)
# p-value = 0.0004 -> 귀무가설 기각 -> 성별에 따른 폐암 생존율 차이는 유의미하다.


## 콕스 비례위험 모형
# 난소암 자료

# fitting
coxph(Surv(futime, fustat) ~ rx, data=ovarian)

coxph(Surv(futime, fustat) ~ rx + age, data=ovarian)

# 추정값 시각화
ovarian$rx <- as.factor(ovarian$rx)
ov.ph <- coxph(Surv(futime, fustat) ~ rx + age, data=ovarian)
ggforest(ov.ph, data = ovarian)

# 생존모형 시각화
ov.ph <- coxph(Surv(futime, fustat) ~ rx + age, data=ovarian)
newdf <- data.frame(age = c(50,60,70),rx = as.factor(c(1,1,1)))
ov.ph1s <- survfit(ov.ph, data = ovarian, newdata = newdf)
ggsurvplot(ov.ph1s, legend.labs = c("age = 50","age = 60","age = 70"))


## 비례위험 가정 확인
ov.ph <- coxph(Surv(futime, fustat) ~ rx + age, data=ovarian)
(test.ph <- cox.zph(ov.ph))
ggcoxzph(test.ph)

## 직접해보기(1/2, p.53)
(lung.ph <- coxph(Surv(time, status) ~ sex + meal.cal, data = lung))
# 1. sex 변수는 유의하고 (p-value=0.005), 
# meal.cal 변수는 유의하지 않다. (p-value=0.356)
(test.ph <- cox.zph(lung.ph))
ggcoxzph(test.ph)
# 2. sex 변수에 대한 p-value > 0.2292 이므로 PH 가정을 위배하지 않는다.

## 직접해보기(2/2, p.54) 
(test.ph <- cox.zph(coxph(Surv(time, status) ~ sex, data = lung)))
ggcoxzph(test.ph)
# Schoenfeld residual plot에 추세가 없고, p-value=0.091>0.05 이므로 sex 변수는 유의수준 0.05에서 
# PH 가정을 위배하지 않음.