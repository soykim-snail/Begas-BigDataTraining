library(survival)
library(survminer)
library(tidyverse) 

## data loading and preprocessing
telecomDataframe <- read.csv("./dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# checking if there exists missing values in the data
any(is.na(telecomDataframe))  
# remove missing values and 
telecom_data <- telecomDataframe[complete.cases(telecomDataframe),] %>%
  mutate(Churn = ifelse(Churn == "Yes",1,0))  
# data$Churn: death = 1, censored = 0
str(telecom_data)  # overview of the data

#========================================
# Problem 1. 전체 고객생존율 (생존함수)을 추정하라. 
# 고객의 20%가 이탈하는 시점의 추정값은 무엇인가?
#========================================
# Kaplan-Meier fitting
telecom_fit <- survfit(Surv(tenure, Churn) ~ 1, data=telecom_data)

# Kaplan-Meier Survival Curve
ggsurvplot(telecom_fit, legend='none', conf.int=TRUE)

idx_largerthan_80percent <- telecom_fit$surv > 0.8
telecom_fit$surv[idx_largerthan_80percent]

# 고객의 20%가 이탈하기 직전의 index
idx <- length(telecom_fit$surv[idx_largerthan_80percent])

# 고객의 20%가 이탈하는 시점의 추정값
telecom_fit$time[idx+1]



#========================================
# Problem 2. 변수 SeniorCitizen (0 Senior, 1 non-senior)이
# 고객이탈율과 관계가 있는지 검정하라.
#========================================
# 변수 seniorcitizen 에 따른 생존함수 추정
telecom_fit2 <- survfit(Surv(tenure, Churn) ~ SeniorCitizen, data=telecom_data)

# log-rank test
survdiff(Surv(tenure, Churn) ~ SeniorCitizen, data=telecom_data, rho=0)
# p-value 가 아주 작으므로, 귀무가설을 기각하고 Senior Citizen에 따른
# 고객이탈율 차이는 유의미하다.

# survival curve 
ggsurvplot(telecom_fit2, conf.int=TRUE, pval = T, pval.method = T)



#========================================
# Problem 3. 고객이 Senior인 경우와 그렇지 않은 경우의
# 생존함수를 각각 추정하여 시각화하라.
#========================================
senior <- telecom_data[telecom_data$SeniorCitizen==0, ]
nonsenior <- telecom_data[telecom_data$SeniorCitizen==1, ]

# 고객이 Senior인 경우
ggsurvplot(survfit(Surv(tenure, Churn) ~ 1, data=senior), 
           conf.int=TRUE, legend='none', title="Senior")

# 고객이 non-Senior인 경우
ggsurvplot(survfit(Surv(tenure, Churn) ~ 1, data=nonsenior), 
            conf.int=TRUE, legend='none', title="non-Senior")


#========================================
# Problem 4. 콕스 비례위험모형을 이용하여 생존율과
# gender, SeniorCitizen, InternetService, MonthlyCharges
# 의 관계를 모형화하라.
#========================================
# (1) 비례위험 가정이 만족하는지를 확인하라.
# (2) 각 설명변수들이 통계적으로 유의한지 확인하라.
# (3) 매달 통신비(MonthlyCharges)가 높을수록 고객이탈확률이 작은지 확인하라.

# 콕스 비례위험모형 fitting
telecom_ph <- coxph(Surv(tenure, Churn) ~ 
                    gender+SeniorCitizen+InternetService+MonthlyCharges, 
                    data=telecom_data)
summary(telecom_ph)
# SeniorCitizen, InternetServiceFiber optic, InternetServiceNo, MonthlyCharges
# 변수가 유의한 것으로 보인다.  

# 비레위험 가정 checking
test_ph <- cox.zph(telecom_ph)
ggcoxzph(test_ph)
# test 결과를 보면 SeniorCitizen, InternetService, MonthlyCharges 변수들은
# 모두 p-value = 0 이므로 PH 가정을 위반하는 변수들이라고 볼 수 있다.
# gender 변수는 Schoenfeld residual plot에 추세가 없고
# p-value가 0.29로서 PH 가정을 위반하지 않는 것으로 보인다.
 
summary(telecom_ph)
# Cox PH모형에서 MonthlyCharges의 추정된 계수는 -0.056
# 이는 MonthlyCharge가 늘어남에 따라 위험도가 감소한다는 뜻이고
# 매달 통신비(MonthlyCharges)가 높을수록 고객이탈확률이 작다고 볼 수 있다.