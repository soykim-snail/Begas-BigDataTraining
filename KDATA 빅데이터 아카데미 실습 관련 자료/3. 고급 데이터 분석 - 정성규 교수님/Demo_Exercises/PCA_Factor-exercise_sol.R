# call libraries
library(stats)
library(car)
library(psych)

# load data
crime <- read.csv("./dataset/crime.csv", header = TRUE)
head(crime)

# preprocessing
rownames(crime) <- crime[, 1]
crime <- crime[, -1]
head(crime)

#========================================
# Problem 1. 자료에 대한 상관관계행렬을 구하고 plot() 명령어를 이욯하여
# 변수간의 관계를 살펴보시오.
#========================================
correlation_matrix <- cor(crime)
correlation_matrix
plot(crime)
# 대체적으로 변수들 사이에 선형 관계가 있는 것으로 보인다.
# 즉, 변수에 서로 상관이 있ㄷ고 볼 수 있다.

#========================================
# Problem 2. 주어진 데이터에 PCA를 시행하고 스크리 도표를 그려 적절한
# 주성분의 갯수 m을 찾으시오.
#========================================
# do PCA
pca_fit <- princomp(crime, cor=T)
summary(pca_fit)
plot(pca_fit, type="lines", main="Scree plot")
m = 3 # or m = 4

#========================================
# Problem 3. 위에서 정한 m개의 주성분이 전체변동율의 몇 퍼센트를 
# 설명하는지 찾으시오.
#========================================
p <- ncol(crime) # number of variables
explanation_ratio <- sum(pca_fit$sdev[1:m]^2) / p  # eigen_values = pca_fit$sdev^2
explanation_ratio
# m=3개의 주성분이 대략 86.8%를 설명한다.

#========================================
# Problem 4. 위에서 정한 m개의 주성분을 토대로 인자분석을 시행하시오.
# (단, 인자회전은 하지 않음) 인자적재행렬을 토대로 각각의 인자가
# 어떤 의미를 가지는지 설명하시오.
#========================================
factorPCA_fit <- principal(crime, nfactors=m, rotate="none")
factorPCA_fit$loadings
# F1: 모든 범죄들에 높은 계수
# F2: 살인에 있어서 높은 계수, murder~assualt와 burglary~auto를 분리하는 계수
# F3: robbery, larceny, auto 계수에서 상대적으로 높지만, 전반적으로 계수들이 작아 해석이 애매하다.

#========================================
# Problem 5. 이번에는 VARIMAX 인자회전을 한 인자분석을 시행하시오.
# 마찬가지로 인자적재행렬을 토대로 각각의 인자가 어떤 의미를
# 가지는지 설명하시오.
#========================================
factorPCA_varimax_fit <- principal(crime, nfactors=m, rotate="varimax")
factorPCA_varimax_fit$loadings
# F1: 실제로 사람에게 상해를 입히는 정도의 크기를 나타내는 계수
# F2: 해석이 쉽지 않음
# F3: auto, robbery에서 높은 계수

#========================================
# Problem 6. 인자회전 전후를 비교하는 그림을 그리고 결과를 해석해보시오.
#========================================
plot(x=c(-1,1), y=c(-1,1), type="n", xlab="Component 1", ylab="component 2")
abline(h=0, lty=2)
abline(v=0, lty=2)
text(factorPCA_fit$loadings[,1], factorPCA_fit$loadings[,2], labels = row.names(t(crime)), cex=1, col="blue")
text(factorPCA_varimax_fit$loadings[,1], factorPCA_varimax_fit$loadings[,2], labels = row.names(t(crime)), cex=1, col="red")