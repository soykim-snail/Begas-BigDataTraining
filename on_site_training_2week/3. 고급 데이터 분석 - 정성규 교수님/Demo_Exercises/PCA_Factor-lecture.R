# call libraries
library(stats)
library(car)
library(psych)

# data pre-processing
cereal = read.csv("./dataset/cereals.csv")
head(cereal)
cereal=cereal[,c("name","calories","protein","fat","sodium","fiber","carbo","sugars",
                 "potass","vitamins")]

cereal[!complete.cases(cereal),]
cereal=cereal[-c(5,21,58),]

rownames(cereal)=cereal[,"name"]
cereal=cereal[,-1]
head(cereal)

# correlation matrix
round(cor(cereal),3)

plot(cereal)

library(stats)
fit <- princomp(cereal, cor=T)  # PCA fitting
summary(fit) 


round(fit$loadings[,],3) # 아이겐벡터를 찾는 것
round(fit$loadings[,1],3)
round(fit$loadings[,2],3) 
plot(fit,type="lines")

round(head(fit$scores),3)
round(head(predict(fit, cereal)),3)

biplot(fit)

# Factor analysis
require(psych) #심리학 자료분석
fit1 <- principal(cereal, nfactors=3, rotate="none") # 주성분분석, 인자갯수는 정해줘야 함 (주성분분석한 결과에서 참고해도 됨.)
fit1$loadings

# VARIMAX 인자회전
fit2 <- principal(cereal, nfactors=3, rotate="varimax") # 0에 가까와 뚤린 값이 많아져라
fit2$loadings

# 회전전후 비교
plot(x=c(-1,1), y=c(-1,1), type="n", xlab="Component 1", ylab="component 2")
abline(h=0, lty=2)
abline(v=0, lty=2)
text(fit1$loadings[,1], fit1$loadings[,2], labels = row.names(t(cereal)), cex=1, col="blue")
text(fit2$loadings[,1], fit2$loadings[,2], labels = row.names(t(cereal)), cex=1, col="red")
