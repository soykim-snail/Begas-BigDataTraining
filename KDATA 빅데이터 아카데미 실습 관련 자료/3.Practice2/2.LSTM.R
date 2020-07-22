"============================================================
  Title    : Time series analysis
  Subtitle : Electric load forecast - RNN/LSTM
  Author   : Begas 
  Date     : 20181023
  Updates  : 20200719
============================================================"

#=========================================================================
# 0. Pacakge load 
#=========================================================================
#패키지 호출
library(forecast);library(tseries);library(imputeTS);library(zoo);library(dplyr);library(keras);

#=========================================================================
# 1. 데이터 로드 및 전처리 
#=========================================================================

#현재 스크립트의 경로를 기본 경로로 설정합니다.
print(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#한 단계 상위 디렉토리를 기본 경로로 설정합니다.
setwd("../")

#전력수요량자료가 있는 경로로 설정합니다.
setwd("./0.Data/ELECTRIC_LOAD")

#자료 읽기
data <- read.delim("./전력수요_시계열자료.txt",header=T,stringsAsFactors = F)

#문자형 시간에 시간 속성 부여
data$time <- as.POSIXct(data$time)

##데이터 시간 순서 정렬 및 결측 시점 확인 

#데이터의 최초 시점 부터 마지막 시점 까지 15분간격으로 시간 벡터를 생성
f.time <- seq(min(data$time), max(data$time), by='15 mins')

#시간대 확인
summary(f.time)

#기존데이터에서 결측된 시점을 채워 넣음
i.dat <- data[match(f.time, data$time),]
i.dat$time <- f.time

#13개의 Time missing이 있는 것을 확인
print(dim(i.dat)[1]-dim(data)[1])

#전체 값의 범위을 살펴봄
boxplot(i.dat$value, main="S사 전력 사용량 범위", ylab="Kwh")

#전체 기간의 패턴을 살펴봄
plot(x=as.POSIXct(i.dat$time),y=as.numeric(i.dat$value),main="S사 전력사용량 추이",xlab="시간",ylab="Kwh",type='l')
#전체 패턴을 살펴봄 Lower bound에 약간의 out-lier로 생각 됨.
#특별히 추세와 이분산은 보이지 않음

#이상치 제거
out.seq <- quantile(i.dat$value,probs=c(0.001),na.rm=T)
i.dat$value[which(i.dat$value <= out.seq[1] )]<-NA
length(which(is.na(i.dat$value))) #13(시점결측)+18(이상치)=31개의 missing이 있는 것을 확인

#이상치 제거가 적절히 되었는지 확인함.
plot(x=as.POSIXct(i.dat$time),y=as.numeric(i.dat$value),main="S사 전력사용량 추이",xlab="시간",ylab="Kwh",type='l')

#시점 결측 (13 개) + 이상치 결측(17 개) 전력 수요량에 대하여 linear interpolation 수행
i.dat$value <- na.interpolation(i.dat$value, option="linear") 
length(which(is.na(i.dat$value))) #모두 interploation 된 것을 확인

#=========================================================================
# 2. 데이터 탐색
#=========================================================================

# 1주일 단위로 나누어 데이터를 시각적으로 살펴 봄
# 전체 시간을 7일 간격으로 분리함
seqs <- seq(min(f.time),length.out=28, by="7 days") 

#1주일의 시작과 끝 시점을 생성
week.info <- rollapply(seqs, 2, function(x){
  w.info <- data.frame(Start=as.POSIXct(x[1]),End=as.POSIXct(x[2]))
}) 

#자료를 1주일 단위로 분리함
week.list <- list()
for( i in 1:nrow(week.info)){
  week.list[[i]] <- which(i.dat[,1]>=week.info[i,1] & i.dat[,1]<week.info[i,2])  
}
week.dat <- lapply(week.list, function(y){return(i.dat[y,])})  
length(week.dat)
#총 26주 + 2일의 자료임

#26개의 plot의 상하한을 고정하기위해 전체값을 기준으로 상하한 생성
y.bound <- c(round(min(i.dat$value,na.rm=T)*0.9,-2)
             ,round(max(i.dat$value,na.rm=T)*1.1,-2))

#플랏 작성
lapply(week.dat,function(week){
  x=week[,1]
  y=week[,2]
  plot(x,y,xlab="Time",ylab="Kwh",axes=F,type='l',ylim=y.bound)
  title("S사 전력소비",line=1,cex=2)
  title(paste0(substr(min(x),1,10)," ~ ",substr(max(x),1,10)),line=0,cex=0.5)
  axis(1,as.POSIXct(paste0(seq(as.POSIXct(substr(min(x),1,10)),length.out=8,by="1 days")," 00:00:00")),c("토","일","월","화","수","목","금","토"))
  axis(2,label=round(seq(y.bound[1],y.bound[2],length.out=4),0),at=round(seq(y.bound[1],y.bound[2],length.out=4),0))
})

#27개의 주간 Plot을 살펴보니 휴일 정보가 중요한 요인으로 작용하는 것으로 생각 됨
#1일 안에 각 시점별로 값의 패턴이 있음

#=========================================================================
# 3. 추가 변수 생성 및 Normalization
#=========================================================================

#요일 정보 추가
i.dat$week <- weekdays(i.dat$time)

#국경일 정보 생성

#광복절 인덱스 추출
idx1 <- which(i.dat$time>=as.POSIXct("2017-08-15 00:00:00") & i.dat$time<as.POSIXct("2017-08-16 00:00:00"))

#임시공휴일,개천절,추석 인덱스 추출
idx2 <- which(i.dat$time>=as.POSIXct("2017-10-02 00:00:00") & i.dat$time<as.POSIXct("2017-10-07 00:00:00"))

#한글날 인덱스 추출
idx3 <- which(i.dat$time>=as.POSIXct("2017-10-09 00:00:00") & i.dat$time<as.POSIXct("2017-10-10 00:00:00"))

#크리스마스 인덱스 추출
idx4 <- which(i.dat$time>=as.POSIXct("2017-12-25 00:00:00") & i.dat$time<as.POSIXct("2017-12-26 00:00:00"))

#휴일 변수 생성 평일 0 , 휴일 1
holiday <- rep(0, nrow(i.dat))
holiday[c(idx1, idx2, idx3, idx4)] <- 1
i.dat$holiday <- holiday

#요일 정보의 평일 휴일 변환 월~일 --> (월~금)0,(토~일)1
i.dat$week[-which(i.dat$week=="일요일" | i.dat$week=="토요일")] <- 0
i.dat$week[which(i.dat$week=="일요일" | i.dat$week=="토요일")]  <- 1

#국경일 정보와 요일 정보를 합쳐 휴일 변수 생성 (평일, 월~금 : 0) (국경일, 토,일 : 1,2)
i.dat$holiday <- as.numeric(i.dat$week)+as.numeric(i.dat$holiday)

#다음달이 휴일인지 여부에 대한 변수로 바꾸기 위해 휴일 정보를 앞으로 1일 당김
#11일은 1월 1일이므로 마지막1일은 1로입력
i.dat$holiday <- c(i.dat$holiday[-c(1:96)], rep(1,96))

# 휴일의 범주가 1,2를 가지고 있는 부분을 1로 통일 
table(i.dat$holiday)
i.dat$holiday[which(i.dat$holiday==0)] <- 0
i.dat$holiday[which(i.dat$holiday==1)] <- 1
i.dat$holiday[which(i.dat$holiday==2)] <- 1
table(i.dat$holiday)

#학습에 불필요한 요일 및 시간 삭제
head(i.dat)
data <- data.matrix(i.dat[,-c(1,3)])
head(data)

#min-max normalization
plus <- min(data[,1]) #추후 예측결과 데이터 역변환시 필요 
product  <- (max(data[,1])-min(data[,1])) #추후 예측결과 데이터 역변환시 필요  
data[,1] <- (data[,1]-min(data[,1]))/(max(data[,1])-min(data[,1]))
summary(data[,1]) #scale이 0~1사이의 값이 되었음을 확인함

#=========================================================================
# 4. Batch 생성 Generator 
#=========================================================================

# data                : input data
# taget_name          : y/target/label 값의 변수명  'value'
# interval            : 96 시점 간격
# time                : 6 일 (마지막 일은 y로 사용)
# min_index,max_index : 원본데이터에서의 자료 범위 
# batch_size          : 배치당 96개 샘플 

generator <- function(data,interval,time,min_index,max_index,batch_size,taget_name){
  
  i <- min_index
  function() {
    
    start <- i
    end   <- (start-1)+batch_size
    input_index<-sapply(c(start:end),seq,by=interval,length.out=time,simplify = F)
    reset <- max(as.numeric(unlist(input_index))) > max_index
    
    if (reset){
      start <- min_index
      end   <- (start-1)+batch_size
      i <<- end+1    
      input_index<-sapply(c(start:end),seq,by=interval,length.out=time,simplify = F)
    
    }else{
      i <<- end+1    
    }

    samples <- array(0, dim = c(length(input_index), time-1,dim(data)[2]))
    targets <- array(0, dim = c(length(input_index)))
    
    for( j in 1:length(input_index)){
      indices      <- input_index[[j]]
      samples[j,,] <- data[indices[-time],]
      targets[[j]] <- data[indices[time],taget_name]
    }

    return(list(samples, targets))
  }
}

train_gen<-generator(
  data
  ,interval   = 96 # 96 시점 간격으로
  ,time       = 6  # 6 개의 시점을 고려 6개중 마지막 시점은 y로 사용 
  ,min_index  = 16320-(97*78) # 자료 시작
  ,max_index  = 16320 #자료 끝
  ,batch_size = 96 #배치 1개당 96개 샘플 사용
  ,taget_name = 'value' #전력량 변수명 
)

test_gen<-generator(
  data
  ,interval   = 96
  ,time       = 6
  ,min_index  = 16321 # 12-23일부터 예측값을 뽑을 수 있도록 인덱스를 설정함
  ,max_index  = nrow(data)
  ,batch_size = 96
  ,taget_name = 'value'
)


#=========================================================================
# 5.LSTM 딥러닝 방법을 활용한 전력 수요량 예측
#=========================================================================

# #모형 구조 생성
# model <- keras_model_sequential() %>% 
#   layer_lstm(units = 96, input_shape = list(NULL, dim(data)[[-1]]), return_sequences = TRUE)%>%
#   layer_dropout(0.5)%>%
#   layer_lstm(256)%>%
#   layer_dropout(0.5)%>%
#   layer_dense(1)
# 
#
# #모형 개요
# summary(model)
#
#파라메터 개수 
#((2*96) + 96 + (96^2)  )*4 
#((96*256) + 256 + (256^2))*4
#(256*1) + 1
#
# #Weight 업데이트 방법 및 Loss 설정 
# model %>% compile(
#   optimizer = optimizer_adam(),
#   loss = 'mse'
# )
# 
# #모형 학습 시작
# history <- model %>% fit_generator(
#   train_gen,
#   steps_per_epoch = 500,
#   epochs = 10
# )
#학습 모형 가중치 저장
#save_model_weights_hdf5(model,paste0(getwd(),"/pre_trained_lstm_model.h5"))

#=========================================================================
# 6. 학습된 모델 로드
#=========================================================================

#모델 재구성
model <- keras_model_sequential() %>%
  layer_lstm(units = 96,input_shape = list(NULL, 2),return_sequences = TRUE)%>%
  layer_dropout(0.5)%>%
  layer_lstm(256)%>%
  layer_dropout(0.5)%>%
  layer_dense(1)

# 모델 로드
load_model_weights_hdf5(model, paste0(getwd(),"/pre_trained_lstm_model.h5"))

#=========================================================================
# 7.예측 결과 평가 및 Plotting
#=========================================================================

#정확도 평가를 위해 함수 생성
mape <- function(actual, pred){
  mape <- mean(abs((actual - pred)/actual))*100
  return (mape)
}

#예측값 및 실제값 생성
test_set <- list()
y_hat    <- list()
y        <- list()

for(i in 1:9){
  test_set[[i]] <- test_gen() #Test set 생성
  y[[i]] <- test_set[[i]][[2]]*product[1]+plus[1] #실제값 생성
  y_hat[[i]] <- model%>%predict(test_set[[i]][[1]]) #예측값 생성
  y_hat[[i]] <- y_hat[[i]]*product[1]+plus[1] #min-max normalization된 값을 원래의 값으로 변환
}

#Plotting을 위해 예측값 실제값 벡터화
y.true <- c()
y.pred <- c()
for(i in 1:9){
  y.true <- c(y.true, y[[i]])
  y.pred <- c(y.pred, y_hat[[i]])
}

#결과 Plotting
png("result_plot.png",height=1600,width=2400,units = "px",res=200)
plot(y=y.true,x=as.POSIXct(i.dat[16801:nrow(i.dat),'time']),type='l',ylim=c(100,1000),col='gray60',lwd=2.5,ylab="Kwh",xlab="Time",main="LSTM")
lines(y=y.pred,x=i.dat[16801:nrow(i.dat),'time'],col="darkorange",lwd=1.5,lty=2,ylab="Kwh",xlab="Time")
legend("topleft",c("True","Pred"),lty=c(1,2),lwd=2,col=c("gray90","darkorange"))
mtext('2017-12-23(토) ~ 2017-12-31(일) 전력 수요 예측 결과',outer=T,cex=1.5)
title(paste0("MAPE : ",round(mape(y.true,y.pred),2)),cex=0.1,adj=0,line=0.2,font=1)
dev.off()



