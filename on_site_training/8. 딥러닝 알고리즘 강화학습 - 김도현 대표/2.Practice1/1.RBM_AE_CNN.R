"============================================================
  Title    : RBM vs AE vs CNN
  Updates  : 20200718 
============================================================"

rm(list = ls())

#=========================================================================
# 0. 패키지 로드
#=========================================================================
library(deepnet); library(keras); library(data.table); library(imager);
library(e1071)

#=========================================================================
# 1. 데이터 준비
#=========================================================================
#현재 스크립트의 경로를 기본 경로로 설정합니다.
print(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#한 단계 상위 디렉토리를 기본 경로로 설정합니다.
setwd("../")

#MNIST 자료가 있는 경로로 설정합니다.
setwd("./0.Data/MNIST")

#mnist 데이터 로드 
mnist <- readRDS("mnist.RDS")

#학습데이터 개수 지정
num_train <- 60000

#테스트 데이터 개수 지정
num_test <- 10000

#이미지 height,width 지정 
x_pixl <- 28; y_pixl <-28;

# 이미지 Normalization 0 ~ 255 -> 0 ~ 1
X_train <- mnist$train$x/255 
X_test  <- mnist$test$x/255 

# Flatten (이미지를 하나의 벡터형태로 펼치기)
X_train_flatten <- array_reshape(X_train, c(num_train, x_pixl*y_pixl))
X_test_flatten  <- array_reshape(X_test, c(num_test, x_pixl*y_pixl))

# Label(Y) 생성 
Y_train<-mnist$train$y
Y_test<-mnist$test$y

# dimension 확인
cat("Dim(X_train): ", dim(X_train), "\n")
cat("Dim(X_test): ", dim(X_test), "\n")
cat("Dim(X_train_flatten): ", dim(X_train_flatten), "\n")
cat("Dim(X_test_flatten): ", dim(X_test_flatten), "\n")
cat("Dim(Y_train): ", dim(Y_train), "\n")
cat("Dim(Y_test): ", dim(Y_test), "\n")


#=========================================================================
# 2.이미지 탐색
#=========================================================================

# 그래픽 파라메터 설정
par(mfrow=c(5,5), mai=c(0,0.1,0.3,0)) #margin size (b, l, t, r)

#학습자료의 처음 25개의 이미지 Plotting
for(n in 1:25){
  image(t(X_train[n,,])[,y_pixl:1], axes=F, col=gray((0:255)/255))
}
par(fig=c(0,1,0.9,1), new=T)
plot.new()
title("X_train: Frist 25 images", cex.main=2, col.main="purple")
dev.off()


# 테스트자료의 처음 25개의 이미지 Plotting
par(mfrow=c(5,5), mai=c(0,0.1,0.3,0))
for(n in 1:25){
  image(t(X_test[n,,])[,y_pixl:1], axes=F, col=gray((0:255)/255))
}
par(fig=c(0,1,0.9,1), new=T)
plot.new()
title("X_test: Frist 25 images", cex.main=2, col.main="purple")
dev.off()


#=========================================================================
# 3. Model: Restricted Boltzmann Machine
#=========================================================================

# RBM 학습
RBM_mnist <- rbm.train(X_train_flatten,
                       hidden = 100,
                       batchsize=128,
                       numepochs = 10,
                       cd=1 )

# Train 자료를 input으로 하여 RBM 모델의 Hidden node 값을 추출
RBM_hidden_node_train <- rbm.up(RBM_mnist, X_train_flatten)

# Test자료를 input으로 하여 RBM 모델의 Hidden node 값을 추출
RBM_hidden_node_test <- rbm.up(RBM_mnist, X_test_flatten)

# Test자료 이미지 복원 
RBM_recon_vec <- rbm.down(RBM_mnist, RBM_hidden_node_test)  

# 이미지 복원값 Array로 만들기
RBM_recon <- array(RBM_recon_vec, dim=c(dim(RBM_hidden_node_test)[1], x_pixl, y_pixl))

#생성된 특징(Feature)의 dimmension 확인 
# Train 자료 Hidden node 값 dim
cat("RBM_hidden_node_train Dim: ", dim(RBM_hidden_node_train), "\n")
# Test 자료 Hidden node 값 dim
cat("RBM_recon_vec RBM_hidden_node_test: ", dim(RBM_hidden_node_test), "\n")
# Test 자료의 output array 변환 후 dim
cat("RBM_recon Dim: ", dim(RBM_recon), "\n")

#=========================================================================
# 4. Model: Autoencoder
#=========================================================================

# 모델 정의 
AE <- keras_model_sequential()

AE %>%
  layer_dense(input_shape = c(x_pixl*y_pixl), units = 100, activation = "relu",name='AE_hidden_layer') %>%
  layer_dense(units = x_pixl*y_pixl, activation = "sigmoid",name='output_layer')

#정의된 모델 확인
summary(AE)


# loss 및 optimizer 설정
AE %>% compile(
  optimizer = "adam",
  loss = 'binary_crossentropy'
)

# 학습
AE %>% fit(
  X_train_flatten, X_train_flatten, 
  shuffle = TRUE, 
  epochs = 10, 
  batch_size = 128, 
  validation_split = 0.2
)

# Test자료를 input으로 하여 AE 모델의 Hidden layer의 값을 추출
layer_name            <-  'AE_hidden_layer'
feature_extract_model <-  keras_model(inputs = AE$input,outputs = get_layer(AE, layer_name)$output)
AE_hidden_layer_train <-  predict(feature_extract_model,X_train_flatten)
AE_hidden_layer_test  <-  predict(feature_extract_model,X_test_flatten)

#테스트자료의 이미지 복원값 계산 
AE_recon_vec <- AE %>% predict(X_test_flatten)

#계산된 결과값 이미지 어레이로 변경
AE_recon <- array(AE_recon_vec, dim=c(num_test, x_pixl, y_pixl))

# 생성된 특징(Feature)의 dimmension 확인 
# Train 자료의 특성 추출 값 dim
cat("AE_hidden_layer_train Dim : ", dim(AE_hidden_layer_train), "\n")

#Test 자료의 특성 추출 값 dim
cat("AE_hidden_layer_test Dim : ", dim(AE_hidden_layer_test), "\n")

#Test 자료의 복원이미지 차원 dim
cat("AE_recon Dim : ", dim(AE_recon), "\n")

#=========================================================================
# 5. RBM, AE 복원 이미지 확인 
#=========================================================================

# 원본이미지 vs 복원이미지 확인
par(mfrow=c(3,5), mai=c(0,0.1,0.3,0))
#원본이미지 
for(n in 1:5){
  image(t(X_test[n,,])[,y_pixl:1], axes=F, col=gray((0:255)/255),main="Origianl")
}

#RBM 복원이미지
for(n in 1:5){
  image(RBM_recon[n,,][,y_pixl:1], axes=F, col=gray((0:255)/255),main="RBM")
}

#AE 복원이미지
for(n in 1:5){
  image(AE_recon[n,,][,y_pixl:1], axes=F, col=gray((0:255)/255),main="AE",col.main="purple")
}

dev.off()

#=========================================================================
# 6.특성 추출값 + SVM 을 이용한 multi-class classification
#=========================================================================


#RBM 특성 추출 데이터 Column name 변경 및 데이터 프레임 변환
colnames(RBM_hidden_node_train)<-paste("RBM_Feature_",1:100)
colnames(RBM_hidden_node_test)<-paste("RBM_Feature_",1:100)
RBM_hidden_node_train<-data.frame(RBM_hidden_node_train)
RBM_hidden_node_test<-data.frame(RBM_hidden_node_test)

#AE 특성 추출 데이터 Column name 변경 및 데이터 프레임 변환
colnames(AE_hidden_layer_train)<-paste("AE_Feature_",1:100)
colnames(AE_hidden_layer_test)<-paste("AE_Feature_",1:100)
AE_hidden_layer_train<-data.frame(AE_hidden_layer_train)
AE_hidden_layer_test<-data.frame(AE_hidden_layer_test)

# SVM 자료 준비
rbm_x<-RBM_hidden_node_train
ae_x<-AE_hidden_layer_train
y<-as.factor(Y_train)

#SVM 모델 Fitting

#학습 시간이 오래걸려 미리 학습한 파일을 저장해 두었습니다. 
#svm_rbm_model <- svm(rbm_x, y, probability = TRUE)
#svm_ae_model  <- svm(ae_x, y, probability = TRUE)
#saveRDS(svm_rbm_model,"svm_rbm_model.RDS")
#saveRDS(svm_ae_model,"svm_ae_model.RDS")

#학습된 모델 로드 
svm_rbm_model <-readRDS("svm_rbm_model.RDS")
svm_ae_model <-readRDS("svm_ae_model.RDS")

#계산 시간이 오래걸려 미리 결과 파일을 저장해 두었습니다. 
#예측값 산출
#svm_rbm_pred_prob <- predict(svm_rbm_model, RBM_hidden_node_test, decision.values = TRUE, probability = TRUE)
#svm_ae_pred_prob <- predict(svm_ae_model, AE_hidden_layer_test, decision.values = TRUE, probability = TRUE)

#예측 정확도 계산
#pred_result<-data.frame(rbm_pred=svm_rbm_pred_prob,ae_pred=svm_ae_pred_prob,true_label=Y_test)
#saveRDS(pred_result,"svm_pred_result.RDS")

#예측값 테이블 로드
pred_result <-readRDS("svm_pred_result.RDS")

#RBM 예측 정확도
rbm_acc<-sum(pred_result[,1]==pred_result[,3])/nrow(pred_result)

#AE 예측 정확도
ae_acc<-sum(pred_result[,2]==pred_result[,3])/nrow(pred_result)


#=========================================================================
# 6. CNN을 활용한 multi-class classification
#=========================================================================


##CNN 학습을 위한 데이터 준비

#CNN 파라메터 설정
batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 28
img_cols <- 28

# CNN 학습을 위한 MNIST 자료 차원 변경
X_train <- array_reshape(X_train, c(nrow(X_train), img_rows, img_cols, 1))
X_test <- array_reshape(X_test, c(nrow(X_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# 변경된 차원 확인
cat('X_train_shape:', dim(X_train), '\n')
cat(nrow(X_train), 'train samples\n')
cat(nrow(X_test), 'test samples\n')

# Softmax output에 알맞은 형태로 one-hot encoding
Y_train <- to_categorical(Y_train, num_classes)
Y_test <- to_categorical(Y_test, num_classes)

##CNN 모델 정의

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_dropout(rate = 0.25) %>% 
  layer_flatten() %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# 학습된 모델 로드
model<-load_model_hdf5("cnn_mnist_model.h5",compile = F)

# 모델 컴파일
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)


# 모델 학습
# model %>% fit(
#   X_train, Y_train,
#   batch_size = batch_size,
#   epochs = epochs,
#   validation_split = 0.2
# )
# save_model_hdf5(model,"cnn_mnist_model.h5")

# 모델 평가
scores <- model %>% evaluate(
  X_test, Y_test, verbose = 0
)

#예측 정확도 계산 
cnn_acc<-scores[[2]]


#=========================================================================
# 7. 최종 결과 확인
#=========================================================================
cat('RBM Test accuracy:', rbm_acc, '\n')
cat('AE Test accuracy:', ae_acc, '\n')
cat('CNN Test accuracy:', cnn_acc, '\n')
