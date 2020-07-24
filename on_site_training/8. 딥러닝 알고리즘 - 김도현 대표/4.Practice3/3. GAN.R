# coding: utf-8
"============================================================
  Title    : Generative Adversarial Network
  Subtitle : MNIST Dataset
  Author   : Begas - Chanyoung Lee
  Date     : 20180819
  Updates  : 20190428 
============================================================"

rm(list = ls())
##############################################################
##     Step 0. 작업환경     
##############################################################

# 작업디렉토리 지정
# 현재 Rscript의 경로 확인 및 working directory 설정
print(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#한단계 상위 디렉토리로 나가기
setwd("../")
#데이터 폴더 지정
setwd("./0.Data/MNIST")

# 패키지 로드
library(keras)


##############################################################
##     Step 1. 데이터     
##############################################################

# MNIST 자료 다운로드시
# dataset_mnist 이용
# mnist <- dataset_mnist()
# c(c(x_train, y_train), c(x_test, y_test)) %<-% mnist

# mnist_data.RDS 이용
mnist <- readRDS("./mnist.RDS")
#mnist[[1]][[1]] : x_train (image) #mnist[[1]][[2]] : y_train (label)
#mnist[[1]][[1]] : x_test (image) #mnist[[1]][[2]] : y_test (label)

x_train <- mnist[[1]][[1]]

cat("min(x_train) :", min(x_train),
    "\nmax(x_train) :", max(x_train))

# scaling 0~255 to -1~1
x_train <- (x_train - 127.5) / 127.5
cat("min(x_train) :", min(x_train),
    "\nmax(x_train) :", max(x_train))
# dimension 확인
cat("Dim(x_train) :", dim(x_train))


##############################################################
##     Step 2. 모형 구성
##############################################################

### 2-0. 모형 구성 모수 설정
latent_dim <- 100
height     <- 28
width      <- 28
channels   <- 1 # gray scale

### 2-1. Generator 네트워크 구성(난수로 부터 가짜 이미지를 생성하는 모델)
generator_input  <- layer_input(shape=c(latent_dim) )
generator_output <- generator_input %>% 
  layer_dense(units=128*7*7, kernel_initializer=initializer_random_normal(stddev = 0.02)) %>%  
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_reshape(target_shape=c(7, 7, 128)) %>%  
  layer_upsampling_2d(size=c(2, 2)) %>% 
  layer_conv_2d(filters=64, kernel_size=5, padding="same") %>% 
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_upsampling_2d(size=c(2, 2)) %>% 
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_conv_2d(filters=channels, kernel_size=5, activation="tanh", padding="same")
# 모형 정의
generator <- keras_model(generator_input, generator_output)
# 모형 확인
summary(generator)

### 2-2. Discriminator 네트워크 설정(입력 자료에 대하여 진위를 판별하는 모델)
discriminator_input  <- layer_input(shape=c(height, width, channels))
discriminator_output <- discriminator_input %>% 
  layer_conv_2d(filters=64, kernel_size=3, padding="same") %>% 
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_dropout(rate=0.3) %>% 
  layer_conv_2d(filters=128, kernel_size=5, strides=2) %>% 
  layer_activation_leaky_relu(alpha=0.2) %>% 
  layer_dropout(rate=0.3) %>% 
  layer_flatten() %>% 
  layer_dropout(rate=0.4) %>% 
  layer_dense(units=1, activation="sigmoid")
# 모형 정의
discriminator <- keras_model(discriminator_input, discriminator_output)
# 모형 확인
summary(discriminator)

# Discriminator의 optimize 함수 설정

discriminator_optimizer <- optimizer_adam(lr=0.0002, clipvalue=1.0,
                                          decay=1e-8, beta_1 = 0.5)
# 모형 Compile
discriminator %>% compile(optimizer=discriminator_optimizer,
                          loss="binary_crossentropy")

### 2-3. Adversarial 네트워크 설정(Discriminator의 가중치를 고정한 뒤 Generator의 가중치에 대해서만 학습하는 모델)  
# discriminator의 weight를 freeze
freeze_weights(discriminator)
# GAN 모형 구성
gan_input    <- layer_input(shape=c(latent_dim))
gan_output   <- discriminator(generator(gan_input))
# 모형 정의
gan          <- keras_model(gan_input, gan_output)

# GAN의 optimize 함수를 설정
gan_optimizer <- optimizer_adam(lr=0.0002, clipvalue=1.0,
                                decay=1e-8, beta_1 = 0.5)
# 모형 Compile
gan %>% compile(optimizer=gan_optimizer,
                loss="binary_crossentropy")


##############################################################
##     Step 3. 모형 학습 및 가짜 이미지 생성    
##############################################################
### 3-0. 생성 이미지 디렉토리 설정
#경로 재설정
print(dirname(rstudioapi::getActiveDocumentContext()$path))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#저장 디렉토리 설정
save_dir <- file.path("RESULT","MNIST")

### 3-1. 학습 모수 설정
iterations <- 100
batch_size <- 100

### 3-2. 모형 학습 및 가짜 이미지 생성
start <- 1
for (step in 1:iterations){
  cat("\r",step) 
  random_latent_vectors <- matrix(rnorm(batch_size*latent_dim, mean=0, sd=1), nrow=batch_size, ncol=latent_dim)
  
  generated_images <- generator %>% predict(random_latent_vectors)
  stop             <- start + batch_size - 1
  real_images      <- array(x_train[start:stop,,], dim=c(batch_size, 28, 28, 1))
  
  rows <- nrow(real_images)
  combined_images            <- array(0, dim=c(rows*2, dim(real_images)[-1]))
  combined_images[1:rows,,,] <- generated_images
  combined_images[(rows+1):(rows*2),,,] <- real_images
  
  labels <- rbind(matrix(1, nrow=batch_size, ncol=1), matrix(0, nrow=batch_size, ncol=1))
  labels <- labels + (0.5*array(runif(prod(dim(labels))),dim=dim(labels)))
  
  d_loss <- discriminator %>% train_on_batch(combined_images, labels)

  random_latent_vectors <- matrix(rnorm(batch_size*latent_dim), nrow=batch_size, ncol=latent_dim)
  misleading_targets    <- array(0, dim=c(batch_size, 1))
  
  a_loss <- gan %>% train_on_batch(random_latent_vectors, misleading_targets)
  
  start  <- start + batch_size
  if (start > nrow(x_train) - batch_size)
    start <- 1
  if (step %% 100 == 0){
    save_model_weights_hdf5(gan, file.path("RESULT","MNIST","MODEL","gan_mnist.h5"))
    cat("\n","discriminator loss :", d_loss,"\n","adversarial loss   :", a_loss,"\n")
    image_array_save(array_reshape(generated_images[1,,,],dim=c(28,28,1)),
                     path=file.path(save_dir, paste0("generated_mnist", step, ".png")))
    #image_array_save(array_reshape(real_images[1,,,],dim=c(28,28,1)),
    #                 path=file.path(save_dir, paste0("real_mnist", step, ".png")))
  }
}



