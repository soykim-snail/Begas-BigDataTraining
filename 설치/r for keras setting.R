# install.packages("tidyverse")
# install.packages("devtools")
# install.packages("remotes")
# install.packages("reticulate")
# remotes::install_github("rstudio/keras", dependencies = TRUE)
# install.packages("tensorflow")
library(devtools)
library(remotes)
library(reticulate)
library(tidyverse)
library(keras)
# library(tensorflow)


# python 지정
use_python("C:/ProgramData/Anaconda3/python")
py_discover_config()

# set enviroment using reticulate : python_version 확인 반드시 필요함
conda_create("r-reticulate")
use_condaenv("r-reticulate")
conda_install("r-reticulate", 'keras', pip = TRUE, python_version = "3.7")
# install_tensorflow()

# install_keras
install_keras()
# install_keras(method = "conda")

# --------------------------------------------------------------------------------------------- # 
# test 


# 
mnist <- keras::dataset_mnist()
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

plot(history)

model %>% evaluate(x_test, y_test)

model %>% predict_classes(x_test)