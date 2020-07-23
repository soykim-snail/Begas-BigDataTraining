install.packages("lime")
install.packages("kernlab")
library(lime)
library(e1071)
library(caret)
library(kernlab)
data(mtcars)
mtcars


row.names(mtcars) <- 1:nrow(mtcars)
test_set <- sample(1:nrow(mtcars), 4)
train_df <- mtcars[-test_set, ]
test_df  <- mtcars[test_set, ]



# SVM 모델링
svm_md <- train(mpg~. , data = train_df, method = "svmLinear")
predict(svm_md, newdata = test_df)


# Train data와 f(x) 정보 입력
explainer <- lime(train_df, svm_md, 
                  bin_continuous = TRUE,     # 연속형 변수 처리방법 지정
                  n_bins = 5) 


# g(x) 생성하여 주요변수 확인
explanation <- explain(x = test_df[, -1], explainer, 
                       n_features = 4,                # 주요변수 개수 지정
                       n_permutations = 5000,         # Sample N 수
                       dist_fun = "gower",            # 거리측정 방법
                       feature_select = "auto",       # 변수선택법 : n_features개의 변수를 선택하는 방법
                       kernel_width = 0.7)            # Sample을 추출할 지역의 범위

data.frame(explanation[, 2:9])

plot_features(explanation, ncol = 1)

# Text Data
# Packages 설치 후 Library 불러오기
install.packages("text2vec")
install.packages("xgboost")
library(text2vec)
library(xgboost)
library(lime)

# Data Set Loading 및 Train / Test 나누기
data(train_sentences)
data(test_sentences)
train_sentences[1,]


# Text Data 처리 함수 만들기
get_matrix <- function(text) {
  it <- itoken(text, progressbar = FALSE)
  create_dtm(it, vectorizer = hash_vectorizer())
}

# 데이터 전처리
dtm_train = get_matrix(train_sentences$text)
#test data
sentences <- head(test_sentences[test_sentences$class.text == "OWNX", "text"], 2)
dtm_test  = get_matrix(sentences)
# XGBoost 모델링
xgb_model <- xgb.train(list(max_depth = 7, eta = 0.1, 
                            objective = "binary:logistic",
                            eval_metric = "error", 
                            nthread = 1),
                       xgb.DMatrix(dtm_train, 
                                   label = train_sentences$class.text == "OWNX"),
                       nrounds = 50)
predict(xgb_model, newdata = dtm_test)


# Train data와 f(x) 정보 입력
explainer <- lime(train_sentences$text, xgb_model, 
                  preprocess = get_matrix,                # 전처리 함수
                  keep_word_position = FALSE)          # Text에 order를 줄지 여부
# g(x) 생성하여 주요변수 확인
explanations <- explain(sentences, explainer, 
                        n_labels = 1,                      # 분류 분석에서는 label 지정 필요
                        labels = NULL,  
                        n_features = 4,                   # 주요단어 개수 지정
                        n_permutations = 5000,        # Sample N 수
                        feature_select = "auto",        # 단어(변수)선택 방법
                        single_explanation = FALSE)  # test 문장을 한 개로 통합할지 여부
# 결과확인
data.frame(explanations)


plot_features(explanations)


# Image Data
# Packages 설치 후 Library 불러오기
install.packages("magick")
library(keras)
library(abind)
library(magick)
library(lime)
library(tensorflow)
# Image data Path 지정
img_path <- system.file('extdata', 'produce.png', package = 'lime')
# Load a predefined image classifier
model <- application_vgg16(
  weights = "imagenet",
  include_top = TRUE
)
model


# 이미지 전처리 함수
img_preprocess <- function(x) {
  arrays <- lapply(x, function(path) {
    img <- image_load(path, target_size = c(224,224))
    x <- image_to_array(img)
    x <- array_reshape(x, c(1, dim(x)))
    x <- imagenet_preprocess_input(x)
  })
  do.call(abind, c(arrays, list(along = 1)))
}

test_img <- img_preprocess(img_path)
dim(test_img)

plot_superpixels(img_path, n_superpixels = 50, colour = "gray")

# Train data와 f(x) 정보 입력
explainer <- lime(img_path,                                 #Lime에서는 path를 Image로 인식
                  as_classifier(model, unlist(labels)), 
                  img_preprocess)                         #이미지 전처리 과정


# g(x) 생성하여 주요변수 확인 (시간 많이 소요)
explanation <- explain(img_path, explainer, n_labels = 2, n_features = 2, 
                       n_permutations = 2000, feature_select = "auto",
                       n_superpixels = 50,             # Super Pixel 수 지정
                       weight = 20,                     # 색상 차이 대비 거리 차이의 가중치
                       n_iter = 10,                       
                       p_remove = 0.5,                 # 각 Sample에서 SuperPixel이 제외될 확률
                       batch_size = 10,                 # 한번에 처리 할 explanation 수
                       background = 'grey'
)

plot_image_explanation(as.data.frame(explanation))

#####################################################
# Shapley value
####################################################
# Table Data
# Packages 설치 후 Library 불러오기
install.packages("iml")
library("iml")
library("randomForest")
set.seed(1)
# 데이터 불러오기 및 ML 모델
data("Boston", package  = "MASS")
rf = randomForest(medv ~ ., data = Boston, ntree = 50)
X = Boston[which(names(Boston) != "medv")]
head(X)


# Shapley Value 계산
predictor = Predictor$new(rf, data = X, y = Boston$medv)  #predictor에 모델정보와 데이터 저장
shapley = Shapley$new(predictor, x.interest = X[1,])  #Local(row) 지정
shapley$plot()
  

shapley$explain(x.interest = X[2,]) #같은 모델에 새로운 Local(row) 지정
shapley$plot()                                 


results = shapley$results #data frame으로 정리해서 보기

head(results)