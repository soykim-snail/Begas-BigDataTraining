

Bias & Variance tradeoff

![DATA COOKBOOK :: Bias - Variance Trade-off(편향-분산 트레이드 오프 ...](https://t1.daumcdn.net/cfile/tistory/996DB433599AC34225)

기계학습 방법롤

- supervised learning
- unsupervised learning

variable

- numeric
  - coutinuous
  - discrete
- categorical
  - binary
  - niminal
  - ordinal

Bias & Variance

	- MSE (mean squared error) = $Bias^2 + Var$

Inputs : concept, attributes, instance

> 데이터가 무엇인지 이해하는 것이 시스템으로 어떻게 처리되는지 아는 것 보다 중요
>
> 데이터 이해 --> 데이터 가공처러 ---> 데이터 표준화

training data : training data + validation data (7:3)

K-fold Cross Validation



Fashion MNIST

https://www.kaggle.com/zalando-research/fashionmnist

---

강사 : 장원철 교수

강의 : 2020-07-17

## 군집분석(Cluster Analysis)

> 탐색적 자료분석에 해당한다. 
>
> 성공사례는 K-mean 이용하여 분류하여 신한카드 코드9 만들기 사례

##### 군집분석이란?

데이터간 거리를 이용하여 집단으로 그룹화 한다. 

---

##### 거리(distance)

거리를 어떻게 잴까? 정의해야 함. 

- 유클리드 거리
- 맨하탄 거리(절대값)
- 마할라노비스
  - *마할라노비스 : 공분산 행렬로 나눠 더블카운팅을 경감시킨다.* 
  - $d(x ,y) = (x-y)'\sum^{-1}(x-y)$
- correlation (곡의 유사성, 문서의 유사성), ... )

상관계수 : 1-상관계수 해야 거리로 사용

거리만 잘 정의되면 numerical, non-numerical 데이터들의 복합체에서도 사용가능 하다!

변수의 단위에 영향을 받으니까 정규화 (또는 표준화) 필요함 (평균 0, 표준편차 1로 만든다.)

---

##### Linkage Method

- 최단연결법(Single Linkage Method) : 고리모양이면 매우 부적절
- 최장연결법(Complete Linkage Method) : 타이트하게 묶어줌
- 평균연결법 (average): 거리들의 평균으로
- 중앙연결법(centroid) : 군집의 중심들의 평균 (덴드로그램에서 인버전을 발생시키는 문제 있음)
- Minimax : 군집내의 최대거리를 지정함 (ex. 통신기지국의 셀타운)

---

##### 다른 분석들과 비교

- 판별분석은 레이블링 있음, 군집분석은 레이블없는 상태를 논의

- 요인분석은 변수를 그룹화, 군집분석은 개체를 그룹화

---

#### 계층적 군집분석 방법 

- 군집의 수를 미리 정할 필요가 없다 (단일에서 세분화 되는 스펙트럼에서 선택하면 됨)
- 다양한 연결법이 있고, 결과가 다르게 나온다. : 최단 연결법, 최장  연결법, 중심 연결법, 평균 연결법 .... 등
- 병합을 단계적으로 수행한다. (적정 수준에서 끊어 주면 됨)

###### 덴드로그램

![img](https://support.minitab.com/ko-kr/minitab/18/cluster_obs_dendrogram_with_final_partition_glove_testers.png)

계층적군집방법에서는 별다른 performance measure가 없기 때문에 덴드로그램만 던져진다.



#### 비계층적 군집분석 방법

- 군집수를 미리 정해야 함

- K-mean (최초 포인트에 따라서 결과가 많이 달라짐, 비교하면 됨)

  | ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/K_Means_Example_Step_1.svg/124px-K_Means_Example_Step_1.svg.png) | ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/K_Means_Example_Step_2.svg/139px-K_Means_Example_Step_2.svg.png) | ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/K_Means_Example_Step_3.svg/139px-K_Means_Example_Step_3.svg.png) | ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/K_Means_Example_Step_4.svg/139px-K_Means_Example_Step_4.svg.png) |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | initial point 잡기                                           | 포인트에 가까것 클러스터링                                   | 평균으로 중심으로 재설정                                     | 수렴할 때 까지 반복                                          |

  

- K's ++ : k-mean의 문제점 개선

  

- ##### 자기조직화 지도(Self-Organizing Map)

- ![img](https://upload.wikimedia.org/wikipedia/commons/7/70/Synapse_Self-Organizing_Map.png)

  - 입력층(input layer)과 경쟁층(competitive layer)
  - input들은 학습을 통해 경쟁층에 매핑됨 (지도, map) : 
    - 유사도 계산, 할당, 재조정 (처음 weight는 임의, 조정과정에 따라 가중됨)
  - 계속 반복
  - input 노드들이 특정 중심들로 잡아당겨지는 효과가 발생함

  - ![img](https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/TrainSOM.gif/220px-TrainSOM.gif)
  - 클러스팅한 결과는 다른 분석에 사용함 (ex. K means 라거나...)

- 밀도 기반 군집(DBSCAN, Density base spatial clustering of applications with noise)

  - mean shift algorithm으로 군집시킨다. 





## 연관규칙 분석(Association Rule)

기저귀와 맥주 매출이 동반 상승하는 현상이 발견되었고, 상품진열에 반영하니 매출이 5배 늘었다.

item 간의 if-then 관계를 찾는 방법

비지도학습의 일종

평가척도

##### 지지도(support)

- 관련성이 있다고 판단되는 품목들을 포함하고 있는 사건의 비율
- $P(A \cup B)$

##### 신뢰도(confidence)

- 조건부 확률
- $P(B|A)$

##### 향상도(lift)

- 우연에 비해 향상되었나?
- $P(B|A)/P(B)$



##### Apriori 알고리즘

- 특정 지지도 이상을 갖는 항목만 셀렉 (빈발항목집합)

- 셀렉된 것만 연관규칙 계산

  

## 주성분 분석(Principal Component Analysis)

데이터의 변동을 가장 잘 설명할 수 있도록 저차원으로 프로젝션 시키는 방법

변수만큼 주성분 나오는데, 설명력 큰 것만 셀렉함.

표준화 하여 분석한다 (즉, cor=T 옵션으로 분석)

- 단, 측정단위가 없다면 변환 불필요.

- 의미있는 단위라면 변환 안시킴 (ex.물리적 거리 (ex.위경도 좌표) )



## 고급회귀분석

#### 1. 주성분회귀(principal component regression)

#### 2. 부분최소제곱 (partial leat squres)

주성분분석으로 componet를 뽑되, 고유값도 크고, Y와 상관관계가 높은 것을 뽑는다.

#### Shrinkage Method

- Ridge Regression

- LASSO

- Elasticnet



---

강사: 원중호 교수

날짜: 2020-07-20

## 의사결정 나무(Decision Tree)

지도학습이다. 

분류 및 예측에 사용됨

if-then 규칙을 도출한다

좋은 해석력 (예측력이 좋다는 것은 아님.)



변수가 범주형자료이면, 절단기준을  정해야 함. (어떻게? 데이터를 보고.)

**Decision Tree의 구성요소 :** 

- root node, 
- internal node, 
- terminal node (=잎사귀, leaf node), 
- depth : 뿌리마디로부터 끝 마디까지 마디수

**분석단계**

- 형성 (growing)
  - 최적의 분리규칙(split rule)을 찾아서 나무를 성장 시키는 단계
  - 분리기준(split criterion), 정지규칙(stopping rule)을 지정해야 함.
  - p차원의 공간을 가능한 동질적인 공간으로 분할하는 과정임
  - 불순도 측도(impurity)
    - 지니 지수
    - 엔트로피 지수
- 가지치기 (prunning)
  - overfitting을 줄이고, 일반화 성능을 높이기 위해 가지치기
- 타당성평가
  - 여러방법
  - validation data로 검정



### 앙상블(Ensemble)

여러개의 예측모형들을 결합하여 하나의 예측모형을 만드는 방법

왜 하나? : 모형의 안정성, 예측력의 증가

**방법들**

- Bagging (bootstrap aggregating)
- Random Forest
- Boosting

---

### SVM (Support Vector Machine)

more firm theoretical bases than NN. 

Vapnik (90년대 중반~)에서 시작

#### optimal separating hyperplane (최적분리초평면)

운이좋은 경우 n차원 샘플을 초평면으로 분리 가능. 어느 초평면이 최적일까?에 대한 해답

여백을 수치화 하여 최대화 한 것이 최적.

margin의 정의 = 선과 샘플 사이의 거리의 최소값

#### support vector machine



---

### kNN (K-nearest neighbors)

가장 가까운 k개 중 다수가 속한 집단으로 분류

참 과정이 어떤 함수를 따르든 적당히 잘 예측함

**거리 계산은**

- euclidean
- manhattan
- minkowski
- 등등 사용
- (주의!) 거리계산에 앞서 데이터 표준화 필요함

**k 결정**

- 일반적으로 1~20 범위
- 오분류율이 가장 낮은 k 선택

**분류경계값**

- 다수결에 대한 measure
-  0.5 로 설정하는게 상식에 부합

**3개 이상의 그룹으로 분류하는 문제로 쉽게 확장 가능**

**장단점**

- 장점
  - 단순, 모수에 대한 가정 없어서 쉽게 이용 가능
  - 학습 데이터가 많은 경우 좋은 성능
- 단점
  - 근접 이웃 찾는데 계산 시간 많이 걸림
  - p가 증가할 때, 이웃이 멀어지는 문제
- 



---

### Naive Bayes Classification

모든 예측 변수를 서로 독립으로 가정하여 단순히 곱하여 사후확률을 구하는 방법임

베이즈공식으로 사후확률을 구하는 것으로 사건의 원인을 추측함.

장점

- 단순, 빠름
- 노이즈, 결측치에 강하다
- 차원이 높으면 독립이 아닌 사건들에 대해서 독립가설을 적용하는 것이 합리적이라는 이론적 근거가 밝혀져 있음

단점

- 가정에 결함 있음
- 범주형 변수에만 사용 가능
- 추정된 확률을 신뢰할 수 없음

---

## 딥러닝 개요

강사: 베가스 컨설팅 김윤응 팀장

강의일: 2020-07-21

---

- activation functions

![Complete Guide of Activation Functions | by Pawan Jain | Towards ...](https://miro.medium.com/max/1192/1*4ZEDRpFuCIpUjNgjDdT2Lg.png)

(참고!) logistic 함수와 Sigmoid 함수는 정확히 같은 것의 다른 이름

$$\frac{e^x} {e^x + 1} = \frac{1} {1+e^{-x}}$$ 



## 딥러닝 알고리즘

강사: 베가스 대표 김도현

강의일: 2020-07-22

딥러닝 vs. shallow learning (노드 수 많게)

사전학습으로 활용한다.

**학습의 일반적인 방법**

학습은 최소화하고자 하는 loss 함수를 정의한 후,

gradient descent 기법을 이용해서 문제를 해결하는 것
$$
\theta_1 = \theta_0 -\eta*\Delta(\theta_0)
$$

- $\Delta(\theta_0)$  : 미분? 고민할게 없다. 수학자가 한다.

- $\eta$ : 학습률? 함께 고민한다. 
  - 기울기가 크면 최저에 멀듯 하니 크게, 기울기가 0에 가까우면 최저가 가까울 듯 하니 작게 ... 식으로 하는게 상식에 부합

- $\theta_0$ : 출발점? Restricted Boltzman, AutoEncoder 방법 등을 적용하니 잘 작동하더라.

sigmoid 함수 사용으로 발생한 문제들을 해결

- 미분값이 작아짐(vinishing gradient problem) : 방향만 쓰고 값은 버리자 (rectified linear unit, ReLU)

#### 최적화 기법들

##### 훈련자료수 조절

- stochastic gradient descent (SGD, 하나씩 학습)
- batch gradient descent (BGD, 통으로 학습)
- mini-batch gradient descent (MBGD, 쪼개서 학습)

##### 학습률 변형

- momentum : 기존 진행 방향을 기억해서 일부 반영하자.
  - 가속되는 경향이 문제
- AdaGrad : 학습률은 조금 둔화시키고, 변화의 폭은 반영해 준다. 
  - 미분값이 클때는 스텝이 큰 게 문제
- RMSProp : AdaGrad의 가중평균
- ADAM : momentum과 RMSProp의 혼합형

### Restrictied Boltzmann Machine

비지도학습 모델

차원축소, 분류, 선회회귀, feature learning, collaborative filtering, topic modeling.. 등 다양하게 사용된다.

비선형에서의 주성분분석에 해당한다.

확률적 모형

### Autoencoder

차원축소, 노이즈제거

RBM과 유사하지만 결정적 모형

### Convolution Neural Network

- convolution 연산을 사용하는 신경망
- convolution? 
  - 값에 필터를 곱하는 것

### Recurrent Neural Network

- 전 까지의 값이 다음에 영향을 준다면, 나는 어떻게 모델링을 하면 좋을까?
- 시간의 흐름을 갖는 sequence 데이터를 다룬다. (음성, 동영상, 주식 시세 ... 등)

### Long Short Term Memory

---

### Generative Adversarial Network

그림을 생성하는자와 평가하는 자가 대립하여 서로의 성능을 점차 개선해가는 비지도학습 신경망모델

generative model : 데이터 분포를 학습하고 이로부터 새로운 데이터를 생성

### eXplainable Artificial Intelligence

- AOT (and or template)

- LIME

  - 번수들을 범주화 한후 (연속형, 텍스트, 이미지 ... 모두 적절히 segmentation)
  - 일단 랜덤샘플링하고
  - 가까운 점에 가중치를 많이주는 방법으로 모델링

- Shapley Value :  수요예측, 가격예측한 후에 해석하기에 적절함

  - 설명이 직관적이고 이해하기 쉽다. 

  - 확률을 몬테카를로 기법으로 추정하는 방법을 썼다.

  - 게임이론으로 경제학자가 만들었음.

    

### Reinforcement Learning

Markov decision process(MDP) 상황에 한정하여 해결함.

*Markov property(마코프 확률과정): 기억하지 않는 확률과정*

##### MDP 구성요소

- S : 상태 (state)
- A : 행동 (action)
- R : 보상 (reward)
- $P^a_{ss`}$ :상태변환확률 (state transition probability)
- $\gamma$ : 할인률 (discount factor)
- $\pi$ : 정책 (policy)

6개 구성요소를 정의할 수 있으면 문제를 풀 수 있다.

푼다는 것은 보상의 현재가치의 최대화하는 해를 구하는 것

##### Multi-armed Bandit

신약개발, 넷플릭스 영화 추천 등에 사용됨