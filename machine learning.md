

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

