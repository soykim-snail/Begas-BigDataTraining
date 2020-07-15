Lectures date 2020-07-14

lecturer 오희석 교수 (서울대 통계학과 )

---

# 1. 시각화 및 기초 통계

Why Info graphics?

- 인지 과정을 도와주는 시각화

![Whiskey flavor profiles - 86 scotch whiskies, 12 flavor categories](https://i.imgur.com/1fh6eyc.png)



데이터 군집화(clustering) + 구글지도 매시업(mash-up)

![Rplot02](https://revolution-computing.typepad.com/.a/6a010534b1db25970b01a3fb27e46e970b-800wi)

---

### 기술통계

수집한 자료의 정리, 표현, 요약, 해석 등을 통해 자료의 특성을 규명하는 통계적 방법

- 중심 경향성의 측도 (central tendency) : 평균(sample mean), 중앙값(median : 매우 로버스트한 방법), 최빈값(mode)

- 산포도의 측도 (dispersion) : 범위(range), 분산(variance), 표준편차(standard deviation), IQR(interquartile range)
- Five-Number Summary : 최소값, 1분위수, 중앙값, 3분위수, 최대값 --> box plot을 만든다.



#### 자료의 종류

- 연속형 자료
- 이산형 자료
  - 계수형 자료 (counting)
  - 범주형 자료(categorical)

#### box plot 

*Thanks to John Tukey*

- inner fense : [Q1 - 1.5`*`IQR, Q3 + 1.5`*`IQR]  --> 밖에 있으면 minor outliers

- outer fense : [Q1 - 3`*`IQR, Q3 + 3`*`IQR]  --> 밖에 있으면 serious outliers

---

### 추론통계

모집단에 대한 미지의 양상을 통계학을 이용하여 추측하는 과정

추정, 유의성 검정, 가설 검정



###### 점추정(point estimation)

- 모평균을 표본평균으로 추정, 모분산은 표본분산으로 추정 ---> 의사결정에 사용 불가

###### 구간추정(interval estimation)

- 추정량의 신뢰구간을 제공하여 모수의 범위를 추측하는 것

- 신뢰구간은 추출시마다 바뀐다.

###### 유의성 검정

- 귀무가설을 기각하고자 하는 분명한 이유를 갖고 하는 행위로, 귀무가설에 대한 반증의 강도를 제공하는 과정이다.
- 검정통계량
- 오류
  - 제1종 오류 : 귀무가설이 옳은데 기각할 오류 (유의확률, P-Value)(반증의 강도로 사용됨)
    - 테스트 전에 유의수준(ex. alpha=0.05)을 선언해 놓는다.
  - 제2종 오류 : 귀무가설이 틀린데 기각 못 할 오류

- 절차

  1. 귀무가설, 대립가설, 유의수준을 설정한다.

  2. 표본을 추출하고 검정통계량의 값을 계산한다.

  3. 검정통계량의 값을 유의수준과 비교하여 평가한다. 

     이때, 유의확률이나 기각역을 통해 검정통계량의 값을 평가하고 귀무가설을 기각할 수있는지 판단한다. (검정통계량의 분포를 알아야 하는데 표본평균이면 대충 정규분포)

     ** Bootstrap** 방법론이 나와서 정규분포 안써도 됨 *thanks to Bradley Efron*

  4. 가설을 기각할 수 있는지 없는지를 판단하고, 결론을 낸다.

(주의!) 샘플이 커지면 당연히 P-Value는 작아진다. 합리적 판단되려면 유의수준도 낮추어야 한다.

대응비교 : 진통제 효과



# 2. 회귀분석/분산분석

상관계수

- roh = Corr(X, Y)
- -1과 1사이의 값
- 직선형태의 관계에 한정됨

표본상관계수

- *r*

검정통계량 ~ t(n-2) 따른다. 
$$
T = sqrt(n-2) * r / sqrt(1-r^2)
$$


Pearson's correlation : 두 변량이 정규모집단으로부터 랜덤표본일 때

Spearman's : 아닐 때



회귀분석

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Normdist_regression.png/300px-Normdist_regression.png)

-  "Regression" 명명은 Francis Galton에서 유래

  *선조의 영향이 기울기가 1보다 작구나, 진화적으로 퇴보하는구나*

$$
y=beta_0 + beta_1 x
$$

- 할일

  1. 절편과 기울기 추정

  2. 기울기=0 이란 귀무가설을 검정



오차항의 가정

- 정규분포는 원래 정보량 0인 오차에서 유래. Gauss는 원래 오차분포로 명명

- 따라서, 선형모형의 잔차도 바로 정규분포가 될 것이 기대됨 --> 분석을 마쳤을 때 잔차가 정규분포와 흡사하면 올바른 분석을 했다고 안심하면 됨. 이것을 잔차분석이라 함.

예측모형으로 사용될 수 있음



$R^2$ (결정계수, Multiple R-Squared) : 

- 회귀직선이 얼마나 데이터를 잘 설명하는가

- 1이면 모든 점들이 회귀직선 상에 있음
- 설명변수가 늘어가면 당연히 값이 커지니, (ex. 강아지 키를 넣어도 커진다.)
- Ajusted R-squred를 쓰자. ($R^2$을 자유도로 나눈 것)

F-statistic

- 모형전체를 테스트하고자 할 때
- 즉,  $H_0: beta_1 = beta_2 =0$ 을 테스트 하고자 할 때

모형평가 차트

```R
m <- lm(.....)
par(mfrow=c(2,2))
plot(m)
```



범주형 변수 분석

- 지시변수(indicator variable) 사용하여 범주 표시 
- 일반적으로 범주가 p개일 때, 지시변수 p-1개 사용됨

(주의!) x = {1, 2, 3 } 따위로 해서는 안됨.



### 분산분석 (Analysis of Variance, ANOVA)

실험계획법에서 가장 많이 사용되는 분석방법. 회귀분석이랑 비슷하다.

randomization : 실험을 랜덤하게 하면 전수실험과 같은 효과 가능하다. 그렇게 실험계획 하라. 그 결과는 ANOVA로 분석 해라.

ANOVA : 자료에서 발생하는 변동성을 모형에 의한 변동과 오차에 의한 변동으로 분해하고 비교하여 요인의 유의성을 검증. 회귀분석과 정확히 동일한 컨셉



##### 일원배치법



(참고) 귀무가설이 기각되면, multiple comparison 할 것. 유의수준 쌓이면 alpha가 커짐. 콘트롤 필요하며, 보수적인 방법은 bone paritiy correction. -- 현대는 고차원 데이터에는 맞지 않음. 요즘은 FDR(False discovery rate)사용



##### 이원배치법

반복이 없는 이원배치법

반복이 있는 이원배치법 : 교호작용

---



강의일: 2020-07-15

강사: 정성규 교수 (서울대 통계학과)

# 1. 인자분석

### 주성분분석(Principal Component Analysis)

머신러닝 비지도학습과 같은 컨셉

100년전, 다변량 숫자변수를 행렬대수 기법을 이용하여 차원을 낮추는 기술

주성분 2개를  뽑으면 평면에 그려 눈으로 볼 수 있음

영상인식에 사용할 수 있음

절차

	1. 주어진 자료를 본다
 	2. 좌표를 중심으로 이동시킨다. (평균을 0으로)
 	3. 산포가 최대가 되는 방향을 찾는다 (자료를 가장 잘 설명하는 직선을 긋는다)
 	4. 수직을 찾는다
 	5. 3, 4가 새로운 축이 된다.
 	6. 자료를 x축으로 내리면 차원축소 된다!!

주성분 정의

	- 첫번째 주성분 : 변수들의 전체 분산 중 가장 큰 부분을 설명할 수 있는 선형결합
	- 두번째 주성분 : 첫번째 주성분과 독립이면서 잔여분산을 최대한 설명하는 선형결합

고유값과 고유벡터

- k 차원을 획기적으로 차원을 낮춰주는 값
- 고유값 (lamda) : 설명하는 양
- 고유벡터(e) : 성분



스펙트럴 분해(Spectral Decomposition)

고유값과 고유벡터로 데이터를 분해



rule of thumb : Scree Plot을 찍어 무릅을 찾고 바로 앞까지 선택

80%룰 : 80%변동을 설명할 수 있는 성분까지 선택



공분산 행렬을 이용한 주성분분석의 문제점

- 측정단위에 민감함
- 상관을 1로 맞추고 분석한다. 



데이터가 있다 --> 수학적으로 주성분을 뽑는다. --> 주성분을 선택하고 --> 다음 분석에 인풋으로 넣는다.



### 인자분석(Factor Analysis)

영향을 미치틑 어떤 가상의 요인들이 있음을 믿고 있음 

실제 보이는 변수들이 있음

데이터를 보고 화살표를 찾는 분석론



인자추출 : 주성분방법, 주인자방법, 최대가능도방법(Maximum Likelyhood Method) ... 등으로 뽑는다.

인자회전 : 해석이 용이하도록 축을 회전

- 회전방법 : VARIMAX(더 적은 팩터들과 연관), QUARTIMAX(비뚤게 회전) ... 





# 2. 생존분석

위험함수를 추정해 내면, 결국 생존함수를 알 수 있다.

##### Kaplan-Meier Method (누적한계 추정법)

- 생존시간의 분포형태에 대한 사전믿음 불필요함, 
- 중도절단이 있어도 사용 가능





# 3. 시계열분석

> 트렌드를 제거하고 나온 오차가 의미가 있음. 가까운 과거가 더 의미가 있음.
>
> 목표는 미래 예측
>
> 우선 그림을 그려 형태를 관찰하자

##### 시계열성분(component)

- 성분들로 구성되어서 분해할 수 있다고 믿는다
  - 불규칙성분(irregular component)
  - 체계적성분(systematic component) :  추세성분(trend), 계절성분(seasonal), 순환성분(cyclical)

##### 분해법(decomposition method) 

분해해서 각 성분들을 추정하여 원래 시계열을 해석한다

- 가법모형(additive model)과 승법모형(multiplicative model) 있음. 승법 원하면 로그 취해 가법으로 변환
- `ts(...)` 제너레이트 해서 `decompose(...)` 하면 해결

##### 지수평활법 (exponential smoothing method)

- 40년 역사, 잘 맞지는 않지만 많이 쓴다.

- 먼과거의 가중값을 지수적으로 줄여나가는 방식

- 계산이 쉽다
- parameters (1~0 사이의 값, 1이면 most recent observation에 몰빵)
  - alpha : 평균 통제
  - beta : 추세(기울기) 통제
  - gamma : 계절성분 통제

###### 1. 단순지수평활법

- beta=F, gamma=F... 가장 잘 설명되는 alpha 추정한다(내가 정해도 된다)

###### 2. 이중지수평활법

- gamma=F... alpha와 beta 추정한다

###### 3. Holt-Winters 지수평활법

- alpha, beta, gamma를 추정함