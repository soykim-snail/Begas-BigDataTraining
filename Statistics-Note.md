Lectures date 2020-07-14

lecturer 오희석 교수 (서울대 통계학과 )

---

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