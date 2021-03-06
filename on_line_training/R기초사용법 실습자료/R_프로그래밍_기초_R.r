#============================================================
# title: "R 프로그래밍 기초"
#============================================================

# getwd 함수를 사용하여 현재 작업공간 위치를 알 수 있음
print(getwd())

# setwd 함수를 사용하여 사용자가 사용할 작업 공간 위치를 설정 할 수 있음
# setwd('경로설정')

# *************************
# 제 3 장 R 데이터 구조 ***
# *************************

#============================================================
# 벡터(vector) 데이터 구조 학습 
#============================================================

x <- 10
y <- c(1, 2, 3, 4, 5)
z <- c(x, y)
str(z)
length(z)

# 숫자형 벡터와 문자형 벡터가 결합되면 문자형으로변환된다. 
a <- c('one', 'two', 'three')
d <- c(y, a)
str(y)
str(d)

# 논리형 벡터
b <- c(TRUE, FALSE, TRUE)
str(b)

# 지정된 값만큼 증감하는 데이터 생성
seq(1, 7, by=2)   # from, to, by, length, along
seq(1, -1, by=-0.5)
seq(1, 7, length=3)
rev(seq(1:5))   # rev : 자료의 순서를 역순으로 만드는 함수, 5:1

# 값이 반복되는 데이터 생성
rep(c(1,2,3), 3)  # rep(a, b)는 a를 b만큼 반복
rep(1:3, 3)      # a:b는 a부터 b까지의 수
rep(c(4, 2), times=2)
rep(c(4, 2), times=c(2, 1))
paste('no', 1:3)     # 반복되는 문자에 첨자를 붙여줌

vec1 <- c(1, 2, 3, 4, 5)  #1~5까지 자료를 갖는 vec1 변수 생성
vec1[2]     # 두 번째 자료
vec1[c(2, 3, 5)]    # vec1의 2, 3, 5의 값만 표현
vec1[c(-2, -3)]    # vec1의 2, 3번째 자료 값 삭제
vec1[vec1 > 2]    # vec1에서 2보다 큰 값만 표현
vec1[2] <- 6    # 두 번째 위치의 2값이 6으로 대체됨

replace(vec1, 3, 2)   # vec1의 세 번째 자료를 2로 변경  replace(벡터, 위치, 값)
append(vec1, 8, after=5)    # vec1의 5번째 자료 다음에 8을 삽입   append(벡터, 값, after=위치)


#============================================================
# 행렬(Matrix) 데이터 구조 학습 
#============================================================

matrix(1:9, nrow=3)                            # nrow  : 행의 개수 지정
matrix(c(1, 4, 7, 2, 5, 8, 3, 6, 9), byrow=T, ncol=3)  # ncol : 열의 개수 지정,  byrow=T : 행 기준 행렬을 생성
r1 <- c(1, 4, 7)                                 #r1, r2, r3 행 벡터 생성
r2 <- c(2, 5, 8)
r3 <- c(3, 6, 9)
rbind(r1, r2, r3)                                # rbind : 행을 기준으로 결합
cbind(r1, r2, r3)                                # cbind : 열을 기준으로 결합

m1 <- 1:9
dim(m1) <- c(3, 3)

#행렬과 관련된 여러 함수와 성분의 추출과 삭제 등에 관해 알아봄
mat <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9), ncol=3, byrow=T) #행 기준 3열의 행렬 생성
mat[1,]                                       #행렬 mat의 1행의 값
mat[,3]                                       #행렬 mat의 3열의 값
mat[mat[,3] > 4, 1]                        #3열에서 4보다 큰 행의 값 중 1열의 모든 값
mat[mat[,3] > 4, 2]                        #3열에서 4보다 큰 행의 값 중 2열의 모든 값
mat[2,, drop=F]                            #2행 값만을 행렬 형태로 추출
is.matrix(mat[2,, drop=F])                #mat[2,,drop=F]가 행렬인지 확인

#============================================================
# 배열(Array) 데이터 구조 학습
#============================================================

#배열을 생성하기 위한 함수로 array() 함수와 dim() 함수가 있음
array(1:6)                                      #1~6의 자료로 1차원 배열 생성
array(1:6, c(2, 3))                             #1~6의 자료로 2차원 배열 생성
array(1:8, c(2, 2, 2))                          #1~8의 자료로 3차원 배열 생성
arr <- c(1:24)                                 #1~24의 자료 생성
dim(arr) <- c(3, 4, 2)                        #dim() 함수를 이용하여 3행 4열의 행렬 2개 생성

#배열의 연산
ary1 <- array(1:8, dim = c(2, 2, 2))  
ary2 <- array(8:1, dim = c(2, 2, 2))
ary1 + ary2                                    #자료의 덧셈
ary1 * ary2                                     #자료의 곱셈
ary1 %*% ary2                                #두 배열 원소들의 곱의 합
sum(ary1 * ary2)                              #ary1 %*% ary2 와 같은 결과를 냄

#배열원소의 추출 및 삭제
ary1[,,1]
ary1[1,1,]
ary1[1,,-2]

#============================================================
# 리스트(List) 데이터 구조 학습 
#============================================================

lst <- list('top', c(2, 4, 6), c(T, F, T))              #list(문자, 숫자, 논리형 객체) 
lst[[1]]                                                  #[[1]] 첫 번째 성분
lst[1]                                                    #[1] 첫 번째 리스트
mat1 <- matrix(1:4, nrow=2)
list1 <- list('A', 1:8, mat1)
son <- list(son.name = c('Minsu', 'Minchul'), son.cnt = 2, son.age = c(2, 6))

#리스트 속성 : 벡터의 속성과 같이 자료의 개수, 형태, 구성요소의 이름 등을 보여주는 length, mode, names로 구성
length(son)                                   #son 리스트 자료의 개수
mode(son)                                    #son 리스트 자료의 형태
names(son)                                   #son 리스트 각 구성요소의 이름

#예제1
a <- 1:10
b <- 11:15                                     
klist <- list(vec1=a, vec2=b, descrip='example')
klist[[2]][5]                                      #두 번째 성분 vec2의 5번째 원소
klist$vec2[c(2, 3)]                             #vec2의 2, 3번째 원소

#============================================================
# 데이터프레임(Data Frame) 데이터 구조 학습 
#============================================================

#data.frame() : 이미 생성되어 있는 벡터들을 결합하여 데이터 프레임을 생성
char1 <- rep(LETTERS[1:3], c(2, 2, 1))          #벡터 char1
num1 <- rep(1:3, c(2, 2, 1))                      #벡터 num1
test1 <- data.frame(char1, num1)              #test1 데이터 프레임 생성

#as.data.frame() :모든 다른 종류의 자료객체들을 데이터 프레임으로 변환
a1 <- c('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o')
dim(a1) <- c(5,3)
test3 <- as.data.frame(a1)                        #a1 메트릭스를 데이터 프레임으로 변환

df1 <- data.frame(Col1 = c('A', 'B', 'C'), Col2 = c(1, 2, 3), Col3 = c(3, 2, 1))
#df1[행, 열]
df1[ , 'Col3']    #결과 : 3, 2, 1 출력
df1[1, ]           #결과 : A  1  3 출력
df1[3, 'Col1']    #결과 : C 출력


# *************************
# 제 4 장 R 데이터 수집 ***
# *************************

#============================================================
# Sample Data 생성 및 저자
#============================================================

n <- 1e6
DT <- data.frame( a=sample(1:1000, n, replace=TRUE),
                  b=sample(1:1000, n, replace=TRUE),
                  c=rnorm(n),
                  d=sample(c('foo', 'bar', 'baz', 'qux', 'quux'), n, replace=TRUE),
                  e=rnorm(n),
                  f=sample(1:1000, n, replace=TRUE) )

# txt 파일로 저장하기 예시  1 
write.table(DT, 'C:\\Temp\\테스트\\test.txt', sep=',', row.names=FALSE, quote=FALSE)
                           
# CSV 파일로 저장하기 예시  2
write.csv(DT, 'C:\\Temp\\테스트\\test.csv', row.names=FALSE, quote=FALSE)

#============================================================
# 외부 데이터 불러오기
#============================================================

# read.csv()로 CSV 파일 가져오기 예시 1 
DF1 <-read.csv('C:\\Temp\\테스트\\test.csv', stringsAsFactors=FALSE)

# read.table()로 txt 파일 가져오기 예시 2
DF2 <- read.table('C:\\Temp\\테스트\\test.txt', header=TRUE, sep=',', quote='',
                            stringsAsFactors=FALSE, comment.char='', nrows=n,
                            colClasses=c('integer', 'integer', 'numeric', 'character', 'numeric', 'integer'))
                            
# fread()로 CSV 파일 가져오기 예시 3 : data.table::fread()로 csv file import.
install.packages("data.table")
library(data.table)
DT1 <- fread('C:\\Temp\\테스트\\test.csv')

# read.csv.sql()로 CSV 파일 가져오기 예시 4 : sqldf::read.csv.sql()로 csv file import
install.packages("sqldf")
library(sqldf)
SQLDF <- read.csv.sql('C:\\Temp\\test.csv', dbname=NULL)


# *************************
# 제 5 장 R 데이터 가공 ***
# *************************

#============================================================
# 연산ㅈ
#============================================================

# 산술 연사
1 + 3     # 1과 3 더하기
6 - 1      # 6에서 1 빼기
3 * 9       # 3과 9 곱하기
21 / 7      # 21을 7로 나누기
30 %/% 8      # 30을 8로 나눈 몫
30 %% 8        # 30을 8로 나눈 나머지
8^2       # 8의 2 제곱

# 비교 연산
7 > 5        # 7은 5보다 크다 
3 >= 6      # 3은 6보다 크거나 같다
4 < 3        # 4는 3보다 작다
2 <= 9       # 2는 9보다 작거나 같다
6 == 6       # 6은 6과 같다
6 != 6         # 6은 6과 같지 않다

# 논리 연산
x<-1:5
y<-5:1

# x 값이 3보다 크고 y 값이 1보다 크면 TRUE, 아니면 FALSE
(x > 3) & (y > 1)   

# x 값이 3보다 크거나 y 값이 2보다 크면 TRUE, 아니면 FALSE
(x > 3) | (y > 2)

#============================================================
# 변수명 변경하기
#============================================================

library(dplyr)
library(MASS)

data(Boston)
str(Boston)
Boston_rename <- rename(Boston, room = rm, Y = medv)
View(Boston_rename)

#============================================================
# 파생 변수 생성하기
#============================================================

Boston$rm_6 <- ifelse(Boston$rm >= 6, "lot", "few")

#============================================================
# 조건문(if/ifelse)
#============================================================

x <- c(1, 2, 3, 4)
y <- c(2, 1, 4, 5)
if(sum(x) < sum(y)) print(x)     #x의 합이 y의 합보다 작을 경우 실행

#if(조건) {
#    조건이 True 일때 실행문
# } else {
#    조건이 False 일때 실행문
# }
x <- c(1,2,3,4)
y <- c(2,1,4,5)
if(mean(x)>mean(y)) print('Mean(x)>Mean(y)') else print('Mean(x)<Mean(y)')

#============================================================
# 필요한 데이터 추출
#============================================================

# select() 함수 사용

# Boston 데이터 셋에서 여러 변수들 중 지정한 rm 변수만 추출
# dplyr 패키지의 select() 함수를 사용한다는 것을 명시적으로 표현
Boston %>% dplyr::select(rm)

# Boston 데이터 셋에서 rm, lstat, medv 변수 추출
Boston %>% dplyr::select(rm, lstat, medv)

# Boston 데이터 셋에서 지정한 rm 변수만을 제외하고 추출
Boston %>% dplyr::select(-rm)  

# Boston 데이터 셋에서 rm, lstat 변수를 제외하고 추출
Boston %>% dplyr::select(-rm, -lstat)

# filter() 함수 사용

# Boston 데이터 셋에서 rm 변수가 6 이상인 것만 추출
Boston %>% filter(rm > 6)

# Boston 데이터 셋에서 lstat 변수는 모집단의 하위계층의 비율(%)이다. 평균 12.65
# Boston 데이터 셋에서 rm 이 6 이상이면서 lstat 가 12 이상인 경우만 추출  
Boston %>% filter(rm > 6 & lstat > 12)

#============================================================
# 데이터 정렬
#============================================================

# arrange() 함수

Boston %>% arrange(ptratio)

# 비율이 높은 지역에서 부터 낮은 지역으로 정렬
Boston %>% arrange(desc(ptratio))

#============================================================
# 결측값 (missing value)
#============================================================

x<-c(1, 2, 3, NA, 5)                        # 벡터 x에 결측값 할당
x
x*2                                             # 결측값이 있으면 연산을 해도 결과가 NA
is.na(x)                                         # 벡터 x에서 결측값 조회
table(is.na(x))                                 # 벡터 x의 결측값 건수 조회
sum(x)                                          # 벡터 x의 합계는 결측값이 있어서 결과가 NA
sum(x, na.rm=T)                              # 결측값을 제외하고 합계를 구함

#============================================================
# 가로 데이터를 세로로 전환하는 melt() 함수
#============================================================

library(reshape2)

data(airquality)
head(airquality)

# 기준이 되는 열을 지정하지 않으면 모든 열을 반환한다.
aq_melt1 <- melt(airquality)

#  기준이 되는 식별자로 Month와 Day를 지정하고 Ozone 값을
#  파악할 수 있도록 데이터를 생성
aq_melt2 <- melt(airquality, id.vars=c("Month", "Day"), measure.vars = "Ozone")

#============================================================
# 세로 데이터를 가로로 전환하는 cast() 함수
#============================================================

library(reshape2)

aq_melt <- melt(airquality, id.vars=c("Month", "Day"), na.rm=TRUE)

# 월(Month), 일(Day)을 기준으로 Ozone, solar.r 등을 행으로 변환
aq_dcast <- dcast(aq_melt, Month+Day~ variable)

# 월, 일별로 variable 열에 있는 Ozone, Solar.R, Wind, Temp 순서로 배열을 생성
acast(aq_melt, Day ~ Month ~ variable)

#============================================================
# mutate() 함수로 열 추가
#============================================================

library(dplyr)

# 데이터를 측정한 년도 열을 추가
aq_mutate <- mutate(airquality, years="2018")

# Temp의 평균 값 79.00 보다 높으면 High, 낮으면 Low 를
# 나타내는 Temp_HL 열 추가
aq_mutateHL <- mutate(aq_mutate, Temp_HL = ifelse(Temp > 79, "High", "Low"))

#============================================================
# distinct() 함수로 중복값 제거
#============================================================

# airquality 데이터셋에서 온도(Temp)에 따른 종류를 확인
# 중복된 값을 제거하면 해당 열이 총 몇 가지 관측기로 구성되어 있는지 확인할 수 있다.
distinct(airquality, Temp)

#============================================================
# 사용자 함수 정의
#============================================================

mean_fn <- function(data) {
  result <- sum(data) / length(data)
  return(result)
}

x <- rnorm(100, mean = 4, sd = 1)
mean_fn(data = x)

round2 <- function(x, n) {
  sing.x <- sign(x)
  z <- abs(x)*10^n
  z <- z + 0.5
  z <- trunc(z)
  z <- z/10^n
  result <- z*sing.x
  
  return(result)
}

x.c <- c(1.75, 1.54, 1.55, 1.85, 1.65)    # 벡터 x.c 생성

round(x.c, 1)    # R 의 {base} 함수를 사용한 결과

round2(x.c, 1)   # 사용자 정의 함수 round2() 를 사용한 결과

# *************************
# 제 6 장 R 그래프 함수 ***
# *************************

#============================================================
# 산점도 그래프
#============================================================

plot(rnorm(100)) #100개의 난수에 대하여 산점도 생성
plot(rnorm(100), main = 'Test Graph', xlab = 'index', ylab = 'value')
plot(rnorm(100), ann = FALSE)
title(main = 'Test Graph', xlab = 'index', ylab = 'value')
with(iris, plot(Petal.Length, Petal.Width, pch = as.integer(Species)))
legend(1.5, 2.4, c('setosa', 'versicolor', 'virginica'), pch = 1:3)
plot(iris[, c('Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width')])

#============================================================
# 막대 그래프
#============================================================

hight_x <- c(120, 125, 130.5, 138, 142, 150)
barplot(hight_x)
barplot(hight_x, main = '초등학생 평균 키', names.arg = c('1학년', '2학년', '3학년', '4학년', '5학년', '6학년'), ylab = '평균키(Cm)')

#============================================================
# 선 그래프
#============================================================

hight_x <- c(120, 125, 130.5, 138, 142, 150)
class_x <- c(1, 2, 3, 4, 5, 6)
plot(class_x, hight_x, type = 'l', main = '초등학생 평균 키', ylab = '평균키(Cm)', xlab = '학년')

#============================================================
# 히스토그램
#============================================================

data(Cars93, package  = 'MASS')
hist(Cars93$MPG.city, main = 'City MPG (1993)',  xlab= 'MPG')

hist(Cars93$MPG.city, breaks = 20,main = 'City MPG (1993)',  xlab= 'MPG')

#============================================================
# 박스 플롯
#============================================================

x<-c(2, 5, 8, 5, 7, 10, 11, 3, 4, 7, 12, 15)
z<-c(3.5, 2.2, 1.5, 4.6, 6.9)

boxplot(x,z)

#또는 

tmp_df <- data.frame(gubun = c(rep(1, 12), rep(2, 5)), value = c(x, z))
boxplot(value ~ gubun, data = tmp_df)



