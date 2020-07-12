# Today I learned : R 사용법

Goal : being fluent in using R studio

## Data Structure

* Matrix :  2 dimensional data structure with elements of **identical type.** 

* Array : higher dimensional version of matrix, if you can imagine higher than 3 dim.
  * `array1 * array2` : multiple by element
  * `array1 %*% array2` : sum of multiples
  * `sum(array1 * array2)`
* List : [[]]  vs. []

## Data import

- `read.table` : a lot of options to be described, like comment.char, nrows, colClasses, ...
- `read.csv`
- `data.table::fread` : fast reading big size data
- `sqldf::read.csv.sql` : connection be open to SQLite. 한글경로 인식못함.

## Data Managing

- `reshape2::melt` : 테이블을 녹여내어 세로로 긴 데이터로 뽑아낸다.

  ![](https://ae01.alicdn.com/kf/HTB1q6b3JpXXXXcBXVXXq6xXFXXXL/mini-induction-gold-melting-furnace-for-2kg-4kg-6kg-8kg.jpg)

  - `melt(<테이블 데이타>, id.vars=<id로 사용할 변수>, [ measure.vars=<표출할 변수> ], [na.rm=True])`

- `reshape2::dcast`
  - `dcast(<녹은 데이타> <기준으로 잡는 변수>~<표출할 변수>)` : 가로로 긴 테이블이 된다.
  - `acast(<녹은 데이타> <행기준 변수>~<열기준 변수>~<표출할 변수>)` : 다차원 array가 된다.

## Graph

- 산점도 그래프
  - `plot`
  - `with`
- 막대 그래프
  - `barplot`
- 선 그래프
  - `plot(<x축 데이타>, <y축 데이타>, type = 'l')`

- 히스토그램
  - `hist(<데이타>, breaks=<무엇이 올까요??>)`

- 박스 플롯
  - `boxplot(<데이터들>)` 또는 `boxplot(<value>~<구분>, data)`