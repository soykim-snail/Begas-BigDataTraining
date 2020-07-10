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