1. R 4.0.2 설치
2. Rtools 40 (x86_64)설치 (3.6.x 버전은 Rtools 36)
   - https://cran.r-project.org/bin/windows/Rtools/
3. Anaconda3 설치(2020.02 ver)
   - https://www.anaconda.com/products/individual
4. Anaconda3 가 설치된 폴더의 전체 권한 허용
5. Anaconda prompt 관리자 권한 실행
6. pip install --upgrade pip로 python pipeline 최신버전 업데이트 
7. conda update --all로 모든 라이브러리 업데이트
8. 텐서플로우 설치(gpu 없으므로 only cpu ver) : pip install tensorflow-cpu (2.2.0 ver)
9. https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads  에서 vc_redist.x64.exe 설치(설치 후 재부팅)
10. 첨부한 R code 실행