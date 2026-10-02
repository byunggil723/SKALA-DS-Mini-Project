# Data

이 폴더에는 MIT-Stanford Battery Dataset의 MATLAB v7.3/HDF5 원본 파일을 둔다. `.mat` 파일은 총 8GB 이상이므로 `.gitignore`에서 제외된다.

## 다운로드

프로젝트 루트에서:

```bash
python data/download_data.py
```

다운로드 파일:

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2018-02-20_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

완료된 파일은 예상 byte 크기를 비교해 자동으로 건너뛴다. 중간에 끊긴 파일은 HTTP Range가 지원되면 이어받는다.

`2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat`은 선택 추가 EDA 배치다. 이 파일이 없어도 세 개 기본 배치로 전처리·EDA·모델링을 수행할 수 있다.

원본 데이터: <https://data.matr.io/1/>
