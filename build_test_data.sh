#!/bin/bash

python3 precompute_test_inputs/build_test_data.py
python3 precompute_test_inputs/build_test_data_features.py

## 1. 공개된 모델을 20%의 테스트 데이터셋에서 돌려서 결과를 뽑아봄.
## 결과가 우리 것이 좋으면 바로 논문을 진행.
## 2. test data를 100%를 다 뽑아서 거기서 evaluation을 진행.
## 20%를 써도 compare를 진행할 수 있다.