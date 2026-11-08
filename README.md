# KLUE-NER 한국어 개체명 인식 (klue/bert-base 파인튜닝)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uisang2004-maker/klue-ner-bert/blob/main/ner_klue_bert.ipynb)

한국어 NER 벤치마크 **KLUE-NER**로 `klue/bert-base`를 파인튜닝해, 문장 속 개체를 찾아 유형을 분류하는 프로젝트입니다.
글자 단위 레이블을 서브워드 토큰에 맞추는 전처리부터 토큰 단위 손실, 개체 단위 F1 평가까지 PyTorch로 직접 구현했습니다.

> **입력** 세종대왕은 1443년에 훈민정음을 창제했다.
> **출력** 세종대왕(PS), 1443년(DT)

| 유형 | 의미 | | 유형 | 의미 |
|---|---|---|---|---|
| PS | 인명 | | DT | 날짜 |
| LC | 지명 | | TI | 시간 |
| OG | 기관 | | QT | 수량 |

## 결과

> 3 epoch 학습까지 완료했으며, 검증·테스트 F1과 유형별 결과는 평가 셀을 포함해 다시 실행한 뒤 추가할 예정입니다.

## 구현 내용

| 단계 | 내용 |
|---|---|
| 데이터 | KLUE-NER parquet 파일을 pandas로 로드. 정답이 공개된 validation을 test로 쓰고, train의 10%를 검증 데이터로 분리 |
| 레이블 정렬 | 레이블은 **글자**마다, 모델 예측은 **서브워드 토큰**마다라서 단위가 다름. 토크나이저의 `return_offsets_mapping`으로 토큰마다 원문 글자 위치를 받아, **토큰의 첫 글자 레이블**을 토큰 레이블로 사용 |
| 특수 토큰 처리 | `[CLS]`, `[SEP]`, `[PAD]`는 오프셋이 `(0, 0)`이므로 레이블을 **-100**으로 두어 손실 계산에서 제외 |
| 모델 | `AutoModelForTokenClassification` (BERT의 모든 토큰 벡터 위에 13개 BIO 태그 분류 층) |
| 손실 | `(batch, seq_len, 13)` logits를 `(batch × seq_len, 13)`으로 펼쳐 토큰 단위 cross entropy 계산 |
| 평가 | 태그 O가 대부분이라 정확도는 의미가 약해서, `seqeval`의 **개체 단위 F1** 사용. 손실은 실제 토큰 수로 가중 평균 |
| 학습 | AdamW, 선형 warmup 스케줄러, 혼합 정밀도(AMP) + `GradScaler` |
| 추론 | 토큰별 BIO 태그를 오프셋으로 원문에 되짚어 `(개체 문자열, 유형)`으로 묶는 `extract_entities` 구현 |

### 문서 분류와의 차이

[NSMC 감성 분류 프로젝트](https://github.com/uisang2004-maker/nsmc-sentiment-classification)와 같은 학습 파이프라인에서, 예측 단위가 문장에서 토큰으로 바뀌며 달라진 부분입니다.

| | 문서 분류 | 개체명 인식 |
|---|---|---|
| 예측 단위 | 문장 하나에 레이블 하나 | 토큰마다 레이블 하나 |
| BERT 출력 | `[CLS]` 위치의 문장 벡터 | 모든 토큰의 벡터 시퀀스 |
| logits 모양 | `(batch, 2)` | `(batch, seq_len, 13)` |
| 평가 지표 | 정확도 | 개체 단위 F1 |

## 구현하며 찾은 버그

| 버그 | 증상 | 원인과 해결 |
|---|---|---|
| 레이블 정렬 조건 반전 | 실제 토큰이 전부 -100이 되고 특수 토큰에만 레이블이 붙어, **모델이 실제 글자를 하나도 배우지 못함** (에러 없음) | `start == end`(특수 토큰) 분기와 일반 토큰 분기가 뒤바뀌어 있었음. 조건을 바로잡고, 작은 입력으로 기대값을 확인하는 테스트로 검증 |
| 검증 분리가 `if` 블록 안에 들어감 | 기본 설정에서 `val_df`가 만들어지지 않아 `NameError` | 선택적인 학습 데이터 축소와 항상 실행되는 검증 분리를 별도 단계로 분리. 문서 분류 때와 같은 실수라 이후 체크리스트에 추가 |
| 개체 묶기의 초기값 | `UnboundLocalError` | 반복문 전에 아직 없는 반복 변수로 `current`를 만들고 있었음. "만들고 있는 개체 없음"을 뜻하는 `None`으로 시작 |
| 노트북에 남은 전역 변수 | 함수 안 변수명 오타가 에러 없이 이전 셀의 같은 이름 변수를 참조해, 결과가 빈 리스트가 됨 | 함수 안 변수명을 반복 변수와 일치시키고, 짧은 변수명이 만드는 혼동을 기록 |

## 하이퍼파라미터

| 항목 | 값 |
|---|---|
| Pretrained model | `klue/bert-base` |
| Max length | 128 |
| Batch size | 32 |
| Learning rate | 5e-5 |
| Warmup ratio | 0.1 |
| Epochs | 3 |
| Seed | 42 |

## 실행 방법

1. 위의 **Open in Colab** 배지를 누릅니다.
2. **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다. CPU로 실행하면 매우 느립니다.
3. 셀을 위에서부터 차례로 실행합니다. 데이터와 모델은 자동으로 내려받습니다.

## 다음 단계

- 유형별 F1을 비교하고, 틀린 예측을 경계 오류(개체 범위)와 유형 오류로 나눠 분석
- 첫 글자 레이블 방식이 개체 경계를 어긋나게 만드는 사례 확인
- `klue/roberta-base`와 성능 비교, epoch 수에 따른 검증 F1 변화 확인

## 참고

- 데이터: [KLUE-NER](https://huggingface.co/datasets/klue/klue) (Korean Language Understanding Evaluation)
- 모델: [klue/bert-base](https://huggingface.co/klue/bert-base)
- 『Do it! BERT와 GPT로 배우는 자연어 처리』 6장 개체명 인식을 참고했습니다.
