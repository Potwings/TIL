# TIL
오늘 뭐했지?

## 2024/11/14

Trasformers를 통해 모델을 불러올 때 두가지 종류가 존재함. (AutoModel, AutoModelfor…..)

AudoModel - 다양한 세팅을 통하여 원하는 목적에 맞춰 사용 가능(nn.Module 상속)

AutoModelFor…. - 목적에 맞춰 사전 학습된 모델을 가져와 사용 가능
(ex. AutoModelForSequenceClassification)

두 모델의 내용을 확인해보면 마지막 층에서 차이가 존재함.

```python
# AutoModel
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
```

```python
# AutoModelForSequenceClassification
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
  # classifier에서 class 개수만큼 output 반환(여기서는 2)
)
```

Model에 input_ids와 attention_mask 입력 시 반환해주는 값도 차이가 존재한다.

AutoModel은 last_hidden_state, pooler_output을 반환해주는데 이 값들을 활용하여 원하는 label을 생성하는데 사용한다.

AutoModelForSequenceClassification의 경우 구해야하는 label이 지정되어있으므로 예측 값과의 오차를 제공해준다.

Pytorch의 nn.Module

init - 상속받는 모듈의 기능을 가져옴. 상속과 foward의 세팅을 담당하는 코드들이 포함

foward - 변수들을 내보내게싼 의미. 어떤 입력들을 주면 모델에서 특정 계산에 의해 출력하는 과정들을 포함.

진행 과정

1. init에 선언한 모델에 입력을 주어 output 계산
2. 1번의 결과에 첫번째값(CLS)의 pooler_output 가져온 후 dropout 적용하여 classifier에 입력
    - label이 존재할 경우 (training / validation) → loss 계산
    - label이 존재하지 않을 경우 (test) → 예측 결과 반환
