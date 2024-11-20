# TIL
오늘 뭐했지?

## 2024/11/20

TODO
- F1 Score vs AUC 확인 후 학습시킨 모델에 수치 확인
- https://www.youtube.com/watch?v=eDSXcJOCT5k 내용 정리
- 임시 파일 이슈 분석
- Bert 파인튜닝 강의 마무리

### 분류 성능 평가 지표 정리

- 각 용어들 내용 학습 완료

### OOM 이슈 조치

- 파일 전달 시 파일 크기가 커서 OOM 발생
    
    stream을 활용하여 파일 전체를 메모리에 올리지 않도록 개선
    
    ```java
    //AS-IS
    byte[] fileByte = FileUtils.readFileToByteArray(new File(filePath));
    
    response.setContentType("application/zip");
    response.setHeader("Content-Transfer-Encoding", "binary");
    
    response.getOutputStream().write(fileByte);
    response.getOutputStream().flush();
    response.getOutputStream().close();
    ```
    
    ```java
    //TO-BE
    try (InputStream inputStream = new FileInputStream(filePath);
          OutputStream outputStream = response.getOutputStream()) {
    
          // Set HTTP headers for file download
          response.setContentType("application/zip");
          response.setHeader("Content-Transfer-Encoding", "binary");
    
          byte[] buffer = new byte[8 * 1024];
          int bytesRead;
    
          while ((bytesRead = inputStream.read(buffer)) != -1) {
              outputStream.write(buffer, 0, bytesRead);
          }
    
          outputStream.flush();
    ```
    

## 2024/11/19

### 카카오 AI 컨퍼런스 영상 내용 분석( https://www.youtube.com/watch?v=2wsxPekh_ak)

- 라벨링 개선
    - 분류 모델(라벨 추천, AI 참고 영역) + LMM(분류 근거, 이미지 설명)으로 AI 활용
    - LMM에서는 LoRA를 활용한 파인튜닝을 진행하였으나 최근에는 프롬프트 엔지니어링 + Few Shot으로 처리
    - 카테고리 별로 라벨링을 진행하나 다른 카테고리와 score값의 차이가 적을 경우 사람이 교차 검증을 진행한 후 분류모델, 유사도 DB에 반영한다.
- LLM으로 스팸 분류
    - 유해 모니터링
        - 데이터를 수집하는 것이 아닌 실제 사용자 보호를 위한 관리
        - 운영자의 효율성을 위해 유입 컨텐츠에 대한 1차 필터링 진행
            - 운영자의 혼란을 방지하기 위해 분류 사유 함꼐 제공 → **이를 위해 LLM 사용**
            - 프롬프트 엔지니어링을 통해 진행
                - 분류 사유를 출력 후 분류 결과 출력하도록 했더니 **F1-Score 개선**
    - AI 가드레일
        - AI가 생성한 답변의 안정성과 윤리성을 검토하는 시스템
        - AI는 중립적이고 윤리적인 답변만 해야 한다.
        - 허나 다양한(safe + unsafe + normal) 데이터를 포함했을 때 성능이 가장 좋았다.
        → 균형있는 데이터는 분류 성능의 향상으로 이어진다.

### F1-Score부터 시작된 통계 관련 내용 학습

- Accuracy, Precision, Recall, F1-Score, Fall-Out, TruePositive… (별도 글로 정리 예정)

### OOM 이슈 분석 진행

- AWS에서 지속적으로 OOM이 발생하며 서비스 중단되는 이슈 발생하여 분석
    - 파일 전달 시 파일 크기가 커서 OOM 발생하고 있었음
        
        → Buffer를 활용하도록 개선 진행 예정
        
        check) 왜 Buffer를 활용하면 메모리 개선되는지?
        
    - 모니터링 툴로 분석해보니 메모리 사용량이 계속해서 증가하고 있음

Work TODO

- OOM 이슈 조치 - 지속적으로 증가하는 메모리 사용량 원인 확인
- 통계 관련 내용 학습 마무리
- 카카오 컨퍼런스 스미싱 관련 내용 확인 https://www.youtube.com/watch?v=eDSXcJOCT5k

Personal TODO

- 통계 관련 학습 내용 글 작성
- Buffer 파일 전송 관련 글 작성

## 2024/11/18

Bert 학습 방식 - Masked Language Modeling, Next Setence Prediction 동시 진행

Masked Language Modeling

- Input에 random하게 몇 개의 token을 masking
- left, right context만을 가지고 masked word의 원래 단어를 예측

Next Setence Prediction

- training data로 senetence pair(두 개의 문장을 묶어서 만듦) 사용
- 주요 목적은 output의 C([CLS])를 train 시키는 것

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
