# AI Insight Engine

자유 응답 형태의 의견 데이터를 **주제별로 자동 분류하고, 자연어로 검색**하는 웹 앱입니다.
설문 서술형 답변이나 민원 텍스트처럼 "읽어야 할 문장이 수백 건"인 자료를 훑는 데 씁니다.

LLM API를 사용하지 않습니다. 문장 임베딩과 K-Means만으로 동작합니다.

## 기능

| 기능 | 설명 |
|---|---|
| 주제 자동 분류 | 문장을 임베딩해 K-Means로 묶습니다. 주제 개수(k)는 사용자가 조정합니다 |
| Topic Summary | 주제별 의견 수·비율·TF-IDF 키워드·대표 의견을 한 표로 보여줍니다 |
| 적정 k 진단 | k를 2~10으로 바꿔가며 실루엣 점수를 계산해 비교표와 추천값을 보여줍니다 |
| Topic Map | PCA 2D 산점도. 점에 hover하면 원문이 보입니다 |
| Semantic Search | 검색어와 **의미가 비슷한** 의견을 찾습니다. 단어가 달라도 찾습니다 |
| 결과 다운로드 | `id, text, cluster` CSV (UTF-8-SIG, Excel에서 바로 열림) |

## 입력 데이터 형식

CSV 파일이며 **`text` 컬럼이 반드시 있어야 합니다.**

```csv
id,text
1,취업을 준비하면서 지역 청년 입장에서 채용정보가 부족하다.
2,버스 배차간격이 너무 깁니다.
```

- `id` 컬럼은 없어도 됩니다 (자동 생성)
- 인코딩은 UTF-8-SIG / UTF-8 / CP949를 순서대로 시도합니다
- 앞뒤 공백·빈 문자열·중복 문장은 자동으로 제거됩니다

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

첫 실행 시 임베딩 모델(약 500MB)을 내려받으므로 시간이 걸립니다.

## 배포 (Streamlit Community Cloud)

1. 이 저장소를 GitHub에 올립니다
2. [share.streamlit.io](https://share.streamlit.io) → **Create app**
3. 저장소 선택 후 Entrypoint를 `streamlit_app.py`로 지정
4. Deploy → Build log 확인

첫 배포는 모델 다운로드 때문에 오래 걸립니다. 앱이 뜨지 않으면 build log에서
메모리 관련 메시지를 먼저 확인하세요. 무료 티어에서 PyTorch + 임베딩 모델은
메모리 여유가 많지 않습니다.

## 프로젝트 구조

```
ai-insight-engine/
├── streamlit_app.py    # 분석 코어 + UI
├── requirements.txt    # 배포 서버가 설치할 패키지
├── README.md
└── .gitignore
```

`streamlit_app.py`는 두 층으로 나뉩니다.

- **분석 코어** — `read_and_clean_csv()`, `build_analysis()`, `semantic_search()` 등.
  Streamlit에 의존하지 않으므로 노트북에서 그대로 import해 쓸 수 있습니다.
- **UI 레이어** — `render_*()`, `main()`. Streamlit 위젯은 여기에만 있습니다.

## 동작 방식

```
CSV → 정제 → 문장 임베딩 → K-Means → TF-IDF 키워드 → 대표 의견 → PCA 2D
                    ↓
              Semantic Search
```

임베딩 모델은 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
(384차원, 다국어)를 씁니다. 벡터를 L2 정규화하므로 K-Means의 유클리드 거리가
코사인 유사도와 같은 순서를 갖습니다.

## 알려진 한계

이 앱의 출력을 그대로 결론으로 쓰면 안 되는 지점들입니다.

- **주제 개수(k)는 데이터가 정해주지 않습니다.** 실루엣 점수는 보통 작은 k를
  선호하지만, 그 결과가 해석에 쓸모 있다는 뜻은 아닙니다. 최종 판단은 사람이 합니다.
- **클러스터 번호는 임의값입니다.** 재분석하면 같은 주제가 다른 번호를 받습니다.
  번호에 이름을 하드코딩하지 마세요.
- **한 문장에 두 주제가 섞여 있으면 한쪽으로만 배정됩니다.** K-Means는 문장을
  한 클러스터에만 넣기 때문에, 겹치는 의견이 많은 자료일수록 경계가 흐려집니다.
- **Topic Map은 판정 도구가 아닙니다.** 384차원을 2차원으로 눌러 그린 그림이라
  원본 분산의 일부만 반영합니다. 겹쳐 보인다고 분류가 실패한 것은 아닙니다.
  실제 반영 비율은 지도 위 캡션에 표시됩니다.
- **검색은 관련 문장이 없어도 Top-K를 채웁니다.** 유사도 점수는 이 데이터 안에서의
  상대값이라 절대 기준이 없습니다. 무관한 결과가 섞이면 최소 유사도를 올리세요.
- **TF-IDF 키워드는 주제 이름이 아닙니다.** 불용어 목록을 바꾸면 키워드도 바뀝니다.
  한국어 형태소 분석 대신 조사·어미를 규칙으로 떼어내는 방식이라 완전하지 않습니다.

## 보안

이 앱은 API key를 사용하지 않습니다. 이후 LLM API를 붙인다면 키를 코드나
저장소에 직접 넣지 말고 배포 플랫폼의 **Secrets** 기능을 쓰세요.
`.gitignore`가 `.env`와 `.streamlit/secrets.toml`을 제외하도록 설정돼 있습니다.

업로드된 CSV는 서버 메모리에서만 처리되며 저장소에 저장되지 않습니다.
의견 데이터에는 개인정보가 섞일 수 있으므로 `.gitignore`가 `*.csv`를 제외합니다.
