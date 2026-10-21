# flowRL

Flow matching / diffusion 기반 생성 모델과 강화학습(RL)을 결합하는 연구 저장소입니다.
목표는 **ICML / ICLR** 수준의 논문으로 이어질 아이디어를 발굴하고 검증하는 것입니다.

## 연구 범위

### 주력: Flow matching / diffusion 정책을 쓰는 RL
- 오프라인 RL, 온라인 RL, offline-to-online 파인튜닝에서의 표현력 높은 (multimodal) 정책
- Q-가이던스, 정책 경사, 증류(distillation) 등 생성 정책을 보상/가치 방향으로 최적화하는 방법
- 로보틱스 / VLA의 flow-matching action head에 대한 RL 개선

### 보조: Flow 모델 자체를 RL로 파인튜닝
- 이미지·비디오 등 생성 모델의 reward fine-tuning (정책 경사 계열, 확률적 최적 제어 계열)
- 위 주력 주제와 공유되는 핵심 문제: 다단계 샘플링 과정의 최적화, likelihood 계산의 어려움, 탐색/엔트로피 제어, few-step 추론

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `docs/literature_survey.md` | 관련 연구 조사 정리 |
| `docs/research_ideas.md` | 연구 공백 및 아이디어 후보 |

실험 코드와 결과는 아이디어가 정해지는 대로 추가합니다.
