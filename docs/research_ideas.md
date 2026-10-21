# Flow 정책 RL 연구 아이디어 후보

관련 연구 정리와 공백 분석은 [`literature_survey.md`](literature_survey.md)를 참고하세요.

> **출처 확인 방식에 관한 고지.** 이 보고서의 근거인 네 개의 연구 노트는 모두 arxiv.org, openreview.net 등을 직접 열지 못하는 환경에서 작성되었다. 따라서 모든 논문은 **전문(full text)을 읽은 것이 아니라 검색 결과**(arXiv 초록·HTML 발췌, 학회 proceedings·virtual site, 프로젝트 페이지, GitHub README)로만 확인했다. 수치는 모두 "저자 보고값"이며 논문 표와 대조하지 않았다. 학회명은 proceedings, 학회 virtual page, 공식 repo 중 하나로 확인된 경우에만 붙였고, 그렇지 않으면 **"(venue 미확인)"**으로 표시했다. Diffusion-QL, SfBC, QGPO, IDQL, EDP, SRPO, DTQL 같은 2022–2024 기초 논문 일부는 노트 작성자의 배경지식에 근거하며 이번에 재확인되지 않았다. OpenReview 리뷰 본문은 하나도 확보하지 못했으므로, 리뷰어 관점의 비판은 반영되어 있지 않다.


먼저 일정을 짚어 둔다. 오늘(2026-10-21) 기준으로 가장 가까운 현실적 목표는 ICML 2027로 보인다. 정확한 마감일은 노트에 없으니 별도로 확인해야 한다. 이 분야는 한 달에 수십 편이 나오므로, 착수 직전에 각 아이디어의 "가장 가까운 선행연구"를 arXiv에서 다시 검색해야 한다. 특히 2610.x ID 논문들이 그렇다. 아래 계산량은 노트에 근거한 대략값이다. FQL류 1M step 학습은 step당 약 2.65 ms이므로 순수 학습만 약 45분, 평가를 포함하면 GPU당 run 하나에 약 1시간으로 잡았다 ([arXiv 2605.26552](https://arxiv.org/html/2605.26552)). VLA 규모 수치는 노트에 GPU 사양이 없으므로 **우리 추정**임을 밝힌다.

| # | 아이디어 | 설정 | 가장 가까운 선행연구 | 계산량 | 신규성 리스크 | 권장 순위 |
|---|---|---|---|---|---|---|
| 1 | Verifier-robust test-time steering | Offline (+로봇 steering) | SORL, V-GPS, Value Flows, QPILOTS | 낮음 (~0.5–1k GPU-h) | 중 | 1순위 |
| 2 | 표현력 감사와 compute-matched 벤치마크 | Offline/O2O/온라인 | FlowRL taxonomy, ReBRAC-v2 | 중–높음 (~3–5k GPU-h) | 중 (분석 논문의 임팩트 리스크) | 3순위 (1·3의 동반 연구로 적합) |
| 3 | Steer-then-Unfreeze O2O | O2O, 로봇 | DSRL, EXPO, PA-RL, OGPO, Three Regimes | 중 (~1.5–2.5k GPU-h) | 낮음–중 | 1순위 |
| 4 | Mode를 보존하는 RL fine-tuning | 온라인/O2O/VLA | DiverseGRPO, TMPO, DPPO, QAM | 중 (~1–2k GPU-h) | 중 (이식으로 보일 위험) | 2순위 |
| 5 | ODE-consistent 탐색 noise 학습 | 온라인, VLA | ReinFlow, πRL, Uncertainty-Gated, GSB-MDPO | 높음 (VLA 포함 시) | 중–높음 | 2순위 |

## 아이디어 1: Q를 verifier로 쓰는 test-time scaling의 과낙관을 교정한다

**문제와 공백.** 2025–26년 offline flow 정책은 추론 시 계산량으로 성능을 사는 방향으로 가고 있다. SORL은 Q를 verifier로 쓰는 best-of-N을 쓰고 ([SORL](https://nico-espinosadice.github.io/projects/sorl/)), Value Flows는 rejection sampling을 쓴다 ([Value Flows](https://arxiv.org/pdf/2510.07650)). QPILOTS와 test-time gradient guidance는 Q-gradient로 sampler를 유도하며 ([QPILOTS](https://arxiv.org/pdf/2606.14801); [arXiv 2606.11087](https://arxiv.org/pdf/2606.11087)), V-GPS는 범용 로봇 정책의 샘플을 재순위화한다 ([V-GPS](https://proceedings.mlr.press/v270/nakamoto25a.html)). 그런데 오프라인에서 학습된 Q는 OOD action에서 과대추정된다. N을 키울수록 선택은 Q 오류가 가장 큰 action 쪽으로 쏠린다. 일종의 winner's curse다. 이미지 쪽은 best-of-n의 과낙관과 SMC weight degeneracy를 명시적 문제로 다루지만 ([Nested SMC](https://arxiv.org/html/2608.20123); [TRI-TSMC](https://arxiv.org/html/2605.25123)), 정책 RL에는 이에 대한 분석이 없다.

**핵심 아이디어.** 이 연구는 세 단계로 구성된다. 먼저 현상을 측정한다. N(1–256)과 guidance 강도에 따른 실제 return 곡선, 그리고 선택된 action의 Q̂와 Monte Carlo return 사이의 gap을 그려 "verifier 착취 곡선"을 정의한다. 다음으로 N에 의존하는 pessimism을 넣은 선택 규칙을 제안한다. N개 후보 최대값의 기대 과대추정은 극값 이론상 대략 σ·√(2 log N)로 커진다. 따라서 critic ensemble이나 분포형 flow critic(Value Flows, floq)의 불확실성 σ̂에 √(log N)을 곱한 LCB에, FAC가 쓰는 flow behavior density 페널티를 더한 점수로 action을 고른다. 이렇게 하면 N이 커져도 성능이 단조 증가하도록 보정할 수 있다. 마지막으로 이를 denoising 단계별 Q potential을 쓰는 twisted SMC로 확장한다. 중간 단계 value는 BDPO처럼 학습한다. 이로써 best-of-N보다 적은 NFE로 같은 품질을 얻는 것을 목표로 한다.

**신규성.** SORL, V-GPS, Value Flows는 Q를 그대로 신뢰한다. QPILOTS류는 gradient 유도라서 과낙관을 다루지 않는다. 이미지 쪽 SMC 연구는 reward model이 고정되어 있다고 가정하며, bootstrapped Q처럼 외삽 오류가 구조적으로 커지는 verifier는 다루지 않는다. 기여는 "N-scaling 하에서의 offline RL pessimism"이라는 새 축을 이론과 측정으로 정립하는 데 있다. 경험적 관찰만이 아니라 N과 critic 분산의 함수로 과낙관 상한을 제시하는 것까지 포함한다.

**최소 실험 계획.** 벤치마크는 OGBench의 어려운 도메인이다. SORL과 SQAM이 가장 큰 차이를 보인 antsoccer, antmaze-large/giant, humanoidmaze, cube-double, puzzle 계열에서 도메인당 5개 과제를 쓴다. 여기에 D4RL antmaze를 더하고, 로봇 확장으로는 V-GPS 설정이나 Robomimic Can/Square 하나를 쓴다. Base 정책은 FQL의 BC flow, SORL, GFP다. Baseline은 greedy Q best-of-N(SORL식), IDQL/SfBC식 exp(βQ) resampling, Value Flows rejection sampling, Q-gradient guidance(QPILOTS 재현), CFGRL이다. 대조군으로 ReBRAC에도 같은 test-time 계산을 준다(Gaussian에서 N개 샘플). Gaussian으로도 같은 이득이 나는지 확인하기 위해서다. 지표는 N별 success rate 곡선, Q̂−MC gap, decision당 추론 ms와 NFE, 그리고 N=1 성능을 해치지 않는지다. 시드는 8개로 한다.

**예상 위험.** Q가 잘 보정된 과제에서는 이득이 작을 수 있다. 그래서 과제 선택이 결과를 좌우할 수 있으며, 이를 막으려면 과제 전체 결과를 사전에 정한 방식으로 보고해야 한다. ensemble critic은 학습 비용을 2–5배 늘린다. SMC 확장은 중간 단계 Q 학습이 불안정하면 실패할 수 있으므로 별도 단계로 분리해 진행하는 것이 안전하다. 2026년 test-time steering 논문이 3편 이상 나왔으므로, 그중 하나가 이미 pessimism을 다뤘는지 본문을 확인해야 한다.

**대략 계산량.** 정책은 재사용하고, critic ensemble 학습은 run 하나당 FQL 1회 수준(GPU당 약 1시간)으로 본다. 30개 과제 × 8 시드 × base 3종이면 약 720 run, 즉 약 700 GPU-h다. 평가 sweep은 수백 GPU-h면 된다. 단일 노드(8 GPU)로 2–3주 안에 가능하다.

## 아이디어 2: 표현력이 언제 보상으로 이어지는지 compute-matched로 감사한다

**문제와 공백.** 이 분야의 핵심 동기인 "multi-step 표현력"은 측정된 적이 거의 없다. 비교는 FQL 하이퍼파라미터를 상속한 같은 50개 과제에 집중되어 있고, 계산 비용 보고는 일관성이 없다. 온라인 쪽은 vanilla SAC/PPO 대비 이득을 보고한다. FlowRL taxonomy(RLC 2026)가 Gym, DMC, IsaacLab에서 온라인 벤치마크를 표준화했지만 ([FlowRL taxonomy](https://arxiv.org/html/2603.27450v2)), offline/O2O에서의 multimodality 필요성과 강한 Gaussian 대조군 문제는 남아 있다.

**핵심 아이디어.** 두 부분으로 구성된다. 첫째는 **최적 action 집합이 증명 가능하게 multimodal인 진단 과제군**이다. 대칭 미로, 해가 여럿인 puzzle, 서로 다른 최적 전략이 섞인 데이터셋, 품질이 다른 mode가 섞인 데이터셋을 OGBench/D4RL 변형으로 만든다. 둘째는 **표현력 지표**다. 추출된 정책의 mode coverage, 최적 action 집합까지의 Wasserstein 거리, 그리고 "증류 gap"(같은 critic에서 full multi-step flow와 그 one-step 증류의 return 차이)을 정의한다. 이 지표로 계열 A–G의 대표 방법과 Gaussian 방법을 동일 코드베이스, 동일 튜닝 예산, FLOPs·추론 ms 기준 Pareto로 비교한다. 최종 산출물은 데이터셋만 보고 표현력 있는 정책이 필요한지 예측하는 **"multimodality necessity score"**다. 예를 들어 behavior 분포의 고-advantage 영역에서 조건부 mode 수를 세는 방식이 있다.

**신규성.** FlowRL taxonomy는 온라인 표준화이고 표현력 측정을 하지 않는다. ReBRAC-v2는 제목상 Gaussian 경쟁력을 주장하지만 본문을 확인하지 못했다. FQL을 비롯한 개별 논문의 ablation은 자기 방법 중심이다. 동일 예산 메타 비교, 진단 과제군, 예측 지표를 함께 묶은 연구는 노트에서 찾지 못했다. 결과가 "대부분의 과제에서는 Gaussian으로 충분하고, 특정 구조에서만 flow가 필요하다"로 나오더라도 커뮤니티의 평가 관행을 바꾸는 기여가 된다.

**최소 실험 계획.** 벤치마크는 OGBench state 과제 20개 부분집합(난이도별 층화), D4RL antmaze/adroit, 진단 과제 6–10개다. 비교 방법은 FQL, SORL, GFP, FAC, ReFORM, QAM(또는 SQAM), RQL, MeanFlowQL 중 코드가 공개된 6–8개와 Gaussian 계열 ReBRAC, ReBRAC-v2(가능하면), IQL, DDPG+BC다. 선택적인 온라인 부록에서는 SAC Flow, FPO, DPMD를 CrossQ, BRO, SimBa와 같은 UTD·wall-clock으로 비교한다. 지표는 정규화 return, mode coverage, 증류 gap, 학습·추론 FLOPs, 하이퍼파라미터 민감도(튜닝 예산 대비 성능 곡선)다.

**예상 위험.** 분석 논문은 "놀라움"이 없으면 낮게 평가받는다. 일부 2026년 방법은 코드가 없어 재현이 어렵다. 재현 성능이 원 논문보다 낮으면 "구현이 잘못됐다"는 반론을 받으므로, 원 논문 수치의 재현을 사전 조건으로 두어야 한다. ReBRAC-v2 본문과 겹칠 가능성도 확인해야 한다.

**대략 계산량.** 8개 방법 × 26개 과제 × 8 시드 × 튜닝 config 3개면 약 5,000 run이다. JAX vmap으로 GPU당 여러 시드를 돌리면 3–5k GPU-h로 추정한다. 아이디어 1·3과 코드베이스를 공유하면 한계비용이 낮아진다.

## 아이디어 3: 조향으로 시작해 포화를 감지하면 flow를 푼다 (Steer-then-Unfreeze)

**문제와 공백.** O2O에서 고정된 base를 조향하는 방법(DSRL, EXPO, PA-RL, LPS, residual)은 dip이 없지만 base support에 갇힌다. full flow fine-tuning(SAC Flow, ReinFlow, QAM/TRQAM, OGPO)은 가소성이 크지만 불안정하다. OGPO는 바로 이 trade-off를 manifold 확장으로 측정했다 ([OGPO](https://arxiv.org/html/2605.03065)). 노트들은 "latent manifold가 포화되면 가중치 업데이트로 전환하는 hybrid"와 "frozen flow map 아래에서 도달 가능한 분포의 특성화"를 열린 문제로 꼽는다 ([DSRL](https://proceedings.mlr.press/v305/wagenmaker25a.html); [The Gaussian Is Enough](https://arxiv.org/pdf/2609.27070)). Three Regimes 원리는 flow 정책에서 검증되지 않았고 ([Three Regimes](https://arxiv.org/html/2510.01460v3)), flow O2O의 Q overestimation도 분리 측정된 적이 없다.

**핵심 아이디어.** 이론, 알고리즘, 진단 세 부분으로 구성된다. 이론 부분에서는 frozen flow map f_θ: z ↦ a가 유계 source(ReFORM의 hypersphere 등) 위의 diffeomorphism일 때 noise-space 조향이 도달할 수 있는 정책 집합이 정확히 BC support 위의 분포라는 점을 보인다. 그 위에서 조향 최적과 진짜 최적 사이의 gap을 "support 밖 Q 이득"으로 상한한다. 알고리즘 부분에서는 상태별 **포화 통계량**을 정의한다. 예를 들어 critic의 action gradient ∇ₐQ 중 flow map Jacobian J = ∂a/∂z의 열공간에 직교하는 성분의 크기, 즉 조향으로는 따라갈 수 없는 개선 방향을 쓸 수 있다. 처음에는 DSRL식 조향만 학습하고, 통계량이 임계값을 넘는 상태 영역에서만 flow 가중치를 trust region 안에서 업데이트한다. 업데이트에는 QAM이나 FQL식 국소 목표를 쓴다. 진단 부분에서는 Three Regimes 분류로 초기 단계를 고른다. Superior 체제면 긴 조향 단계를, Inferior 체제면 빠른 unfreeze와 data replay를 택한다. 학습 내내 Q̂−MC gap과 manifold 확장을 기록해 dip의 원인을 분해한다.

**신규성.** DSRL, EXPO, PA-RL은 끝까지 동결한다. PLD는 사후에 SFT로 증류할 뿐 RL 중에 전환하지 않는다. OGPO와 SAC Flow는 처음부터 full fine-tuning을 한다. FQL은 증류된 actor만 바꾼다. **원리적인 전환 기준과 도달 가능 집합에 대한 이론**을 함께 제시한 연구는 노트에서 찾지 못했다. 또한 dip, overestimation, manifold 확장을 한 틀에서 측정하는 것 자체가 flow O2O 최초의 체계적 진단이 된다.

**최소 실험 계획.** 벤치마크는 FQL/Q-chunking 프로토콜의 OGBench O2O(1M offline + 1M online), 그중 cube, puzzle, antmaze, humanoidmaze 10개 과제다. 여기에 D4RL antmaze/adroit와 DSRL·OGPO와 직접 비교할 수 있는 pixel Robomimic Can/Square를 더한다. 확장 실험으로 RLinf의 π0 DSRL 구현 위 LIBERO나 ManiSkill 부분집합을 쓸 수 있다. Baseline은 FQL, QC-FQL, DSRL(SAC 및 NA), EXPO, PA-RL, SAC Flow, FINO, OGPO(코드가 있으면), QAM/TRQAM(코드가 있으면)이고, Gaussian 대조군으로 Cal-QL과 RLPD를 쓴다. 지표는 최종 success, dip 크기(온라인 초기 10% 구간의 최저 정규화 성능), 온라인 학습곡선 아래 면적, OGPO식 manifold 확장, Q̂−MC gap, wall-clock이다. Ablation은 포화 통계량 대 고정 시점 전환 대 무작위 전환, 그리고 체제별 결과다.

**예상 위험.** 포화 통계량이 noisy하면 전환 시점이 흔들린다. 전환 자체가 피하려던 dip을 일으킬 수도 있으므로 trust region 크기가 핵심 하이퍼파라미터가 된다. action chunk가 길면 Jacobian 계산 비용이 커지는데, JVP 기반 무작위 투영으로 완화할 수 있다. 잘 튜닝된 QC-FQL 대비 이득이 작을 수 있으므로, 이득이 dip 제거와 sample efficiency에서 주로 나오도록 지표를 설계해야 한다.

**대략 계산량.** state O2O run 하나는 2M step으로 GPU당 약 2시간이다. 10개 과제 × 8 시드 × 약 10개 방법이면 약 800 run, 1.6k GPU-h다. pixel Robomimic은 약 0.5–1k GPU-h를 더한다. VLA 확장은 우리 추정으로 80GB급 GPU 8장 노드에서 2–4주가 든다.

## 아이디어 4: 거친 mode 선택은 지키고 세밀한 정제만 최적화한다

**문제와 공백.** 이미지 쪽은 KL로 diversity collapse를 막을 수 없다는 증거를 쌓았다. Flow-GRPO의 ablation, DiverseGRPO, TMPO, DFR, GRPO-Guard가 그 예다 ([Flow-GRPO](https://arxiv.org/pdf/2505.05470); [DiverseGRPO](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_DiverseGRPO_Mitigating_Mode_Collapse_in_Image_Generation_via_Diversity-Aware_GRPO_CVPR_2026_paper.pdf); [DFR](https://arxiv.org/html/2610.24897)). 정책 RL에서는 multimodal BC 정책이 RL fine-tuning 중 붕괴하는지 측정한 논문이 없다. RLDT의 주장과 multi-task task collapse 보고만 있다 ([RLDT](https://arxiv.org/abs/2606.08602); [2609.28838](https://arxiv.org/pdf/2609.28838)). QAM, AMDP, MaxEnt 계열은 명목상 tilted 분포 π_0·exp(Q/λ)를 목표로 하므로 mode를 보존해야 한다. 그러나 실제로는 λ가 작고 critic이 편향되어 있어 붕괴할 가능성이 높다.

**핵심 아이디어.** 첫째, 측정 프로토콜을 만든다. trajectory clustering으로 mode 수를 세고, 상태별 action entropy를 추정하고, 학습 중 mode 생존 곡선을 그린다. 이것으로 FQL, ReinFlow, πRL, QAM, SAC Flow의 fine-tuning 중 붕괴를 정량화한다. 둘째, **denoising 시간에 따라 역할을 분리하는 정규화**를 제안한다. 초기 denoising 단계(τ가 noise 쪽)는 어떤 mode를 고를지 결정하므로, 여기에는 base 정책과의 Wasserstein·path-KL 제약을 강하게 건다. 후기 단계는 정밀 위치를 정하므로 Q를 자유롭게 최적화한다. 이는 DiverseGRPO의 단계 의존 제약과 DPPO가 마지막 몇 단계만 fine-tune하는 관행을 원리적으로 통합한다. 셋째, mode 내부의 개선은 허용하고 mode 간 질량 이동은 advantage 차이에 비례하도록 제한하는 분포 매칭 목표를 둔다. 이는 TMPO의 아이디어를 Q-tilted target에 맞게 바꾼 것이다.

**신규성.** DiverseGRPO와 TMPO는 고정 reward model의 이미지 생성 문제이고, 정책 RL에 대응물은 노트에서 발견되지 않았다. DPPO의 부분 단계 fine-tuning은 효율을 위한 장치였지, mode 보존 이론이 아니다. 핵심 기여는 "mode 선택 대 정제"라는 denoising 시간 분해를 정책 RL에서 정식화하고, 보존된 mode가 실제로 **하류 이득**을 준다는 것을 보이는 데 있다. 하류 이득의 후보는 섭동 강건성, 새 goal과 reward로의 빠른 적응, multi-task collapse 완화다.

**최소 실험 계획.** 벤치마크는 ReinFlow 설정의 Robomimic과 Franka Kitchen, DDiffPG식 multimodal 과제, OGBench O2O 일부, ManiSkill multi-task다. 강건성 평가는 LIBERO-Plus 섭동으로 한다 ([LIBERO-Plus](https://arxiv.org/pdf/2510.13626)). Baseline은 ReinFlow, DPPO(마지막 k단계), FPO, SAC Flow, FQL, QAM, FINO이고, Gaussian PPO/SAC 대조군을 둔다. 지표는 success, mode 수와 entropy, 섭동 하 success, goal 이동 후 적응 step 수, multi-task 과제별 최소 성능이다.

**예상 위험.** 가장 큰 위험은 "보존된 다양성이 return으로 이어지지 않는다"는 반론이다. 그래서 하류 과제가 연구의 성패를 결정한다. DiverseGRPO를 단순히 옮겨 왔다는 인상을 피하려면 Q-tilted target 아래의 이론(mode 간 질량 이동 상한)이 필요하다. 초기와 후기 단계의 경계 τ\*도 과제마다 달라질 수 있다.

**대략 계산량.** Robomimic과 Kitchen의 on-policy fine-tuning은 run당 수 GPU-h로 추정한다(ReinFlow 규모, 정확한 수치는 노트에 없음). 6개 과제 × 5 시드 × 8개 방법에 OGBench 부분집합을 더하면 1–2k GPU-h다.

## 아이디어 5: 탐색 noise를 정책의 일부로 학습하고 배포 ODE와의 gap을 묶는다

**문제와 공백.** deterministic ODE는 탐색을 막는다. 현재 해법은 학습된 noise network, 고정 σ의 ODE→SDE, σ schedule 같은 임시방편이다 ([πRL](https://arxiv.org/html/2510.25889v2)). 그 결과 평가조차 확률적 sampler로 해야 하는 불일치가 생긴다(σ_eval = 0.25) ([2609.28838](https://arxiv.org/pdf/2609.28838)). Adjoint Matching 이론에 따르면 memoryless가 아닌 schedule은 target을 편향시키고 schedule lock-in을 일으킨다. 반대로 memoryless schedule은 과탐색을 일으킨다 ([Adjoint Matching](https://arxiv.org/abs/2409.08861); [arXiv 2605.06583](https://arxiv.org/pdf/2605.06583)).

**핵심 아이디어.** 상태와 denoising 시간에 의존하는 noise schedule σ_φ(s, τ)를 정책 파라미터로 둔다. 목적함수는 RL 목표에 탐색 SDE 정책과 배포 ODE 정책 사이의 path-KL 페널티를 더한 형태다. GSB-MDPO가 보인 대로 path-KL은 marginal KL의 상한이다. performance difference lemma와 Pinsker 부등식을 결합하면 |J(π_SDE) − J(π_ODE)| ≤ C·√(path-KL) 형태의 배포 gap 상한을 얻을 수 있다. 학습이 진행될수록 페널티 가중치를 올려, 최종 정책을 deterministic하게 배포해도 성능 손실이 상한 안에 들도록 한다. σ_φ의 크기는 critic ensemble의 불확실성이 큰 상태에서만 커지도록 정규화한다. 이렇게 하면 multi-task에서 이미 해결된 과제의 noise가 줄어 task collapse를 완화할 수 있다. 변형으로, 탐색 sampler와 최적화 경로를 완전히 분리하는 forward-process 업데이트(DiffusionNFT/RoboFFT식, solver에 무관)를 ablation으로 비교한다.

**신규성.** ReinFlow와 πRL Flow-Noise는 noise를 학습하지만 배포 ODE와의 gap을 목적함수로 통제하지 않는다. GSB-MDPO의 path-KL은 업데이트 iterate 사이의 proximal 항이지 탐색과 배포 사이의 항이 아니다. FINO의 entropy-guided sampling은 O2O의 탐색 폭을 다룰 뿐 배포 gap 보장이 없다. 다만 가장 가까운 2609.28838(Uncertainty-Gated Exploration Noise)은 gating 메커니즘을 확인하지 못했으므로, 착수 전에 본문을 반드시 읽어야 한다. 겹친다면 배포 gap 이론과 solver 무관 변형으로 차별화해야 한다.

**최소 실험 계획.** from-scratch 실험은 MuJoCo, DMC-hard, HumanoidBench 일부에서 한다. 비교 대상은 FPO, ReFPO, SAC Flow, DACER, DPMD, ReinFlow이고, Gaussian 대조군 CrossQ, BRO, SimBa를 반드시 포함한다. fine-tuning 실험은 Robomimic과 Franka Kitchen에서 ReinFlow, DPPO와 비교한다. VLA 실험은 RLinf 위 π0 ManiSkill multi-task 부분집합에서 πRL Flow-SDE(고정 σ), Flow-Noise와 비교한다. 지표는 deterministic 배포 success와 확률적 평가 success의 차이(배포 gap), sample efficiency, multi-task 과제별 collapse율, 학습된 σ의 상태별 분포다.

**예상 위험.** 상한이 느슨하면 이론의 실질적 의미가 약해진다. 2609.28838과 겹칠 위험이 가장 크다. VLA 실험은 계산량 때문에 시드 수가 부족해질 수 있다. state 과제에서는 Gaussian 대조군이 비슷한 성능을 낼 수 있으므로, 이득을 multimodal·multi-task 설정에서 보이도록 설계해야 한다.

**대략 계산량.** state 과제는 약 500–800 GPU-h다. VLA 실험은 노트에 하드웨어 수치가 없으므로 우리 추정으로 8-GPU 노드 수 주가 필요하며, 이 아이디어에서 가장 무거운 부분이다. 계산 자원이 제한적이면 VLA를 빼고 Robomimic 수준에서 제출하는 것이 현실적이다.

