# flowrl: flow 정책 RL 실험 코드

FQL 이후 SOTA 후보들을 **같은 코드베이스, 같은 학습 루프**에서 비교하기 위한 실험 기반입니다. JAX/Flax 기반이며 OGBench(offline, offline-to-online)를 대상으로 합니다.

## 출처

| 구성 | 출처 | 커밋 | 라이선스 |
|---|---|---|---|
| 프레임워크 전체와 기존 에이전트 | [ColinQiyangLi/qam](https://github.com/ColinQiyangLi/qam) (QC 코드베이스 기반) | `2726d767` | MIT ([LICENSE](LICENSE)) |
| `agents/rql.py`, `Dataset.sample_traj` | [aoberai/rql](https://github.com/aoberai/rql) | `229c956e` | MIT ([LICENSE-RQL.md](LICENSE-RQL.md)) |

QAM 저장소의 `assets/`, `exp_data/`는 가져오지 않았습니다.

## 포함된 에이전트

`--agent=agents/<name>.py`로 고릅니다.

| 이름 | 방법 |
|---|---|
| `qam` | Q-learning with Adjoint Matching (ICLR 2026). `QAM`, `QAM_FQL`, `QAM_EDIT` 변형은 [experiments/reproduce.py](experiments/reproduce.py) 참고 |
| `rql` | Reversal Q-Learning (arXiv 2606.17551) |
| `fql` | Flow Q-Learning (Q-chunking 버전) |
| `ifql`, `fbrac`, `fawac`, `fedit`, `bam`, `cgql`, `dcgql` | QAM 논문의 flow 기반 비교 방법 |
| `dsrl` | latent noise 조향 (DSRL) |
| `rebrac`, `rlpd` | Gaussian/deterministic 대조군 |

## upstream 대비 변경점

- **RQL 이식**: loss와 네트워크는 원본 그대로입니다. 이 코드베이스에 맞추려고 바꾼 것은 네 가지입니다.
  - `sample_actions(observations, rng, temperature)`로 다른 에이전트와 시그니처를 맞췄습니다.
  - 온라인 UTD 업데이트용 `batch_update`를 추가했습니다.
  - config에 `batch_format='traj'`를 넣어, 학습 루프가 `Dataset.sample_traj`로 시간축 우선(time-major) 배치를 뽑게 했습니다.
  - `create`가 넘겨받은 config를 복사해서 씁니다. upstream처럼 원본에 `discount_mul`(jnp 배열)을 써 넣으면 `main.py`의 `flags.json` 저장이 실패합니다.
- **`utils/datasets.py`**: RQL용 `sample_traj`를 추가했습니다. `ReplayBuffer.add_transition`은 에피소드가 끝날 때 경계(`terminal_locs`, `initial_locs`)를 갱신합니다.
- **`main.py`**
  - `batch_format='traj'`인 에이전트는 오프라인·온라인 모두 `sample_traj`를 씁니다. 이때 `--balanced_sampling`은 지원하지 않습니다.
  - config에 `online_temperature`가 있으면 온라인 탐색 시 그 값을 `sample_actions`에 넘깁니다. RQL 원본은 온라인에서 temperature=1, 즉 EMA가 아닌 actor로 탐색합니다.
  - wandb는 기본으로 꺼져 있습니다(`--wandb_mode=disabled`). 결과는 항상 `save_dir` 아래 CSV로 남습니다. 켜려면 `--wandb_mode=online` 또는 `offline`을 주고, 프로젝트 이름은 `flowrl`입니다.
- **`log_utils.py`**: 최신 wandb(0.31)가 거부하는 `wandb.Settings(start_method=...)`를 뺐습니다.

## 설치와 실행

Python 3.11을 권장합니다. 3.13에서는 `labmaze` 휠이 없어 `ogbench` 설치가 실패했습니다.

```bash
cd flowrl
pip install -r requirements.txt   # GPU: pip install "jax[cuda12]"

# RQL, offline (하이퍼파라미터는 upstream hyperparameters.sh 기준)
MUJOCO_GL=egl python main.py --agent=agents/rql.py --env_name=cube-double-play-singletask-v0 \
  --agent.alpha=10.0 --agent.expectile=0.9 --agent.ensemble_ct=10 --agent.rho=0.5 --agent.h=5 \
  --offline_steps=1000000 --online_steps=0

# QAM, offline-to-online
MUJOCO_GL=egl python main.py --agent=agents/qam.py --env_name=cube-triple-play-singletask-task2-v0 \
  --horizon_length=5 --agent.action_chunking=True --offline_steps=1000000 --online_steps=1000000
```

RQL의 과제별 하이퍼파라미터는 [aoberai/rql의 hyperparameters.sh](https://github.com/aoberai/rql/blob/main/hyperparameters.sh)에 있습니다. QAM 쪽은 [experiments/reproduce.py](experiments/reproduce.py)에 있습니다.

## 테스트

```bash
cd flowrl
JAX_PLATFORMS=cpu pytest tests/test_agents_smoke.py   # 에이전트 13종: 생성, 오프라인 업데이트, UTD 업데이트, 행동 샘플링

# RQL 이식이 upstream과 같은 값을 내는지 확인
git clone https://github.com/aoberai/rql /tmp/rql && git -C /tmp/rql checkout 229c956efb4494c2b9bb0bbddbd67b761c93f1cc
RQL_UPSTREAM=/tmp/rql JAX_PLATFORMS=cpu pytest tests/test_rql_port.py
```

`test_rql_port.py`는 같은 합성 데이터에서 다음 항목을 upstream과 비교합니다. 허용 오차는 1e-6입니다.
- 샘플된 trajectory 배치
- loss와 각 loss 항목
- 업데이트 1회 후의 모든 파라미터
- 샘플된 행동

## 검증 상태와 한계

- **확인함 (CPU, 2026-10-21)**
  - 스모크 테스트 14개 통과
  - RQL 이식과 upstream의 결과 일치
  - 실제 OGBench `cube-double` 환경에서 RQL 평가 루프가 1 에피소드 돌아감
- **아직 확인하지 못함**
  - **실제 학습 실행과 논문 수치 재현**: 개발 환경에서 OGBench 데이터셋 호스트(`rail.eecs.berkeley.edu`) 접속이 막혀 있었고 GPU도 없었습니다. GPU 머신에서 먼저 RQL과 QAM의 논문 수치를 과제 몇 개로 재현해 이식을 검증한 뒤 본 실험에 들어가는 것을 권합니다.
  - **RQL 온라인 단계**: upstream은 action chunk를 감싼 환경 wrapper로 transition을 저장하고, 이 코드는 QAM 학습 루프의 단일 step transition을 씁니다. 그래서 RQL의 offline-to-online 결과는 upstream과 다를 수 있습니다. upstream의 하이퍼파라미터 파일도 모두 `--online_steps=0`입니다.
  - **SQAM, TRQAM**: 공개된 코드를 찾지 못해 포함하지 않았습니다.
