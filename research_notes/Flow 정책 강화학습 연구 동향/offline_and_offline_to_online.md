# Diffusion and Flow-Matching Policies for Offline RL and Offline-to-Online RL (2022–2026)

> Method note (read first): direct fetches of arxiv.org, openreview.net, and mlanthology.org were blocked in this environment (DNS failure / proxy 403). Paper existence, venue, and claims for 2025–2026 work were checked with multiple web searches that returned arXiv, OpenReview, ICLR/ICML virtual-site, PMLR, and GitHub pages. Snippet-level verification is weaker than reading the full PDF, so treat specific numbers as "reported by the authors, not re-checked against tables". Foundational 2022–2024 papers (Diffusion-QL, SfBC, QGPO, IDQL, EDP, Consistency-AC, SRPO, DTQL) are well-established; their arXiv IDs are given from background knowledge and were not re-fetched here. Each such item is tagged [bg]. Items marked "venue unverified" have a real arXiv ID found in search but no venue confirmation.

## Q1. Main method families and how each extracts a Q-maximizing policy while staying close to the data

### Takeaway
There are about six ways to extract a policy: (1) backprop Q through the sampler, with BC on the diffusion model (Diffusion-QL lineage); (2) sample from a BC generative model, then pick or reweight by Q (SfBC, IDQL, Value Flows, SHARSA); (3) guide the sampler with energy, Q-gradients, or CFG (QGPO, CFGRL, QPILOTS); (4) weight the generative training loss by advantage or energy (QIPO, GFP, AWR-style); (5) distill into a one-step actor or train one directly, regularized toward the BC flow (SRPO, DTQL, FQL, OFQL, SSCP, MeanFlowQL, SORL); and (6) treat denoising as an MDP or stochastic control problem and train the full multi-step policy without BPTT (BDPO, QAM, RQL). The field moved from (1) in 2022–23 to (5) in 2024–25, and as of 2026 it is trying (6) to recover full expressivity without BPTT.

### Cited Findings

**Family A: Q-gradient through the diffusion chain plus BC loss (Diffusion-QL lineage)**
- Diffusion-QL (Wang, Hunt, Zhou; ICLR 2023) is the first widely used "diffusion policy as actor" method. Its loss is the diffusion denoising (BC) loss plus −α·Q(s, a), where a is sampled by running the full reverse chain, so gradients backprop through every denoising step. [bg] — [arXiv 2208.06193](https://arxiv.org/abs/2208.06193)
- EDP, "Efficient Diffusion Policies" (Kang et al.; NeurIPS 2023), avoids running the full chain in training through "action approximation" (one-step reconstruction of a from a noised dataset action) and uses fast ODE solvers. It also makes diffusion policies compatible with likelihood-based methods (IQL/CRR-style extraction). [bg] — [arXiv 2305.20081](https://arxiv.org/abs/2305.20081)
- Consistency-AC (Ding & Jin; ICLR 2024) replaces the diffusion model with a consistency model so Q-gradients pass through 1–few steps. [bg] — [arXiv 2309.16984](https://arxiv.org/abs/2309.16984)
- The 2025 consensus critique: "a long denoising chain hinders stable gradient propagation from actions to policy parameters when optimizing against some value function" (motivation of EXPO). — [EXPO, arXiv 2507.07986](https://arxiv.org/pdf/2507.07986)
- FQL frames this as "unstable recursive backpropagation" (BPTT) through a multi-step flow. — [FQL, PMLR v267](https://proceedings.mlr.press/v267/park25f.html)

**Family B: BC generative model plus Q-based selection (rejection / best-of-N / importance resampling)**
- SfBC (Chen et al.; ICLR 2023) learns a high-fidelity diffusion behavior model, samples N candidate actions, and resamples them with weights ∝ exp(β·Q). The policy is never trained against Q. [bg] — [arXiv 2209.14548](https://arxiv.org/abs/2209.14548)
- IDQL (Hansen-Estruch, Kostrikov, Janner, Kuba, Levine; arXiv 2023) trains IQL critics, then reweights or selects samples from a diffusion BC policy using IQL's implicit policy weights. It decouples actor and critic. [bg] — [arXiv 2304.10573](https://arxiv.org/abs/2304.10573)
- Value Flows (2025) uses "rejection sampling to maximize Q estimates while implicitly imposing a KL constraint toward a fixed flow behavioral cloning (BC) policy" for offline extraction. Its contribution is the critic: a flow-matching model of the full return distribution. — [Value Flows, arXiv 2510.07650](https://arxiv.org/pdf/2510.07650); [project page](https://pd-perry.github.io/value-flows/); [code](https://github.com/chongyi-zheng/value-flows)
- SORL (NeurIPS 2025) uses the learned Q as a verifier for best-of-N "parallel scaling" at test time. — [SORL project page](https://nico-espinosadice.github.io/projects/sorl/); [OpenReview](https://openreview.net/forum?id=uVarpp7fhU)

**Family C: Guided sampling (energy guidance, Q-gradient guidance, classifier-free guidance)**
- QGPO / CEP (Lu et al.; ICML 2023) defines the target policy as π ∝ μ(a|s)·exp(βQ). It learns an exact intermediate-time energy model with "contrastive energy prediction" so the guidance at every noise level is exact rather than approximate. [bg] — [arXiv 2304.12824](https://arxiv.org/abs/2304.12824)
- CFGRL, "Diffusion Guidance Is a Controllable Policy Improvement Operator" (Frans, Park, Abbeel, Levine; 2025; venue unverified) uses classifier-free guidance on an optimality/advantage condition as a tunable policy-improvement knob. — [arXiv 2505.23458](https://arxiv.org/pdf/2505.23458)
- 2026 test-time steering papers: QPILOTS, "Efficient Test-Time Q-Steering for Flow Policies" — [arXiv 2606.14801](https://arxiv.org/pdf/2606.14801); "Test-Time Gradient Guidance of Flow Policies in RL" — [arXiv 2606.11087](https://arxiv.org/pdf/2606.11087); "Aligning Flow Map Policies with Optimal Q-Guidance" — [arXiv 2605.12416](https://arxiv.org/html/2605.12416). Venues unverified.

**Family D: Weighted generative training (advantage/energy-weighted diffusion or flow matching)**
- QIPO, "Energy-Weighted Flow Matching for Offline RL" (Zhang et al.; ICLR 2025, venue from background knowledge, not re-verified) weights the flow-matching loss by exp(βQ) to learn the energy-guided target without an auxiliary energy model. [bg] — [arXiv 2503.04975](https://arxiv.org/abs/2503.04975)
- Guided Flow Policy (GFP; Tiofack, Le Hellard, Schramm, Perrin-Gilbert, Carpentier; ICLR 2026) couples a multi-step flow policy with a distilled one-step actor. The actor steers the flow policy through weighted BC that emphasizes high-value dataset actions, rather than treating all dataset actions equally as standard behavior regularization does. — [arXiv 2512.03973](https://arxiv.org/abs/2512.03973); [GitHub (lists ICLR 2026 acceptance)](https://github.com/Simple-Robotics/guided-flow-policy)

**Family E: One-step or distilled actors regularized toward a diffusion/flow behavior model**
- SRPO (Chen et al.; ICLR 2024) trains a deterministic (Gaussian-like) actor to maximize Q, regularized by the pretrained diffusion behavior model's score, so no sampling chain runs at inference. [bg] — [arXiv 2310.07297](https://arxiv.org/pdf/2310.07297)
- DTQL, "Diffusion Policies creating a Trust Region" (Chen, Wang, Zhou; NeurIPS 2024) trains a one-step actor with a diffusion-based trust-region loss that keeps it in high-density behavior regions. [bg] — [arXiv 2405.19690](https://arxiv.org/abs/2405.19690)
- FQL, Flow Q-Learning (Park, Li, Levine; ICML 2025, PMLR 267) is now the reference baseline. It trains a BC flow policy plus a separate one-step noise-conditioned policy μ(s, z) that maximizes Q, with a distillation loss toward the BC flow's ODE output. This avoids BPTT and iterative test-time generation "yet still mostly maintains expressivity". It was evaluated on 73 state- and pixel-based OGBench and D4RL tasks. — [PMLR](https://proceedings.mlr.press/v267/park25f.html); [OpenReview PDF](https://openreview.net/pdf/2f8a68d734574ab1a2831d9dd874defe1d7f2d0e.pdf); [arXiv 2502.02538](https://arxiv.org/pdf/2502.02538)
- Critiques of FQL in follow-ups: "the distillation process still requires repeated queries to the underlying multi-step diffusion or flow model"; FQL "performs a one-step approximation of the complex action distribution and limits the representational capability of flow-based policies"; and "distillation-based approaches... sacrifice model expressivity in the final one-step policy". — [MeanFlowQL, arXiv 2511.13035](https://arxiv.org/pdf/2511.13035); [Bootstrapped FQL, arXiv 2606.10613](https://arxiv.org/html/2606.10613v1); [Q-Flow, arXiv 2605.13435](https://arxiv.org/html/2605.13435v2)

**Family F: Exact behavior regularization along the diffusion path / denoising-as-MDP / stochastic optimal control**
- BDPO, Behavior-Regularized Diffusion Policy Optimization (Gao et al.; ICML 2025) decomposes the KL behavior penalty along the diffusion generation path so exact regularization is tractable. It keeps value functions for intermediate diffusion steps to amortize optimization. Evaluated on 2D synthetic tasks and D4RL. Code at typoverflow/flow-rl. — [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44003); [arXiv 2502.04778](https://arxiv.org/pdf/2502.04778)
- QAM, Q-learning with Adjoint Matching (2026; venue unverified) applies adjoint matching (stochastic optimal control fine-tuning) to optimize the multi-step flow policy directly against Q, with local supervision at each denoising step and no separate one-step policy. — [arXiv 2601.14234](https://arxiv.org/pdf/2601.14234); [alphaXiv](https://www.alphaxiv.org/abs/2601.14234)
- RQL, Reversal Q-Learning (Oberai, Park, Levine; 2026; venue unverified) treats each flow refinement step as an action in an expanded MDP. It generates virtual on-policy trajectories by "reversing" flows from dataset actions and adds bias/variance reduction. The authors say it avoids BPTT, "makes better use of the learned value function", and trains the full flow policy directly. — [arXiv 2606.17551](https://arxiv.org/html/2606.17551)
- DAC, Diffusion Actor-Critic (Fang et al.) formulates constrained policy iteration as diffusion noise regression. [bg; venue not verified in this session] — [arXiv 2405.20555](https://arxiv.org/abs/2405.20555)

**Family G: Support-constrained / noise-space steering (new 2025–26 family)**
- ReFORM (Zhang, So, Ahmad, Yu, Cleaveland, Black, Fan; MIT-REALM; ICLR 2026) trains a BC flow from a bounded hypersphere source, then a second flow steers the source noise, so outputs stay on the data support by construction. The authors say this enforces a support constraint (looser than a density/KL constraint) "without hyperparameter-heavy regularization". A final distillation gives a one-step policy. — [mlanthology ICLR 2026](https://mlanthology.org/iclr/2026/zhang2026iclr-reform/); [project page](https://mit-realm.github.io/reform/)
- DSRL, "Steering Your Diffusion Policy with Latent Space RL" (Wagenmaker et al.; 2025; venue unverified) runs RL in the diffusion policy's latent noise space and leaves the base policy frozen. — [arXiv 2506.15799](https://arxiv.org/pdf/2506.15799)
- Latent Policy Steering (LPS; 2026) backpropagates critic gradients through a one-step MeanFlow policy to update a latent-space actor, and uses Q-chunking critics. — [arXiv 2603.05296](https://arxiv.org/pdf/2603.05296)

**Critic-side regularization using the flow behavior model**
- FAC, Flow Actor-Critic (Chae, Park, Shin, Kim, Han, Sung; KAIST/UNIST; ICLR 2026) uses the flow behavior model's density as a critic penalty on OOD regions, in addition to using it as an actor anchor, with a one-step flow actor. The authors claim SOTA on D4RL and OGBench. — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10006693); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c4e7c50f41480db1de200689c9c8f32d-Abstract-Conference.html); [arXiv 2602.18015](https://arxiv.org/html/2602.18015v1)

**Survey**
- "Diffusion Models for Reinforcement Learning: A Survey" (Zhu et al.; 2023) is the general survey of diffusion as planner, policy, and data synthesizer. [bg] — [arXiv 2311.01223](https://arxiv.org/abs/2311.01223)

### Inferences
- A useful taxonomy axis is where the constraint lives: in the training loss (A, D, E), in the sampling procedure (B, C), in the parameterization or support (G), or exactly along the generative path (F). The 2026 trend is toward constraints by construction (ReFORM's support, RQL/QAM's on-path supervision) and away from hand-tuned α.
- The "expressivity vs. trainability" tension (BPTT vs. one-step distillation) is the single most cited motivation in 2025–26 papers. Whether one-step actors actually lose meaningful expressivity on standard benchmarks is still argued mostly with toy 2D examples, not demonstrated on real tasks (see Gaps).

### Gaps
- I could not open full PDFs, so exact loss forms for the 2026 papers (QAM, RQL, DriftQL, Bootstrapped FQL) come from abstract-level summaries.
- The venue of DAC and of QIPO was not re-verified in this session.

## Q2. 2025–2026 papers that advance the area, and what is new in each

### Takeaway
2025 was the "flow + one-step" year: FQL (ICML'25) became the de facto baseline, joined by SORL (shortcut models, NeurIPS'25), BDPO (ICML'25), Value Flows / floq (flow critics), and Q-chunking (NeurIPS'25). ICLR 2026 brought a wave of FQL successors: GFP, FAC, ReFORM, OFQL, SSCP, FINO, floq. Through mid-to-late 2026, arXiv activity centers on (i) MeanFlow-based one-step policies, (ii) training the full multi-step flow without BPTT (QAM, TRQAM, SQAM, RQL), (iii) drift/repulsion policies, and (iv) test-time Q-steering.

### Cited Findings

**2025 (published venues)**
- FQL (ICML 2025): one-step policy plus BC-flow distillation; 73 tasks on OGBench and D4RL. — [PMLR v267](https://proceedings.mlr.press/v267/park25f.html)
- BDPO (ICML 2025): exact pathwise KL regularization for diffusion policies. — [ICML 2025](https://icml.cc/virtual/2025/poster/44003)
- SORL, "Scaling Offline RL via Efficient and Expressive Shortcut Models" (Espinosa-Dice, Zhang, et al.; NeurIPS 2025):
  - Single-stage training with shortcut models (self-consistency: one big jump ≈ two small jumps) and no distillation stage.
  - Supports any inference budget, including one step.
  - Test-time scaling is sequential (more steps) or parallel (best-of-N with Q as verifier). Extra inference compute can make up for a smaller training budget, with the largest gains on AntSoccer and AntMaze-Large.
  - Stated cost: longer training runtime than the simplest one-step flow baselines, due to the extra consistency losses.
  - Sources: [OpenReview](https://openreview.net/forum?id=uVarpp7fhU); [project page](https://nico-espinosadice.github.io/projects/sorl/); [arXiv 2505.22866](https://arxiv.org/pdf/2505.22866)
- Q-chunking, "Reinforcement Learning with Action Chunking" (Li, Zhou, Levine; NeurIPS 2025): runs RL over action chunks for offline-to-online, which gives temporally coherent exploration and unbiased n-step backups. Variants QC-FQL and QC-IFQL are built on FQL and IFQL; the authors say the FQL-based variant is better. Evaluated on six long-horizon sparse-reward domains. — [arXiv 2507.07969](https://arxiv.org/pdf/2507.07969); [project page](https://colinqiyangli.github.io/qc/)
- MeanFlowQL, "One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow" (Wang et al.; arXiv Nov 2025; an AAAI OJS copy exists, so likely AAAI 2026): a residual reformulation of MeanFlow that folds velocity and noise-to-action into one network and is trained jointly with Q without distillation. Evaluated on 73 tasks: 50 state and 5 visual OGBench tasks across 10 environments, plus 18 D4RL antmaze/adroit tasks. Offline and offline-to-online. Code at HiccupRL/MeanFlowQL. Stated overhead: JVPs in training. — [arXiv 2511.13035](https://arxiv.org/abs/2511.13035); [AAAI OJS PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39885/43846)
- Value Flows (2025; venue unverified): a flow-matching distributional critic plus rejection-sampling extraction. Authors claim it matches or beats all baselines on 9 of 11 domains and gets 1.6× higher success on harder state-based tasks. — [arXiv 2510.07650](https://arxiv.org/pdf/2510.07650)
- floq, "Training Critics via Flow-Matching for Scaling Compute in Value-Based RL" (Agrawalla et al.; ICLR 2026 poster): the critic is a flow, so critic compute scales through integration steps. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007626); [arXiv 2509.06863](https://arxiv.org/pdf/2509.06863)
- CFGRL (2025): CFG as a controllable policy-improvement operator. — [arXiv 2505.23458](https://arxiv.org/pdf/2505.23458)

**ICLR 2026**
- GFP, Guided Flow Policy: value-weighted BC on the flow plus a one-step actor. Authors report SOTA across 144 state- and pixel-based tasks from OGBench, Minari, and D4RL, with the largest gains on suboptimal datasets. — [arXiv 2512.03973](https://arxiv.org/abs/2512.03973); [GitHub](https://github.com/Simple-Robotics/guided-flow-policy)
- FAC, Flow Actor-Critic: flow-density critic penalty. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006693)
- ReFORM: reflected flows with on-support noise manipulation. Authors report it outperforms baselines on performance profiles over 40 OGBench tasks, against baselines with hand-tuned hyperparameters. — [project](https://mit-realm.github.io/reform/)
- OFQL, "One-Step Flow Q-Learning: Addressing the Diffusion Policy Bottleneck in Offline RL" (ICLR 2026; earlier arXiv title "Revisiting Diffusion Q-Learning: From Iterative Denoising to One-Step Action Generation"): learns an average-velocity field so that one-step generation is accurate without distillation or auxiliary modules. It argues that "conventional flow matching frequently yields curved trajectories, limiting one-step inference accuracy". — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10011406); [arXiv 2508.13904](https://arxiv.org/html/2508.13904v3)
- SSCP, "Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning" (Koirala & Fleming; ICLR 2026): an augmented flow-matching objective predicts completion vectors from intermediate flow samples, which enables one-shot action generation. — [mlanthology](https://mlanthology.org/iclr/2026/koirala2026iclr-flowbased/)
- FINO (Shin, Chae, Park, Sung; ICLR 2026): offline-to-online; see Q3. — [OpenReview 6wd38R8L0Z](https://openreview.net/forum?id=6wd38R8L0Z); [GitHub](https://github.com/CTID282/FINO)

**ICML 2026 (confirmed)**
- FAN, "Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning" (Lee et al.; ICML 2026 poster): targets the high compute cost of combining flow policies with distributional critics. — [ICML 2026](https://icml.cc/virtual/2026/poster/61532); [arXiv 2605.01663](https://arxiv.org/html/2605.01663v1)

**2026 arXiv (venue unverified unless stated)**
- QAM (arXiv 2601.14234): adjoint matching on the multi-step flow against Q. A secondary source reports an aggregate score of 44 for QAM vs. 35 for a baseline ("BAM") over 50 OGBench tasks (1M steps, 12 seeds); this is not verified against the paper. The submission date is inconsistent: alphaXiv lists 18 May 2026, while the ID implies Jan 2026 (probably v1 vs. a later version). — [arXiv](https://arxiv.org/pdf/2601.14234); [alphaXiv](https://www.alphaxiv.org/abs/2601.14234)
  - TRQAM, "Trust Region Q Adjoint Matching": adds a trust-region parameter to the SOC sampling process to fight critic-driven instability; claims consistent wins on 50 OGBench tasks, offline and offline-to-online. — [arXiv 2605.27079](https://arxiv.org/abs/2605.27079)
  - SQAM, "Q-Learning with Scalar Adjoint Matching": replaces per-step VJPs with a closed-form scalar adjoint plus a value penalty. Claims an 18–35 pp success-rate gain over the best baseline on the four hardest OGBench domains (a secondary news source attributes it to KAIST). — [arXiv 2610.10437](https://arxiv.org/html/2610.10437); [aiweekly](https://aiweekly.co/alerts/kaist-sqam-beats-ogbenchs-hardest-tasks-by-18-35-points)
- RQL, Reversal Q-Learning (Oberai, Park, Levine): authors report the best average offline performance on 50 simulated robotic tasks against SOTA flow-based offline RL. — [arXiv 2606.17551](https://arxiv.org/html/2606.17551)
- MeanFlow family:
  - MF2B, "From Suboptimal to Optimal: One-Step Latent Flow Policies" (Lu et al.): MeanFlow on latent decision chunks, framed as "quality uplift"; claims SOTA on D4RL with one-step inference. — [arXiv 2610.16984](https://arxiv.org/html/2610.16984)
  - DMPO, "One Step Is Enough: Dispersive MeanFlow Policy Optimization": dispersive regularization plus RL fine-tuning, framed toward online fine-tuning. — [arXiv 2601.20701](https://arxiv.org/abs/2601.20701)
  - MVP, "Mean Flow Policy with Instantaneous Velocity Constraint": authors admit it is harder to train than a standard flow policy. — [arXiv 2602.13810](https://arxiv.org/html/2602.13810v2)
  - MFPO, Mean Flow Policy Optimization (Dong, Zhang, Cheng): online-oriented. — [arXiv 2604.14698](https://arxiv.org/pdf/2604.14698)
  - SOM, "Score-Based One-step MeanFlow Policy Optimization": evaluated only on MuJoCo. — [arXiv 2605.23365](https://arxiv.org/html/2605.23365v1)
  - VGM²P (offline MARL) — [arXiv 2604.08174](https://arxiv.org/html/2604.08174v1); CoFlow (offline MARL, few-step) — [arXiv 2605.01457](https://arxiv.org/pdf/2605.01457)
- Drift-based one-step policies:
  - Drift Q-Learning: a drift-based behavior regularizer plus critic-driven improvement, motivated by avoiding iterative denoising, ODE integration, and distillation. — [arXiv 2606.00350](https://arxiv.org/pdf/2606.00350)
  - Drifting Field Policy — [arXiv 2605.07727](https://arxiv.org/pdf/2605.07727)
  - Value-Guided Repulsion Q-Learning: points out that standard drift repulsion is purely geometric and cannot tell actions apart by value, so it adds Q into the drift interaction term. — [arXiv 2610.20190](https://arxiv.org/html/2610.20190)
- Other 2026 flow-policy papers:
  - Bootstrapped Flow Q-Learning — [arXiv 2606.10613](https://arxiv.org/html/2606.10613v1)
  - Q-Flow, "Stable and Expressive RL with Flow-Based Policy" — [arXiv 2605.13435](https://arxiv.org/pdf/2605.13435)
  - "Optimal Transport Q-Learning for Flow Policy Steering and Acceleration" — [arXiv 2607.06262](https://arxiv.org/pdf/2607.06262)
  - "Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints" — [arXiv 2603.17685](https://arxiv.org/pdf/2603.17685)
- Flow distributional critics (2026): PA-BFM, "Posterior-Averaged Bellman Flow Matching" (claims it beats Value Flows on 50 OGBench tasks and 4 offline-to-online scenarios) — [arXiv 2610.17788](https://arxiv.org/html/2610.17788); "Quantile-Coupled Flow Matching for Distributional RL" (40 OGBench tasks) — [arXiv 2605.08515](https://arxiv.org/pdf/2605.08515)
- Counter-trend paper: "ReBRAC-v2: The Return of the King" (2026) suggests a strong Gaussian/deterministic baseline is being re-argued as competitive with flow methods (title-level evidence only). — [arXiv 2608.01205](https://arxiv.org/pdf/2608.01205)

### Inferences
- FQL and OGBench work as a shared "protocol": nearly every 2025–26 flow paper reports the same 50 state-based OGBench tasks (10 environments × 5 tasks), often with FQL's hyperparameters and the same JAX codebase lineage. That makes progress comparable, but it also risks benchmark overfitting.
- The Park/Levine group (FQL → QC → Value Flows → RQL; also Horizon Reduction/SHARSA in NeurIPS 2025, not re-verified here) and the KAIST/Sung group (FAC, FINO, and possibly SQAM per secondary source) are the densest clusters.
- ReBRAC-v2 suggests the community is asking whether expressive policies are really necessary. This is a natural angle for a critical survey section.

### Gaps
- Venue status (ICML 2026, NeurIPS 2026 submissions) for QAM, RQL, DriftQL, MFPO, TRQAM, SQAM, MF2B, and Q-Flow could not be confirmed. NeurIPS 2026 decisions would postdate most searches.
- Bootstrapped FQL's specific method could not be read.
- Horizon Reduction / SHARSA (Park et al., NeurIPS 2025, which uses flow BC with rejection sampling) was not re-verified in this session.

## Q3. Offline-to-online fine-tuning of diffusion/flow policies: behavior, failure modes, fixes

### Takeaway
Known failure modes:
- The offline→online "dip": an initial performance drop from distribution shift and critic miscalibration on new actions.
- Weak exploration: BC-anchored flow policies stay near the data manifold.
- BPTT instability when updating the full multi-step policy online.
- Indirect fine-tuning: FQL-style methods update a distilled one-step actor rather than the flow itself.

Fixes fall into five groups:
1. Keep the expressive base frozen and learn a small residual or latent steering policy (EXPO, DSRL, PA-RL, LPS).
2. Inject noise or entropy to widen exploration (FINO, ReinFlow, DMPO).
3. Use temporally extended actions and critics (Q-chunking, Adaptive QC).
4. Use conservative or trust-region updates (OGPO, TRQAM).
5. Choose the fine-tuning strategy based on whether the pretrained policy beats the dataset (the "Three Regimes" principle).

### Cited Findings
- **The dip, and EXPO's fix.** EXPO, "Stable RL with Expressive Policies" (2025; venue unverified) keeps a base expressive policy trained by a stable supervised (BC) objective and adds a lightweight Gaussian "edit" policy that moves sampled actions toward higher Q, with on-the-fly action selection. The authors report that "compared to traditional offline-to-online RL methods, EXPO does not experience a large drop in performance from offline pretraining to online fine-tuning", attributed to limiting distribution shift. — [arXiv 2507.07986](https://arxiv.org/html/2507.07986)
- **Latent-noise steering.** DSRL works in the latent noise space of a frozen diffusion policy. On Robomimic Can/Square offline-to-online, adding offline data to the buffer sped sample complexity by about 2×. — [arXiv 2506.15799](https://arxiv.org/pdf/2506.15799)
- **Policy-agnostic fine-tuning.** PA-RL (Mark et al.; 2024) does offline RL and online fine-tuning for any policy class or backbone (diffusion, transformer) via action optimization (sample, then Q-rank/gradient-refine) followed by distilling the improved actions back into the policy. — [arXiv 2412.06685](https://arxiv.org/pdf/2412.06685)
- **Limited manifold expansion, and OGPO's fix.** OGPO, "Sample Efficient Full-Finetuning of Generative Control Policies" (2026) targets the dip with conservative advantages. It reports that DSRL and EXPO fail to converge in image-based settings with no offline data in the replay buffer. Its exploration analysis finds that "QC and DSRL show limited manifold expansion, remaining closer to the BC initialization, while EXPO's residual policy facilitates support expansion but not optimal policy extraction." — [arXiv 2605.03065](https://arxiv.org/html/2605.03065)
- **Exploration beyond the dataset.** FINO (ICLR 2026) injects noise into flow-matching policy training so the policy covers actions beyond the offline dataset, and uses entropy-guided sampling online to balance exploration and exploitation. It claims superior performance under limited online budgets. — [OpenReview](https://openreview.net/forum?id=6wd38R8L0Z); [GitHub](https://github.com/CTID282/FINO); [arXiv 2602.18117 via alphaXiv](https://www.alphaxiv.org/de/abs/2602.18117)
- **Fine-tuning the distilled actor, not the flow.** SAC Flow (2025) notes that FQL and QC-FQL are SOTA for off-policy fine-tuning but "rely on an auxiliary, distilled policy for online updates; they do not directly fine-tune the original flow model". SAC Flow instead treats the velocity network as a sequential model (GRU/Transformer-style reparameterization) to stabilize gradients, and evaluates both from-scratch and offline-to-online. — [arXiv 2509.25756](https://arxiv.org/html/2509.25756v1)
- **Online stochasticity for flow policies.** ReinFlow (2025) injects learnable noise into the flow's deterministic path, turning it into a discrete-time Markov process with tractable likelihoods for policy gradients. It is online fine-tuning of pretrained flow policies on robotic control. — [arXiv 2505.22094](https://arxiv.org/pdf/2505.22094)
- **Action chunking for O2O.** Q-chunking (NeurIPS 2025) uses chunked actions for temporally coherent exploration plus unbiased n-step backups. Adaptive Q-Chunking (2026) notes that all prior methods use a fixed chunk size across states and proposes per-state chunk sizes. — [arXiv 2507.07969](https://arxiv.org/pdf/2507.07969); [arXiv 2605.05544](https://arxiv.org/pdf/2605.05544)
- **Methods that report both offline and O2O results.** MeanFlowQL — [arXiv 2511.13035](https://arxiv.org/abs/2511.13035); TRQAM — [arXiv 2605.27079](https://arxiv.org/abs/2605.27079); PA-BFM (4 O2O scenarios) — [arXiv 2610.17788](https://arxiv.org/html/2610.17788).
- **Other 2025–26 O2O papers with diffusion/flow:**
  - "Offline-to-Online RL with Classifier-Free Diffusion Generation" — [arXiv 2508.06806](https://arxiv.org/html/2508.06806)
  - "Efficient and Uncertainty-Aware Diffusion Framework for O2O RL" — [arXiv 2605.30776](https://arxiv.org/pdf/2605.30776)
  - "Critic-Free Pretraining for Efficient Online RL Fine-Tuning" — [arXiv 2608.10473](https://arxiv.org/pdf/2608.10473)
  - "Discrete Flow Matching for O2O RL" (discrete actions; improves online reward while staying close to the pretrained generator) — [arXiv 2605.12379](https://arxiv.org/html/2605.12379v1)
- **General O2O principle (not diffusion-specific, but directly relevant).** "The Three Regimes of Offline-to-Online RL" (Li, Ni, Sun, Bacon; 2025; an OpenReview PDF header suggests a later venue submission) proposes a stability–plasticity principle: preserve whichever prior is stronger, the pretrained policy π0 or the dataset D. The regimes are Superior (J(π0) > J(πD)), Comparable, and Inferior. π0-centric methods (warm-up, offline-RL regularization) suit Superior; D-centric methods (data replay, parameter reset) suit Inferior. Predictions held in 45 of 63 D4RL settings, with only 3 opposite mismatches. — [arXiv 2510.01460](https://arxiv.org/html/2510.01460v3); [OpenReview PDF](https://openreview.net/pdf/0fd2bc16dd89a3386b5a9a288d4baf4b228a2661.pdf)
- **Demonstration-pretrained diffusion policies.** DPPO, Diffusion Policy Policy Optimization (Ren et al.; ICLR 2025) fine-tunes BC-pretrained diffusion policies with PPO over the denoising MDP. It is the standard pretrained-diffusion → online-RL baseline in robotics; its setting is demos, not offline RL with rewards. [bg] — [arXiv 2409.00588](https://arxiv.org/abs/2409.00588)
- **Community reading list.** awesome-offline-to-online-RL-papers — [GitHub](https://github.com/linhlpv/awesome-offline-to-online-RL-papers)

### Inferences
- Two fine-tuning targets compete: (a) fine-tune a small steerer or residual (EXPO, DSRL, PA-RL, LPS, FQL's one-step actor), which is stable but capped by the base model's support; and (b) fine-tune the full flow (SAC Flow, ReinFlow, QAM/TRQAM, OGPO, DMPO), which is more plastic but less stable. OGPO's "manifold expansion" analysis gives a concrete evaluation lens for this axis.
- Q overestimation in O2O for diffusion/flow policies is usually handled implicitly (BC anchors, support constraints, conservative advantages). I found no paper that isolates and measures it for flow policies specifically. This looks like a research opening.
- The Three Regimes framework has not been tested on flow/diffusion policies. A flow-specific study (does expressivity change which regime applies?) is a cheap, publishable analysis.

### Gaps
- I found no source that directly measures "loss of expressivity / mode collapse during online fine-tuning" for flow policies. The phrase "expressivity collapse" is not used in the literature found.
- No direct source tied Q overestimation specifically to the diffusion O2O drop (searched; not found).
- Quantitative O2O numbers (for example QC vs. FQL on OGBench O2O) were only seen in secondary summaries: a German summary claims QC goes from 52% to 86% over 1M offline + 1M online steps, vs. BFN from 51% to 63%. This is unverified.

## Q4. Standard benchmarks, state of the art, and computational cost

### Takeaway
OGBench (state-based: 10 environments × 5 tasks = 50 tasks, plus pixel tasks) has replaced D4RL as the main benchmark for flow-policy offline and O2O RL since FQL (2025). D4RL (antmaze, adroit, locomotion, kitchen) is still reported for continuity. Robomimic is used for robotic O2O and steering work (DSRL, DPPO), and Minari appears in GFP. No single method is uncontested SOTA. Leading 2026 claims on OGBench come from GFP, FAC, ReFORM, QAM/TRQAM/SQAM, RQL, Value Flows / PA-BFM (critic side), and MF2B (D4RL), all compared mainly against FQL. One-step inference is now table stakes. Training cost differences come from distillation stages, JVPs (MeanFlow), consistency losses (SORL), and BPTT through ODE solvers (ReFORM).

### Cited Findings
- **OGBench and FQL's protocol.** OGBench (Park et al.; ICLR 2025) is the benchmark. [bg] — [arXiv 2410.20092](https://arxiv.org/abs/2410.20092). FQL's 73 tasks cover state and pixel OGBench plus D4RL. — [PMLR](https://proceedings.mlr.press/v267/park25f.html). MeanFlowQL describes the split as 50 state OGBench tasks + 5 visual tasks + 18 D4RL antmaze/adroit tasks = 73. — [arXiv 2511.13035](https://arxiv.org/pdf/2511.13035)
- **Common 2026 protocols.**
  - 50 OGBench tasks: QAM, TRQAM, RQL, PA-BFM — [TRQAM](https://arxiv.org/abs/2605.27079); [RQL](https://arxiv.org/html/2606.17551); [PA-BFM](https://arxiv.org/html/2610.17788)
  - 40 OGBench tasks: ReFORM; Quantile-Coupled FM — [ReFORM](https://mit-realm.github.io/reform/); [QCFM](https://arxiv.org/pdf/2605.08515)
  - 144 tasks across OGBench, Minari, and D4RL: GFP — [GFP](https://arxiv.org/abs/2512.03973)
- **D4RL.** Still used by BDPO (D4RL plus 2D toy tasks), FAC (D4RL plus OGBench), MF2B (D4RL SOTA claim), and the Three Regimes study (MuJoCo, AntMaze, Adroit, Kitchen). — [BDPO](https://icml.cc/virtual/2025/poster/44003); [FAC](https://arxiv.org/html/2602.18015v1); [MF2B](https://arxiv.org/html/2610.16984); [Three Regimes](https://arxiv.org/html/2510.01460v3)
- **Robomimic.** Used for DSRL's O2O experiments (Can, Square). — [DSRL](https://arxiv.org/pdf/2506.15799)
- **Hardest OGBench domains.** SORL's gains concentrate on AntSoccer and AntMaze-Large — [SORL](https://nico-espinosadice.github.io/projects/sorl/). SQAM claims +18–35 pp on the four hardest domains — [SQAM](https://arxiv.org/html/2610.10437). Value Flows claims 1.6× success on harder state-based tasks — [Value Flows](https://arxiv.org/pdf/2510.07650).

**Compute cost**
- FQL is "only slightly slower than Gaussian policy-based offline RL methods, while being faster than most other flow-based methods in terms of both training and inference speeds" (one A5000, 8 seeds). — [FQL OpenReview PDF](https://openreview.net/pdf/2f8a68d734574ab1a2831d9dd874defe1d7f2d0e.pdf)
- A third-party per-step timing table (different hardware) reports training ≈ 2.65 ms for FQL, 1.96 ms for IFQL, and 1.51 ms for ReBRAC; inference ≈ 0.56 / 0.85 / 0.59 ms respectively. — ["Aligning Few-Step Generative Models by Amortizing Sample-based VI", arXiv 2605.26552](https://arxiv.org/html/2605.26552)
- FAC measured wall-clock on an RTX 3090. Among JAX methods, IDQL had the lowest training time, and diffusion-based baselines used substantially more GPU memory. — [FAC](https://arxiv.org/html/2602.18015v1)
- Method-specific costs:
  - SORL: longer training than one-step flow baselines because of consistency losses — [SORL](https://nico-espinosadice.github.io/projects/sorl/)
  - MeanFlowQL: JVP overhead in training — [arXiv 2511.13035](https://arxiv.org/pdf/2511.13035)
  - ReFORM: BPTT through an ODE solver to train the noise generator, more costly than standard policy gradients — [ReFORM](https://mit-realm.github.io/reform/)
  - MFPO (online): about 50% less training time than diffusion policies, but still needs 2 sampling steps; 16 → 2 steps degrades performance substantially — [MFPO](https://arxiv.org/pdf/2604.14698)
  - FAN (ICML 2026): exists specifically because flow policies plus distributional critics are expensive — [FAN](https://icml.cc/virtual/2026/poster/61532)
- Historical costs:
  - EDP reported cutting diffusion-policy training on D4RL locomotion from days to hours via action approximation. [bg, exact figure not re-verified] — [arXiv 2305.20081](https://arxiv.org/abs/2305.20081)
  - SRPO reported much faster action sampling than diffusion-sampling methods because its actor is deterministic. [bg, exact speedup not re-verified] — [arXiv 2310.07297](https://arxiv.org/pdf/2310.07297)

### Inferences
- Because most papers compare to FQL on the same 50 OGBench tasks with FQL's hyperparameters, a fair meta-comparison (one codebase, same seeds and tuning budget) across GFP, FAC, ReFORM, QAM, RQL, MeanFlowQL, and SORL does not seem to exist. That would be a useful contribution.
- Compute reporting is inconsistent (per-step ms vs. wall-clock vs. GPU memory, on different GPUs). A standardized "performance per FLOP / per inference ms" Pareto plot is missing from the field.

### Gaps
- I could not extract per-task numeric tables, such as FQL's OGBench averages or Diffusion-QL's D4RL averages, because full PDFs were inaccessible. Unverified numbers are deliberately left out.
- No single leaderboard or source declares an overall SOTA on OGBench as of Oct 2026.

## Q5. Open problems, limitations, and critiques stated by papers

### Takeaway
Self-reported limitations cluster into eight themes:
1. Expressivity vs. one-step trainability: distillation loses expressivity, while full-chain training needs BPTT or JVPs.
2. Regularization-strength tuning: α/β per task.
3. Uniform treatment of good and bad dataset actions.
4. Reliance on BC-model support quality.
5. Compute overhead of new objectives.
6. Limited exploration and manifold expansion in O2O.
7. Narrow evaluation: MuJoCo/OGBench, few pixel or real-robot results.
8. Fixed design choices such as chunk size.

I found no OpenReview reviewer text in this session.

### Cited Findings
- **Distillation's cost.** Distillation-based one-step methods "sacrifice model expressivity in the final one-step policy", and distillation "still requires repeated queries to the underlying multi-step... model". — [MeanFlowQL](https://arxiv.org/pdf/2511.13035); [Bootstrapped FQL](https://arxiv.org/html/2606.10613v1)
- **Prior one-step accelerations.** These "rely on auxiliary modules or policy distillation, sacrificing either simplicity or performance", and curved flow-matching trajectories limit one-step accuracy. — [OFQL](https://arxiv.org/html/2508.13904v3)
- **BPTT and gradient instability through long chains.** — [EXPO](https://arxiv.org/pdf/2507.07986); [FQL](https://proceedings.mlr.press/v267/park25f.html)
- **Uniform behavior regularization ignores action quality.** GFP's motivation, with gains largest on suboptimal data. — [GFP](https://arxiv.org/abs/2512.03973)
- **Hyperparameter-heavy regularization.** ReFORM's motivation. Its own stated limitations:
  - BPTT cost through the ODE solver
  - inherits errors if the BC model misses the data support
  - can learn more slowly than explicit regularization when high-quality expert data is available
  - Source: [ReFORM project](https://mit-realm.github.io/reform/)
- **Critic-driven instability under SOC / adjoint fine-tuning.** TRQAM's motivation. — [TRQAM](https://arxiv.org/abs/2605.27079)
- **Geometric drift repulsion is value-blind.** — [Value-Guided Repulsion](https://arxiv.org/html/2610.20190)
- **One step is not yet enough for some methods.** MFPO's future work aims to reduce from 2 steps to 1. — [MFPO](https://arxiv.org/pdf/2604.14698)
- **Training difficulty of MeanFlow variants.** MVP is harder to train than a standard flow policy. — [MVP](https://arxiv.org/html/2602.13810v2)
- **Narrow evaluation domains.** SOM was evaluated only on MuJoCo; visual and partially observable manipulation is left to future work. — [SOM](https://arxiv.org/html/2605.23365v1)
- **Fixed chunk size.** Adaptive QC's critique of Q-chunking. — [AQC](https://arxiv.org/pdf/2605.05544)
- **O2O exploration.** QC and DSRL remain close to the BC initialization, and EXPO expands support but does not extract the optimal policy. DSRL and EXPO fail to converge on image tasks without offline data. — [OGPO](https://arxiv.org/html/2605.03065)
- **Compute of expressive critics plus policies.** — [FAN, ICML 2026](https://icml.cc/virtual/2026/poster/61532); [SORL](https://nico-espinosadice.github.io/projects/sorl/)
- **Earlier evidence that extraction matters.** "Is Value Learning Really the Main Bottleneck in Offline RL?" (Park, Frans, Levine, Kumar; NeurIPS 2024) found that policy extraction (behavior-regularized policy gradient such as DDPG+BC vs. AWR) and test-time generalization often matter more than value learning. This is the conceptual basis for the FQL/OGBench line. [bg] — [arXiv 2406.09329](https://arxiv.org/abs/2406.09329)
- **Strong Gaussian baselines still contest the need for expressive policies** (title-level evidence). — [ReBRAC-v2](https://arxiv.org/pdf/2608.01205)

### Inferences (candidate research directions grounded in the gaps above)
- Measure, rather than assert, how much expressivity one-step and distilled actors lose. For example, quantify multimodality or coverage of the extracted policy on OGBench tasks with known multimodal optimal actions, and correlate it with return. Most papers argue the point with 2D toys.
- Adaptive or learned regularization strength for flow policies: per-state α from flow density or uncertainty. This combines FAC's density signal with GFP's value weighting and ReFORM's constraint-free framing.
- An O2O study of flow policies under the Three Regimes lens, including whether noise injection (FINO) or full-flow fine-tuning (TRQAM/OGPO) helps differently in the Superior and Inferior regimes.
- Explicit Q-overestimation diagnostics for flow policies during O2O, plus a fix such as a flow-density-based pessimism schedule that anneals online.
- Unified compute-aware evaluation, giving a Pareto frontier of success vs. training and inference FLOPs across FQL, SORL, MeanFlowQL, QAM, RQL, and GFP.
- Test-time scaling plus verifier robustness: SORL and QPILOTS rely on Q as a verifier, so verifier exploitation (best-of-N reward hacking on an imperfect Q) is under-studied.
- Pixel and real-robot validation of 2026 OGBench-centric methods, which most papers lack (SOM states this explicitly).

### Gaps
- No OpenReview review or rebuttal text was retrievable: openreview.net was blocked for direct fetch, and searches returned no review threads. Reviewer critiques of FQL, SORL, GFP, FAC, ReFORM, and FINO are therefore not captured.
- No "Future Work" sections were read in full. Limitations above come from abstracts, project pages, and search excerpts.
