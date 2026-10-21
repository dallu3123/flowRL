# RL / Reward Fine-Tuning of Flow-Matching and Diffusion Generative Models (2023–2026), with Transfer to Flow/Diffusion Policies

> Provenance note for the report writer: direct WebFetch of arxiv.org, alphaxiv.org and openreview.net failed in this session (DNS resolution error), so paper details below come from search-engine extracts of the primary pages (arXiv abstracts/PDFs, CVF open access, ICLR/ICML/NeurIPS virtual pages, GitHub READMEs). Venue tags are given only where a conference page, proceedings page, or the official repo states them; otherwise "arXiv". OpenReview review text could not be retrieved (see Gaps in section 5).

## 1. Method families and representative papers

### Takeaway
The field has moved through five families: (a) policy gradient on the denoising MDP (DDPO/DPOK → Flow-GRPO/DanceGRPO and their many GRPO variants), (b) direct reward backprop (DRaFT/AlignProp), (c) stochastic-optimal-control / adjoint regression (Adjoint Matching → Reinforce Adjoint Matching), (d) preference / reward-weighted regression on the forward process (D3PO, AWM, DiffusionNFT), and (e) training-free inference-time steering (SMC / FK-steering). The 2025–26 trend is away from likelihood-ratio PPO/GRPO over SDE rollouts and toward "pretraining-shaped" regression losses on clean samples (DiffusionNFT, AWM, RAM), which report 8–50x efficiency gains over Flow-GRPO.

### Cited Findings

**(a) Policy gradient on the denoising MDP**
- DDPO (Black et al., ICLR 2024) frames denoising as a multi-step decision problem and optimizes it with policy gradients, outperforming reward-weighted-likelihood baselines; targets include compressibility and aesthetic quality / human-feedback rewards — [ICLR 2024 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/14f75513f0f1ca01de1e826b52e6b840-Paper-Conference.pdf); [arXiv 2305.13301](https://arxiv.org/abs/2305.13301)
- DPOK (Fan et al., NeurIPS 2023) is concurrent policy-gradient work with KL regularization for text-to-image alignment to a single preference reward — [Semantic Scholar](https://www.semanticscholar.org/paper/DPOK:-Reinforcement-Learning-for-Fine-tuning-Models-Fan-Watkins/a553bf27d801d09f667fe121c0ba9632257f364b) (venue NeurIPS 2023 from prior knowledge; not stated in retrieved snippet)
- Flow-GRPO (Liu et al., NeurIPS 2025) — first online RL for flow-matching models: converts the deterministic ODE sampler into an equivalent SDE with the same marginals to obtain stochasticity for exploration, plus a "denoising reduction" strategy (train with fewer denoising steps, infer with full steps); authors report very little reward hacking — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116065); [arXiv 2505.05470](https://arxiv.org/pdf/2505.05470); [official repo, "[NeurIPS 2025]"](https://github.com/yifan123/flow_grpo)
- DanceGRPO (Xue et al., arXiv 2505.07818) — unified GRPO for diffusion and rectified flow across T2I, T2V, I2V; uses shared noise initialization within a group to curb reward hacking; argues DDPO/DPOK fail to stay stable on large diverse prompt sets and that a DDPO-style objective on rectified-flow SDEs "always diverges" — [arXiv 2505.07818](https://arxiv.org/abs/2505.07818); [Liner review](https://liner.com/review/dancegrpo-unleashing-grpo-on-visual-generation). Venue not confirmed.
- MixGRPO (arXiv 2507.21802) — SDE sampling + GRPO optimization only inside a sliding window [t_w, t_w+W]; ODE sampling (allowing high-order solvers) outside the window → efficiency — [arXiv 2507.21802](https://arxiv.org/abs/2507.21802); [v3 HTML](https://arxiv.org/html/2507.21802v3). Venue not confirmed. An aggregator claims a "MixGRPO-Flash" ~71% training-time reduction, not confirmed from the primary paper — [Emergent Mind (aggregator)](https://www.emergentmind.com/topics/flow-grpo)
- TempFlow-GRPO (ICLR 2026 per official repo; arXiv 2508.04324) — temporal credit assignment: trajectory branching creates divergence points at specific timesteps so per-step "process rewards" are estimated from terminal-reward differences of branched trajectories; noise-aware weighting of steps — [arXiv 2508.04324](https://arxiv.org/pdf/2508.04324); [GitHub "[ICLR 26]"](https://github.com/Shredded-Pork/TempFlow-GRPO)
- BranchGRPO (arXiv 2509.06040; OpenReview forum exists) — structured tree branching of rollouts for stable, efficient GRPO in diffusion — [arXiv 2509.06040](https://arxiv.org/pdf/2509.06040); [OpenReview forum](https://openreview.net/forum?id=T2nP2IQasd)
- Stepwise-Flow-GRPO (Savani et al., CVPR 2026) — per-step credit from each step's reward improvement, intermediate rewards estimated with Tweedie's formula, gain-based advantages — [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/papers/Savani_Stepwise_Credit_Assignment_for_GRPO_on_Flow-Matching_Models_CVPR_2026_paper.pdf)
- Other GRPO-family follow-ups found: Neighbor GRPO (contrastive ODE policy optimization, arXiv 2511.16955) — [arXiv](https://arxiv.org/pdf/2511.16955); OP-GRPO (off-policy GRPO for flow matching, arXiv 2604.04142) — [arXiv](https://arxiv.org/pdf/2604.04142); Pref-GRPO (pairwise preference reward GRPO, arXiv 2508.20751; cited in) — [LeanGRPO bibliography](https://arxiv.org/pdf/2609.03528); LeanGRPO (eliminating redundant recomputation, arXiv 2609.03528) — [arXiv](https://arxiv.org/pdf/2609.03528); NormGuard (norm constraints in flow-matching RL, arXiv 2606.27771) — [arXiv](https://arxiv.org/pdf/2606.27771)
- A 2026 survey, "Advances in GRPO for Generation Models" (arXiv 2603.06623), covers Flow-GRPO, TempFlow-GRPO, MixGRPO and reward-hacking mitigation — [arXiv HTML](https://arxiv.org/html/2603.06623v1)

**(b) Direct reward backpropagation**
- DRaFT (Clark et al., ICLR 2024) backpropagates a differentiable reward through the full sampling chain — [survey arXiv 2505.17352](https://arxiv.org/html/2505.17352v2)
- AlignProp (Prabhudesai et al., arXiv 2023) — end-to-end reward-gradient backprop through the diffusion model; peer-reviewed venue not confirmed in this session — [survey arXiv 2505.17352](https://arxiv.org/pdf/2505.17352)

**(c) Stochastic optimal control / adjoint methods**
- Uehara et al., "Fine-Tuning of Continuous-Time Diffusion Models as Entropy-Regularized Control" (arXiv 2402.15194, 2024; authors incl. Black, Levine) — ELEGANT; fine-tunes drift and initial distribution of a neural SDE to maximize reward with entropy (KL) regularization, applied to images and bio-sequences — [arXiv PDF](https://arxiv.org/pdf/2402.15194); [Semantic Scholar](https://www.semanticscholar.org/paper/Fine-Tuning-of-Continuous-Time-Diffusion-Models-as-Uehara-Zhao/cbeca8850c37cb3cca794091c888b00a193c2b59)
- Tang & Zhou, "Fine-tuning of diffusion models via stochastic control: entropy regularization and beyond" (arXiv 2403.06279; v3 Sept 2025) — rigorous analysis of entropy-regularized fine-tuning, extended to general f-divergence regularizers; SD v1.5 experiments — [arXiv 2403.06279](https://arxiv.org/abs/2403.06279)
- Adjoint Matching (Domingo-Enrich, Drozdzal, Karrer, Chen; ICLR 2025) — casts reward fine-tuning as SOC and turns it into a least-squares regression ("adjoint matching"), combining scalability of gradient-based methods with regression simplicity; reports better consistency, realism and generalization to unseen preference reward models while retaining diversity; higher λ → better consistency/preference but less diversity — [arXiv 2409.08861](https://arxiv.org/abs/2409.08861); [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/852f50969a9e523ec41d26f2f68bd456-Abstract-Conference.html)
- Reinforce Adjoint Matching (RAM; Bergmeister, Jegelka, Nüsken, Domingo-Enrich, Pidstrigach; arXiv 2605.10759, NeurIPS 2026 poster) — under KL-regularized reward maximization, the optimal process tilts the clean-endpoint distribution but leaves the noising process unchanged; combines adjoint-matching optimality with a REINFORCE identity into a consistency loss that "corrects the pretraining target with the reward"; each step: sample clean endpoint from current model → reward → noise as in pretraining → regress. No SDE rollouts, adjoint sweeps, or reward gradients. On SD3.5-M reaches Flow-GRPO's peak reward in 50x / 48x / 34x fewer steps (GenEval / text rendering / human preference) — [arXiv 2605.10759](https://arxiv.org/pdf/2605.10759); [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/154808); [MSR video](https://www.microsoft.com/en-us/research/video/reinforce-adjoint-matching-scaling-diffusion-rl/)
- Further adjoint-matching variants (titles verified via citations only): "Improved techniques for fine-tuning flow models via adjoint matching: a deterministic control pipeline" (arXiv 2605.06583) — [arXiv](https://arxiv.org/pdf/2605.06583); "Efficient adjoint matching for fine-tuning diffusion models" (arXiv 2605.11480; seen only as a citation in) — [search extract via RAM refs](https://arxiv.org/html/2605.10759)

**(d) Preference optimization and reward-weighted / forward-process regression**
- D3PO (Yang et al., CVPR 2024) — fine-tunes from human comparisons without a reward model, treating each denoising step as part of an expanded MDP — [survey arXiv 2505.17352](https://arxiv.org/html/2505.17352v2)
- Self-Sampling Preference Optimization (arXiv 2410.05255) — bridges SFT and DPO for diffusion alignment — [arXiv](https://arxiv.org/pdf/2410.05255)
- Advantage Weighted Matching (AWM; arXiv 2509.25050; ICML 2026 poster) — argues DDPO is an implicit score/flow-matching objective with noisy targets (higher variance, slower convergence); instead trains with the pretraining (flow-matching) loss on clean samples weighted by advantage; reported speedup over Flow-GRPO is 24x in one arXiv version and 34x in a later version; one version reports matching Flow-GRPO's GenEval 0.95 at 8.02x fewer GPU hours on SD3.5-M — [arXiv 2509.25050](https://arxiv.org/abs/2509.25050); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/61737)
- DiffusionNFT (Zheng, Chen, Ye, Wang, Zhang, Jiang, Su, Ermon, Zhu, Liu; ICLR 2026 Oral) — online RL on the *forward* process via flow matching: contrasts positive vs negative generations to define an implicit policy-improvement direction, optimizing a single target policy through an implicit parameterization; works with arbitrary black-box solvers, needs no likelihood estimation, only clean images, CFG-free; up to 25x more efficient than Flow-GRPO; GenEval 0.24 → 0.98. Motivation: reverse-process RL suffers forward inconsistency, solver restriction, and awkward CFG integration (project page: forward-process design prevents "degradation into cascaded Gaussians") — [ICLR 2026 Oral](https://iclr.cc/virtual/2026/oral/10009150); [arXiv 2509.16117](https://arxiv.org/pdf/2509.16117); [NVIDIA project page](https://research.nvidia.com/labs/cosmos-lab/diffusionnft/)
- MeanFlowNFT (arXiv 2607.15273) — extends forward-process RL to average-velocity (MeanFlow, few-step) generators — [arXiv](https://arxiv.org/pdf/2607.15273)
- Stitched Value Model for Diffusion Alignment (arXiv 2605.19804) — value-model approach to alignment — [arXiv](https://arxiv.org/pdf/2605.19804)
- GDRO (CVPR 2026) — group-level reward post-training suited to diffusion — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_GDRO_Group-level_Reward_Post-training_Suitable_for_Diffusion_Models_CVPR_2026_paper.pdf)
- Discriminator-guided RL correcting flow matching ("The Reward Was in Your Data All Along", arXiv 2606.19162) — [arXiv](https://arxiv.org/pdf/2606.19162)

**(e) Inference-time guidance / search**
- FK-Steering (Singhal et al., 2025; arXiv 2501.06848) — Feynman-Kac interacting particle system (rare-event simulation) for reward steering; increasing particle budget improves alignment without updating the model — [arXiv HTML](https://arxiv.org/html/2501.06848v2)
- TRI-TSMC (arXiv 2605.25123) — notes SMC steering propagates particles under base-model dynamics so many start in low-reward regions when the target is far; learns twisting functions iteratively with trust-region updates — [arXiv](https://arxiv.org/html/2605.25123)
- GLASS Flows (arXiv 2509.25170) — replaces SDE transitions between FK-steering resampling steps with a transition sampler for flow models; improves SOTA T2I with FLUX — [arXiv](https://arxiv.org/html/2509.25170v1)
- Nested SMC for discrete diffusion (arXiv 2608.20123) — argues best-of-n suffers overoptimism and bootstrap SMC suffers weight degeneracy — [arXiv](https://arxiv.org/html/2608.20123)
- Uehara et al., "Inference-Time Alignment in Diffusion Models with Reward-Guided Generation: Tutorial and Review" (arXiv 2501.09685) — [arXiv](https://arxiv.org/pdf/2501.09685)

**Surveys**
- Uehara, Zhao, Biancalani, Levine, "Understanding RL-Based Fine-Tuning of Diffusion Models: A Tutorial and Review" (arXiv 2407.13734) — unified view as entropy-regularized MDP; covers PPO, differentiable optimization, reward-weighted MLE, value-weighted sampling, path consistency learning; links to classifier guidance, GFlowNets, path-integral control — [arXiv HTML](https://arxiv.org/html/2407.13734v1)
- "Alignment and Safety of Diffusion Models via RL and Reward Modeling: A Survey" (arXiv 2505.17352) — [arXiv](https://arxiv.org/html/2505.17352v2)
- "Advances in GRPO for Generation Models: A Survey" (arXiv 2603.06623) — [arXiv](https://arxiv.org/html/2603.06623v1)

### Inferences
- Trade-off axes across families: (i) need for likelihoods/ratios (PG/GRPO need per-step Gaussian log-probs → force SDE sampling; AWM/NFT/RAM do not), (ii) need for reward gradients (DRaFT/AlignProp/Adjoint Matching need differentiable reward; GRPO/NFT/AWM/RAM are black-box), (iii) solver lock-in (reverse-process methods tie training to a sampler; forward-process methods are solver-agnostic), (iv) compute (full-chain backprop and full SDE rollouts are expensive; forward-process regression is cheapest).
- The convergence of AWM, DiffusionNFT and RAM on "noise a reward-weighted clean sample and regress as in pretraining" suggests a unifying view: RL fine-tuning = flow matching onto a reward-tilted endpoint distribution, differing in how weights/targets are estimated (advantage weights, positive/negative contrast, REINFORCE-corrected adjoint target).
- Efficiency claims (25x, 34x, 50x over Flow-GRPO) are all author-reported against the same baseline on SD3.5-M; no independent head-to-head among NFT/AWM/RAM was found.

### Gaps
- Could not verify from primary pages in this session (known from prior knowledge, IDs should be checked before citing): Diffusion-DPO (Wallace et al., CVPR 2024, arXiv 2311.12908); ImageReward/ReFL (Xu et al., NeurIPS 2023, arXiv 2304.05977); RAFT reward-ranked fine-tuning (arXiv 2304.06767); TDS twisted diffusion sampler (Wu et al., NeurIPS 2023, arXiv 2306.17775); DAS "Test-time alignment of diffusion models without reward over-optimization" (Kim et al., ICLR 2025, arXiv 2501.05803); SVDD soft value-based decoding (Li et al., arXiv 2408.08252); "Inference-time scaling for diffusion models beyond scaling denoising steps" (Ma et al., CVPR 2025, arXiv 2501.09732). FK-steering is listed as ICML 2025 in prior knowledge but the retrieved page did not state the venue.
- Coefficients-Preserving Sampling (CPS) for Flow-GRPO, G2RPO, TreeGRPO, Dense-GRPO, DiffusionNFT's video extension: no search results retrieved; not verified.
- Venues for DanceGRPO, MixGRPO, BranchGRPO not confirmed.

## 2. What 2025–2026 papers add: efficiency, credit assignment, reward hacking/diversity, KL

### Takeaway
2025–26 contributions cluster into (1) efficiency: partial-SDE windows (MixGRPO), fewer training steps (Flow-GRPO denoising reduction), forward-process / rollout-free regression (NFT, AWM, RAM) and recomputation removal (LeanGRPO); (2) per-timestep credit assignment via branching or Tweedie-based intermediate rewards (TempFlow-GRPO, BranchGRPO, Stepwise-Flow-GRPO); (3) a growing recognition that KL alone does not prevent diversity collapse, motivating distribution-matching objectives, step-dependent regularizers, and ratio-clipping fixes.

### Cited Findings
- Flow-GRPO ablation: removing KL keeps image quality roughly intact on human-preference reward but collapses visual diversity (a form of reward hacking); matching the KL-free reward with quality preserved needs a proper KL term and longer training — [arXiv 2505.05470](https://arxiv.org/pdf/2505.05470)
- GRPO-Guard (CVPR 2026; arXiv 2510.22319): implicit over-optimization arises because the importance-ratio distribution in flow-matching GRPO is left-shifted, which disables PPO clipping for overconfident positive samples; fix = RatioNorm (regulated clipping) — [arXiv](https://arxiv.org/pdf/2510.22319); [paper note](https://en.papernotes.org/CVPR2026/image_generation/grpo-guard_mitigating_implicit_over-optimization_in_flow_matching_via_regulated_/)
- DiverseGRPO (CVPR 2026): GRPO induces mode collapse; groups same-caption samples via spectral clustering and gives exploration rewards inversely proportional to cluster size; replaces uniform late-stage KL with a Wasserstein-type constraint strongest at early denoising steps; up to 18% semantic-diversity gain at matched quality — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_DiverseGRPO_Mitigating_Mode_Collapse_in_Image_Generation_via_Diversity-Aware_GRPO_CVPR_2026_paper.pdf)
- TMPO (arXiv 2605.10983): GRPO-based reward maximization collapses to a single high-reward mode even with KL; replaces reward maximization with trajectory-level reward-distribution matching — [arXiv](https://arxiv.org/html/2605.10983)
- Distribution-based regularization (DFR, arXiv 2610.24897): on GenEval/OCR with SD3.5-M, up to 12% less drift and 29% more diversity than KL-regularized baselines at matched reward, 36–41x faster training than Flow-GRPO — [arXiv](https://arxiv.org/html/2610.24897)
- ReNFT (arXiv 2609.00061): repairs mode collapse in reward post-training via internal probability-mass recalibration — [arXiv](https://arxiv.org/pdf/2609.00061)
- "Resolving the Identity Crisis in T2I" (arXiv 2510.01399): intra-image diversity rewards alone cause global (cross-sample) diversity collapse; also a hacking mode where the model generates fewer people to satisfy count rewards — [arXiv](https://arxiv.org/html/2510.01399)
- DanceGRPO uses shared initial noise within a group to reduce reward hacking — [Liner review](https://liner.com/review/dancegrpo-unleashing-grpo-on-visual-generation)
- Credit assignment: TempFlow-GRPO (branching → per-step process reward) — [arXiv 2508.04324](https://arxiv.org/pdf/2508.04324); Stepwise-Flow-GRPO (Tweedie intermediate reward, gain-based advantages) — [CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Savani_Stepwise_Credit_Assignment_for_GRPO_on_Flow-Matching_Models_CVPR_2026_paper.pdf); BranchGRPO (tree branching) — [arXiv 2509.06040](https://arxiv.org/pdf/2509.06040)
- Efficiency: MixGRPO window — [arXiv 2507.21802](https://arxiv.org/abs/2507.21802); LeanGRPO removes redundant recomputation — [arXiv 2609.03528](https://arxiv.org/pdf/2609.03528); OP-GRPO off-policy reuse — [arXiv 2604.04142](https://arxiv.org/pdf/2604.04142); RAM 34–50x fewer steps — [arXiv 2605.10759](https://arxiv.org/pdf/2605.10759); DiffusionNFT 25x — [project page](https://research.nvidia.com/labs/cosmos-lab/diffusionnft/)

### Inferences
- Credit assignment over denoising time is now treated as a first-class problem, but every method found uses terminal reward + branching or Tweedie denoised estimates; none learns a critic over (x_t, t) jointly with environment time, which is exactly what a policy-RL setting would need.
- Reward hacking in this literature has three distinct mechanisms worth separating in a survey: (i) exploiting the reward model (count hacking, aesthetic overfitting), (ii) diversity/mode collapse (a distributional failure that KL to the base model does not fully fix), and (iii) optimizer-induced over-optimization (GRPO-Guard's ratio-shift finding).

### Gaps
- No quantitative cross-paper comparison of diversity metrics under a common protocol was found.
- Video-specific results (DanceGRPO T2V/I2V numbers) were not retrieved in detail.

## 3. Theory: target distribution, value-function bias, entropy-regularized RL

### Takeaway
The accepted theoretical target is the reward-tilted distribution p*(x) ∝ p_base(x)·exp(r(x)/λ), obtained as the optimum of KL-regularized (entropy-regularized) control. Adjoint Matching shows that with a generic noise schedule, naive SOC fine-tuning is biased by the initial value function; only the "memoryless" schedule removes the noise–sample dependence and provably yields the tilted distribution. RAM further shows the optimal fine-tuned process tilts only the endpoint while leaving the noising process unchanged — the theoretical justification for forward-process regression methods.

### Cited Findings
- Entropy-regularized control formulation of fine-tuning (drift + initial-distribution adjustment) aims to retain diversity while maximizing reward — [Uehara et al. arXiv 2402.15194](https://arxiv.org/pdf/2402.15194)
- Rigorous treatment and extension from KL to general f-divergence regularization — [Tang & Zhou arXiv 2403.06279](https://arxiv.org/abs/2403.06279)
- Tutorial: diffusion fine-tuning as RL in entropy-regularized MDPs; links to classifier guidance, GFlowNets, path-integral control — [arXiv 2407.13734](https://arxiv.org/html/2407.13734v1)
- Adjoint Matching: a unique ("memoryless") noise schedule removes dependency between noise variables and generated samples, giving provable convergence to the tilted distribution; constant σ(t) fine-tuning leads to biased distributions (value-function bias) — [arXiv 2409.08861](https://arxiv.org/abs/2409.08861)
- Critique (2026): the memoryless schedule injects very large noise near early denoising times and may over-explore highly noisy states, weakening reward propagation; training with a non-memoryless schedule locks the model to that schedule at inference — [arXiv 2605.06583](https://arxiv.org/pdf/2605.06583) (search-extract)
- RAM: under KL-regularized reward maximization the optimal process shifts the clean-endpoint distribution toward high reward and leaves the noising process unchanged — [arXiv 2605.10759](https://arxiv.org/pdf/2605.10759)
- Flow Density Control (arXiv 2511.22640) — "generative optimization beyond entropy-regularized fine-tuning", i.e., objectives beyond KL-tilting — [arXiv HTML](https://arxiv.org/html/2511.22640v1)
- QAM (ICLR 2026) frames the target policy as the behavior policy reweighted by exp(Q) — the RL analogue of the reward-tilted distribution — and claims the resulting policy is unbiased at optimum — [arXiv 2601.14234](https://arxiv.org/pdf/2601.14234.pdf); [mlanthology](https://mlanthology.org/iclr/2026/li2026iclr-qlearning/)

### Inferences
- The GRPO/Flow-GRPO line optimizes a PPO surrogate over an SDE whose noise level is a free hyperparameter; by Adjoint Matching's result, such objectives with a KL term do not in general target the exact tilted distribution unless the schedule is memoryless — a theory-practice gap that, as far as found, no GRPO paper analyzes explicitly.
- In policy RL with a bootstrapped Q, the "reward" itself is a moving, biased estimate; combining value-function bias (from schedule) with critic bias is unanalyzed.

### Gaps
- Could not retrieve formal theorem statements (exact memoryless schedule form σ(t)=√(2η_t) etc.) from primary text; verify in the ICLR 2025 PDF.
- No paper found giving convergence/sample-complexity guarantees for GRPO-style flow fine-tuning.

## 4. Transfer to decision-making / robot policies: done vs. not yet done

### Takeaway
Many image-side ideas have already been ported to flow/diffusion policies within ~6–12 months: denoising-MDP PPO (DPPO), learnable-noise ODE→SDE (ReinFlow, πRL Flow-Noise/Flow-SDE), GRPO for VLAs (πRL), likelihood-free ratios (FPO), adjoint matching with critics (QAM, MaxEnt-AM), forward-process RL (RoboFFT), latent-noise steering (DSRL), one-step distillation (FQL) and test-time Q-steering (QPILOTS). Less-transferred: timestep credit-assignment tricks, diversity-collapse/ratio-clipping fixes, RAM-style rollout-free regression, SMC/FK steering with learned value potentials, and memoryless-schedule correctness analysis under bootstrapped critics.

### Cited Findings — already transferred
- DPPO (Ren et al., ICLR 2025) — PPO over denoising transitions as a multi-step MDP for robot diffusion policies — [citation via search](https://arxiv.org/pdf/2507.15073)
- ReinFlow (NeurIPS 2025; arXiv 2505.22094) — injects learnable noise into a flow policy's deterministic path, making it a discrete-time Markov process with tractable likelihood; BC pretrain → online RL; later scaled to VLAs (π0, π0.5, GR00T N1.5) — [GitHub](https://github.com/ReinFlow/ReinFlow); [arXiv](https://arxiv.org/pdf/2505.22094)
- πRL (arXiv 2510.25889) — online RL for flow-based VLAs (π0, π0.5) with Flow-Noise and Flow-SDE formulations to obtain tractable action log-likelihoods; PPO and GRPO in RLinf; LIBERO, ManiSkill3, MetaWorld, CALVIN; one summary reports π0 on LIBERO 57.6% → 97.6% — [arXiv](https://arxiv.org/pdf/2510.25889); [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/resources/publications/pi_rl.html)
- FPO, "Flow Matching Policy Gradients" (McAllister, Ge, Yi, Kim, Weber, Choi, Feng, Kanazawa; arXiv 2507.21053) — likelihood-free ratio proxy from changes in the conditional flow-matching loss — [search extract](https://github.com/justLeo-1/GRPO-Flow) (method description from secondary source; verify)
- GRPO-Flow (GitHub) — GRPO for flow-matching robot policies vs. ReinFlow's PPO on robomimic can — [GitHub](https://github.com/justLeo-1/GRPO-Flow)
- Reinforcement Fine-Tuning of Flow-Matching Policies for VLAs (arXiv 2510.09976) — [Pith](https://pith.science/paper/2510.09976)
- "Reinforcement Learning for Flow-Matching Policies" (arXiv 2507.15073) — [arXiv](https://arxiv.org/pdf/2507.15073); with density transport (arXiv 2606.08602) — [arXiv](https://arxiv.org/pdf/2606.08602)
- Flow Q-Learning (Park, Li, Levine; ICML 2025) — offline RL trains a one-step policy against Q instead of backpropagating through the iterative flow, regularized toward a BC flow — [ICML poster](https://icml.cc/virtual/2025/poster/45637); [PMLR v267](https://proceedings.mlr.press/v267/park25f.html)
- QAM, Q-learning with Adjoint Matching (Li & Levine; ICLR 2026) — uses adjoint matching to turn the critic's action gradient into a step-wise objective, avoiding unstable backprop through denoising; target = behavior policy × exp(Q); 50 OGBench tasks, offline and offline-to-online, strong on sparse reward — [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10006800); [arXiv](https://arxiv.org/abs/2601.14234v2)
- Entropy-Regularized Adjoint Matching for Offline RL (arXiv 2605.06156) — [alphaxiv](https://www.alphaxiv.org/abs/2605.06156); Scalable MaxEnt RL for Diffusion Policies via Adjoint Matching (arXiv 2606.22630) — [arXiv](https://arxiv.org/pdf/2606.22630) (content not retrieved)
- RoboFFT (arXiv 2609.32236) — fine-tunes generative robot policies via online RL with the forward process (DiffusionNFT-style transfer) — [arXiv HTML](https://arxiv.org/html/2609.32236v1)
- DSRL, "Steering your diffusion policy with latent space RL" (Wagenmaker et al., CoRL 2025) — RL in the initial-noise space of a frozen diffusion policy — citation retrieved via [search results](https://arxiv.org/pdf/2607.06262) (method from prior knowledge; verify)
- Reverse Flow Matching: unified online RL with diffusion and flow policies (arXiv 2601.08136) — [arXiv](https://arxiv.org/pdf/2601.08136)
- Q-guidance / steering for flow policies: Aligning Flow Map Policies with Optimal Q-Guidance (arXiv 2605.12416) — [arXiv](https://arxiv.org/pdf/2605.12416); QPILOTS test-time Q-steering (arXiv 2606.14801) — [arXiv](https://arxiv.org/pdf/2606.14801); Optimal Transport Q-Learning for flow policy steering and acceleration (arXiv 2607.06262) — [arXiv](https://arxiv.org/pdf/2607.06262); Q-Flow (arXiv 2605.13435) — [arXiv](https://arxiv.org/html/2605.13435); Reversal Q-Learning (arXiv 2606.17551) — [arXiv](https://arxiv.org/html/2606.17551); Reparameterization Flow Policy Optimization (arXiv 2602.03501) — [arXiv](https://arxiv.org/pdf/2602.03501); VINE: Taming Generative Control Policies for RL (arXiv 2607.10369) — [arXiv](https://arxiv.org/pdf/2607.10369)
- Energy-weighted flow matching for offline RL (ICLR 2025) — reward/energy-weighted regression analogue — cited in [search extract](https://arxiv.org/pdf/2606.22630)

### Inferences — candidate research gaps (not found transferred in this session's searches)
- **Two-axis credit assignment**: TempFlow/Stepwise/BranchGRPO-style per-denoising-step credit has not been combined with environment-time TD credit (Q over (s, a_t^τ, τ)); a "denoising-step advantage" for action chunks is an open slot.
- **RAM / AWM for policies**: rollout-free, regression-shaped RL (noise a reward/Q-weighted action and regress) is the image analogue of AWR-on-flows; RoboFFT covers NFT, but RAM's REINFORCE-corrected adjoint target with a learned Q (and its bias when Q is bootstrapped) appears untested.
- **Memoryless-schedule correctness with critics**: QAM uses adjoint matching, but the interplay of value-function bias (schedule) with critic bias / off-policy data is unanalyzed; also the 2026 critique that memoryless schedules over-explore noisy states may matter more for low-dimensional action spaces.
- **Diversity / multimodality preservation**: DiverseGRPO, TMPO (distribution matching), DFR, and GRPO-Guard's ratio normalization have no policy-RL counterpart found; robot RL fine-tuning commonly collapses multimodal BC policies, so diversity-preserving fine-tuning is a natural port.
- **SMC / FK steering of action chunks with Q potentials**: Q-steering papers exist (QPILOTS, Q-guidance), but particle-based twisted SMC with trust-region twists (TRI-TSMC) for action generation was not found.
- **Few-step generators**: MeanFlowNFT (forward-process RL for MeanFlow) suggests RL for one/few-step flow-map policies; FQL distills to one step but does not do on-policy forward-process RL on flow maps.
- **CFG-free / solver-agnostic training** (DiffusionNFT's selling point) matters for VLAs that deploy with different integrators/step counts than training — unexplored as an explicit evaluation axis.

### Gaps
- FPO, DSRL and DPPO method details were not confirmed from primary pages this session; the MaxEnt-AM (2606.22630) and VINE (2607.10369) contents were not retrieved — they may already cover some "gaps" above. Searches for 2026 robot papers were not exhaustive.

## 5. Open problems stated in papers and reviews

### Takeaway
Explicitly stated open problems: likelihood intractability and solver lock-in of reverse-process RL; per-step credit assignment; diversity collapse not fixed by KL; implicit over-optimization through PPO clipping; schedule choice (memoryless vs. practical); SMC particle inefficiency when the target is far from the base model; and best-of-n overoptimism. OpenReview review text could not be accessed.

### Cited Findings
- Reverse-process RL problems: forward inconsistency, solver restriction, complicated CFG integration — [DiffusionNFT](https://arxiv.org/pdf/2509.16117)
- Existing methods rely on costly SDE rollouts, reward gradients, or surrogate losses, sacrificing pretraining's regression structure — [RAM](https://arxiv.org/pdf/2605.10759)
- DDPO-style objectives have high variance (implicit noisy-target flow matching) — [AWM](https://arxiv.org/abs/2509.25050)
- Prior PG methods unstable at scale on diverse prompt sets — [DanceGRPO](https://arxiv.org/abs/2505.07818)
- KL insufficient against mode collapse — [TMPO](https://arxiv.org/html/2605.10983); [DiverseGRPO](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_DiverseGRPO_Mitigating_Mode_Collapse_in_Image_Generation_via_Diversity-Aware_GRPO_CVPR_2026_paper.pdf)
- Ratio-distribution shift disables clipping → implicit hacking — [GRPO-Guard](https://arxiv.org/pdf/2510.22319)
- Memoryless schedule over-explores noisy states / schedule lock-in — [arXiv 2605.06583](https://arxiv.org/pdf/2605.06583)
- SMC particles start in low-reward regions when target is far from base — [TRI-TSMC](https://arxiv.org/html/2605.25123); best-of-n overoptimism and SMC weight degeneracy — [Nested SMC](https://arxiv.org/html/2608.20123)
- Action log-likelihood intractability after iterative denoising blocks large-scale RL for flow VLAs — [πRL](https://arxiv.org/pdf/2510.25889)

### Inferences
- The most defensible ICML/ICLR-grade directions sit at intersections: (1) unbiased tilted-distribution fine-tuning with bootstrapped critics (AM/RAM theory + TD), (2) diversity-preserving policy fine-tuning (distribution-matching objectives for multimodal action policies), (3) denoising-time × environment-time credit assignment, (4) solver/step-count–agnostic policy RL (NFT/RAM style) evaluated across deployment integrators.

### Gaps
- OpenReview pages (BranchGRPO forum, Flow-GRPO NeurIPS 2025 forum, DiffusionNFT ICLR 2026 forum) could not be fetched (DNS error), so reviewer-stated weaknesses are not included. Recommend manual check: [BranchGRPO forum](https://openreview.net/forum?id=T2nP2IQasd), and Flow-GRPO forum linked from the [NeurIPS poster page](https://neurips.cc/virtual/2025/poster/116065).
