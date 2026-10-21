# Diffusion and Flow-Matching Policies for Online Reinforcement Learning (2023–2026)

Scope note: online RL only (on-policy PG, off-policy actor-critic, MaxEnt). Offline RL and VLA/robot fine-tuning are only mentioned where an online method is the main point. Research cutoff: 2026-10-21.

Verification legend:
- **[V]** means the title, venue, and core claim were confirmed this session through search results (arXiv, OpenReview, PMLR, NeurIPS, ICLR, or ICML proceedings).
- **[P]** means the information comes from prior knowledge and was not re-fetched this session (direct arXiv and OpenReview fetches failed from this environment with DNS errors). The paper exists, but treat specific numbers as needing a check.

Most claims below are at abstract level, because full-text fetches of arXiv and OpenReview were blocked in this environment.

---

## Q1. Main approaches and representative papers: what each does and how it handles the core problem

### Takeaway
The field has split into roughly six families, defined by how they get around the missing tractable log-likelihood π(a|s):
1. **Denoising-MDP / stochastic-chain likelihoods** (DPPO, ReinFlow, DA-MDP). These make every denoising step a Gaussian so the chain likelihood is exact.
2. **Likelihood-free surrogates for PPO** (FPO, ReFPO, PolicyFlow, GenPO with exact inversion).
3. **Q-gradient or score-based actor updates** (QSM, DIPO). These push the score or actions along ∇ₐQ.
4. **Reparameterized backprop through the sampler with entropy proxies** (DACER, SAC Flow, FlowRL).
5. **Weighted or reweighted (flow/score) matching as policy mirror descent** (QVPO, RSM/DPMD/SDAC, FPMD, FMER, GeMPO, RLDT, SMFP).
6. **MaxEnt / Boltzmann-sampling formulations borrowed from diffusion samplers** (DIME, MaxEntDP, AMDP via adjoint matching, FLAG, DA-MDP).

A separate branch, latent-noise steering (DSRL), avoids changing the generative model at all.

### Cited Findings

**Family A – Denoising-MDP / stochastic chain (exact per-step Gaussian likelihoods)**
- **DPPO** (Ren, Lidard, Ankile et al., ICLR 2025) [V]
  - Fine-tunes diffusion policies with policy-gradient RL (PPO) by treating the denoising chain as an MDP, with each denoising step recorded as part of the rollout.
  - Reports the strongest overall performance and efficiency among RL methods for diffusion-policy fine-tuning on common benchmarks. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html); [arXiv 2409.00588](https://arxiv.org/pdf/2409.00588)
  - [P] The setup is a "two-layer" MDP: the environment MDP plus an inner denoising MDP whose Gaussian steps give exact log-probs. It is evaluated mainly as fine-tuning of pretrained policies (Gym locomotion, Robomimic, Furniture-Bench), not from-scratch learning, and also uses DDIM-style fine-tuning of only the last few denoising steps. — [arXiv 2409.00588](https://arxiv.org/pdf/2409.00588)
- **ReinFlow** (Zhang, Yu, Su, Wang, NeurIPS 2025 main track) [V]
  - Injects *learnable* noise into a flow policy's deterministic ODE path, which turns it into a discrete-time Markov process with exact likelihoods, so PPO-style updates apply.
  - Works with very few or even one denoising step. — [NeurIPS 2025 proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/98d6f928497d9eaad9f320fb2db3040d-Abstract-Conference.html); [arXiv 2505.22094](https://arxiv.org/pdf/2505.22094v6)
  - [P] The abstract reports large reward gains on Gym locomotion and Franka Kitchen and large wall-time savings vs DPPO. The exact percentages are not re-verified here. — [arXiv 2505.22094](https://arxiv.org/pdf/2505.22094v6)
- **Diffusion-Augmented MDPs (DA-MDP)** (Sanokowski & Patil, arXiv Dec 2025) [V]
  - Extends MaxEnt RL to diffusion policies by treating each reverse-diffusion transition as an RL decision; only the final denoised action is executed.
  - Instantiated with PPO, REPPO, and a MaxEnt extension of WPO. Reports higher success than a Gaussian MaxEnt-RL baseline on StackCube and PushT. — [arXiv 2512.02019](https://arxiv.org/abs/2512.02019)

**Family B – Likelihood-free / exact-likelihood surrogates for on-policy PPO**
- **FPO, Flow Matching Policy Gradients** (McAllister, Ge, Yi, Kim, Weber, Choi, Feng, Kanazawa; ICLR 2026) [V]
  - On-policy algorithm that replaces the PPO likelihood ratio with a ratio built from (exponentiated differences of) conditional flow-matching losses, i.e., an ELBO proxy. Compatible with PPO-clip.
  - Agnostic to the sampler or integration scheme used at training and inference time.
  - Claims flow policies trained from scratch can represent multimodal action distributions and beat Gaussian policies, especially in under-conditioned settings. — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d43cc5692bf68944ee7cd31b97d0c11-Abstract-Conference.html); [arXiv 2507.21053](https://arxiv.org/abs/2507.21053); [OpenReview PDF](https://openreview.net/pdf/376ea41422d8f63f58a4916b9a9783de90e8e461.pdf)
- **ReFPO** (arXiv 2606.21086, listed as a NeurIPS 2026 poster) [V]
  - Interprets FPO gradient updates as an implicit advantage-weighted *Reflow* process and adds an explicit reflow regularizer, described as a one-line code change.
  - Claims fewer policy-ratio spikes and strong one-step inference. Tested on GridWorld, MuJoCo Playground, and Humanoid Control. — [arXiv 2606.21086](https://arxiv.org/pdf/2606.21086); [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/152804)
- **PolicyFlow** (Yang, Liu, Chen; arXiv 2602.01156; OpenReview forum YETCQLcKtn) [V]
  - Continuous-normalizing-flow policy with a PPO-clip objective.
  - Approximates importance ratios through velocity-field changes along interpolation paths instead of backprop through the flow.
  - Adds a "Brownian-inspired" entropy regularizer. — [arXiv 2602.01156](https://arxiv.org/abs/2602.01156); [OpenReview](https://openreview.net/forum?id=YETCQLcKtn)
- **GenPO** (Ding et al., NeurIPS 2025) [V]
  - Uses *exact diffusion inversion* to build invertible action mappings, which gives computable log-likelihoods. It uses an alternating "doubled dummy action" design.
  - The log-likelihoods are used for entropy and KL estimates and for KL-adaptive learning rates in PPO.
  - Evaluated on 8 IsaacLab tasks (Ant, Humanoid, Anymal-D, Unitree H1, Go2, Shadow Hand, Quadcopter, Franka) and reports beating RL baselines. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119348); [arXiv 2505.18763](https://arxiv.org/pdf/2505.18763)
- **GSB-MDPO, Path-Space Mirror Descent under the Generalized Schrödinger Bridge** (Gong, Wang, Chen, Ding, Zhou, Fu; arXiv 2603.21621; v1 was titled "Proximal Policy Optimization in Path Space: A Schrödinger Bridge Perspective") [V]
  - Casts on-policy generative-policy optimization as a GSB problem over denoising paths and uses a path-measure KL as the proximal term. Key fact used: the path-measure KL upper-bounds the KL between terminal action marginals.
  - Evaluated on 14 tasks across MuJoCo Playground and Gym-MuJoCo. — [arXiv 2603.21621](https://arxiv.org/abs/2603.21621)
- **Constrained Flow Policy Updates: A Generalized Schrödinger Bridge View** (arXiv 2609.32952) [V, existence only; content not retrieved] — [arXiv 2609.32952](https://arxiv.org/abs/2609.32952)

**Family C – Q-gradient / score-based off-policy actors**
- **QSM, Q-Score Matching** (Psenka, Escontrela, Abbeel, Ma; ICML 2024, PMLR 235) [V]
  - Matches the policy's score to the action gradient ∇ₐQ, so the update differentiates only through the denoiser, not the whole sampling chain.
  - Converged policies are reported to be implicitly multimodal and explorative. — [PMLR](https://proceedings.mlr.press/v235/psenka24a.html); [arXiv 2312.11752](https://arxiv.org/abs/2312.11752v4)
- **DIPO** (Yang et al., "Policy Representation via Diffusion Probability Model for RL", arXiv 2023) [P]
  - Performs "action gradient" updates on replay-buffer actions (a ← a + η∇ₐQ) and then fits the diffusion model to the updated actions by behavior cloning. — [arXiv 2305.13122](https://arxiv.org/abs/2305.13122); existence corroborated by [Pith citation index](https://pith.science/citations/feb1bdbe-9bcc-4d51-8132-994b46f4d10d)

**Family D – Reparameterized backprop through the sampler (SAC-style)**
- **DACER** (Wang et al., incl. S. E. Li; NeurIPS 2024) [V]
  - Treats the reverse diffusion as the policy and backprops Q through the chain.
  - Entropy has no closed form, so it is estimated by fitting a Gaussian mixture model. The estimate drives a learned α that scales noise added to the output action.
  - Claims SOTA on most MuJoCo tasks plus a multimodal task. — [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract.html); [arXiv 2405.15177](https://arxiv.org/html/2405.15177v1)
- **SAC Flow** (Zhang, Yu et al., Tsinghua; ICLR 2026) [V]
  - Observes that multi-step flow sampling is algebraically an RNN-style recurrence, so backprop through it suffers vanishing or exploding gradients.
  - Reparameterizes the velocity network as gated (Flow-G, GRU-like) or Transformer (Flow-T) variants and trains end-to-end with SAC plus a noise-augmented rollout.
  - Claims SOTA on continuous control and robotic manipulation. — [arXiv 2509.25756](https://arxiv.org/pdf/2509.25756); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66a3f3d3317da8d4b9c1579e2bc97d5-Abstract-Conference.html)
  - [P] Covers both from-scratch (MuJoCo) and offline-to-online (OGBench/Robomimic) settings. — [arXiv HTML v2](https://arxiv.org/html/2509.25756v2)
- **FlowRL, Flow-Based Policy for Online RL** (Lv et al., NeurIPS 2025) [V]
  - Flow policy trained to maximize Q under a Wasserstein-2 constraint toward a "behavior-optimal" policy implicitly derived from the replay buffer.
  - Reports competitive results on DMControl and HumanoidBench. — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119311); [arXiv 2506.12811](https://arxiv.org/abs/2506.12811)
  - Name clash: this is not the same as the "FlowRL" taxonomy/codebase paper (2603.27450) below.

**Family E – Weighted / reweighted matching as (approximate) policy mirror descent**
- **QVPO** (Ding, Hu et al.; NeurIPS 2024) [V]
  - Q-weighted variational loss, proven to be a tight lower bound of the policy objective under some conditions, with Q-weight transformation functions for generality.
  - Uses a separately designed entropy regularizer (exact entropy is inaccessible) and a behavior policy to reduce variance during interaction.
  - Reports SOTA on MuJoCo. — [arXiv 2405.16173](https://arxiv.org/abs/2405.16173v1); [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/94963)
- **RSM → DPMD & SDAC** ("Efficient Online RL for Diffusion Policy", Ma, Chen, Wang, Li, Dai; ICML 2025) [V]
  - Reweighted Score Matching generalizes denoising score matching with a reweighted loss. It keeps DSM's optimum and cost but needs no samples from the target policy.
  - Yields Diffusion Policy Mirror Descent (DPMD) and Soft Diffusion Actor-Critic (SDAC).
  - The ICML version reports DPMD improving more than 120% over SAC on Humanoid and Ant. arXiv v3 states more than 100%, so cite the proceedings figure. — [PMLR v267](https://proceedings.mlr.press/v267/ma25d.html); [arXiv 2502.00361](https://arxiv.org/html/2502.00361v3)
- **FPMD, One-Step Flow Policy Mirror Descent** (Chen, Ma, Li, Wang, Dai; arXiv 2507.23675, v2 Oct 2025) [V]
  - Uses a link between distribution variance and the one-step discretization error of straight-interpolation flow matching to sample in one step without distillation or consistency training. Comes in flow and MeanFlow variants.
  - Comparable to diffusion baselines on MuJoCo with "hundreds of times fewer" function evaluations at inference. — [arXiv 2507.23675](https://arxiv.org/abs/2507.23675v1)
- **FMER, Flow Matching Policy Optimization with Mirror Descent and Entropy Constraints** (arXiv 2603.17685; v1 titled "Flow Matching Policy with Entropy Regularization") [V]
  - Advantage-weighted conditional flow matching serves as a simulation-free surrogate for PMD.
  - Uses an analytic entropy objective that corrects tanh-squash density distortion, and tunes the MD temperature via effective sample size to keep a trust region.
  - Strong on sparse-reward FrankaKitchen, competitive on MuJoCo. — [arXiv 2603.17685](https://arxiv.org/html/2603.17685v3)
- **GeMPO, Generalized Measure Matching** (arXiv 2603.10250) [V]
  - Builds a virtual target policy measure from a regularized policy-optimization objective, then minimizes divergence to it with reweighted flow matching. It generalizes reweighted FM through flexible weighting functions. — [arXiv 2603.10250](https://arxiv.org/html/2603.10250v2)
- **Stochastic MeanFlow Policies (SMFP)** (arXiv 2605.21282) [V]
  - One-step MeanFlow noise-to-action map combined with Gaussian reparameterization, which gives a tractable entropy surrogate usable in off-policy entropic mirror descent.
  - Strong vs Gaussian and generative baselines on 7 MuJoCo tasks with single-step inference. — [arXiv 2605.21282](https://arxiv.org/html/2605.21282v2)
- **MFPO, Mean Flow Policy Optimization** (arXiv 2604.14698) [V]
  - Evaluated on 5 MuJoCo, 6 hard DMC, and 3 HumanoidBench tasks. Comparable to or better than diffusion baselines with much lower training and inference time. — [arXiv 2604.14698](https://arxiv.org/abs/2604.14698); [awesomepapers summary](https://awesomepapers.io/reinforcement-learning/papers/2604.14698)
- **RLDT, RL for Flow-Matching Policies with Density Transport** (Lei, Daniilidis, Loquercio, UPenn; arXiv 2606.08602) [V]
  - Treats policy improvement as transporting action mass toward high-reward regions. The transport field is built from a MaxEnt objective via Stein Variational Gradient Descent, and the flow is fine-tuned to match it.
  - Uses expected-target estimation on intermediate steps to avoid BPTT.
  - Argues that prior methods either approximate the policy distribution or distill, which biases gradients or loses multimodality. Gains on dense and sparse tasks and on state- and vision-based long-horizon manipulation. — [arXiv 2606.08602](https://arxiv.org/abs/2606.08602); [project page](https://rpfey.github.io/rldt/)

**Family F – MaxEnt / Boltzmann-target formulations (diffusion-sampler view)**
- **DIME** (Celik et al., KIT; ICML 2025) [V]
  - Uses approximate inference with diffusion models (a diffusion-sampler-style variational bound) to obtain a *lower bound* on the MaxEnt objective, with a policy-iteration scheme proven to converge to the optimal diffusion policy.
  - The practical version builds on CrossQ and distributional RL and is evaluated on 13 high-dimensional control benchmarks.
  - Competitive with SOTA non-diffusion methods with fewer design choices and smaller UTD ratios. — [mlanthology ICML 2025](https://mlanthology.org/icml/2025/celik2025icml-dime); [arXiv 2502.02316](https://www.arxiv.org/pdf/2502.02316); [ICML poster](https://icml.cc/virtual/2025/poster/46144)
- **MaxEntDP** (Dong, Cheng, Zhang; ICML 2025, PMLR 267: 13963–13983) [V]
  - Diffusion policy inside the MaxEnt RL framework.
  - On MuJoCo, outperforms Gaussian and other generative policies within MaxEnt RL and is comparable to SOTA diffusion online RL. Code: github.com/diffusionyes/MaxEntDP. — [mlanthology](https://mlanthology.org/icml/2025/dong2025icml-maximum)
  - [P] Fits the Boltzmann target exp(Q/α) via a Q-weighted noise-estimation objective, related to iDEM-style energy matching.
- **AMDP, Scalable MaxEnt RL for Diffusion Policies via Adjoint Matching** (Thilges, Celik, Blessing, Gospodinov, Neumann; arXiv 2606.22630, June 2026) [V]
  - Uses Adjoint Matching (the SOC reward fine-tuning objective of Domingo-Enrich et al.) for simulation-free training that minimizes the reverse KL to the Boltzmann policy, with no explicit likelihoods and no BPTT. — [arXiv 2606.22630](https://arxiv.org/abs/2606.22630)
- **FLAG, Flow Policy MaxEnt-RL by Latent Augmented Guidance** (Sungha Kim et al.; arXiv 2605.30749) [V]
  - Augments the state with the flow latent variable and optimizes a provably consistent proxy MaxEnt objective.
  - Explicitly criticizes importance-weighted supervised generative-policy methods for "importance weight collapse" in high-dimensional action spaces. Claims SOTA. — [arXiv 2605.30749](https://arxiv.org/abs/2605.30749)
- **iDEM** (Akhound-Sadegh et al., ICML 2024) [P]
  - Not an RL paper. It is a simulation-free diffusion sampler for Boltzmann densities using only the energy and its gradient. It is the conceptual template that MaxEnt diffusion-policy methods borrow: sampling from π ∝ exp(Q/α) is a Boltzmann sampling problem. — [arXiv 2402.06121](https://arxiv.org/abs/2402.06121)

**Latent-noise steering**
- **DSRL** (Wagenmaker et al.; CoRL 2025, PMLR 305: 258–282) [V]
  - Runs RL in the latent-noise space of a frozen BC diffusion policy and never changes the base weights.
  - The RLinf docs describe a small SAC agent (~500K parameters) that outputs the initial noise for a frozen π0. — [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html); [arXiv 2506.15799](https://arxiv.org/abs/2506.15799); [RLinf docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)

**Other related online methods** [P, not re-verified this session]
- **Consistency-AC** (Ding & Jin, "Consistency Models as a Rich and Efficient Policy Class for RL", ICLR 2024) includes online RL. — [arXiv 2309.16984](https://arxiv.org/abs/2309.16984)
- **DDiffPG** (Li et al., "Learning Multimodal Behaviors from Scratch with Diffusion Policy Gradient", NeurIPS 2024) handles explicit mode discovery from scratch. — [arXiv 2406.00681](https://arxiv.org/abs/2406.00681)

**Survey / unified framework**
- **FlowRL: A Taxonomy and Modular Framework for RL with Diffusion Policies** (Gao, Chen, Chen, Dai; RLC 2026; v2 June 2026) [V]
  - Unified taxonomy plus a JAX/JIT modular codebase that decouples architectures, actor/critic variants, and generative models.
  - Standardized benchmarks on Gym-Locomotion, DMC, and IsaacLab. — [arXiv 2603.27450](https://arxiv.org/html/2603.27450v2)

### Inferences
- The Dai group (Georgia Tech) is the most prolific single line here (RSM/DPMD/SDAC → FPMD → FlowRL taxonomy). Its consistent framing is *policy mirror descent realized by reweighted (score/flow) matching*. GeMPO, FMER, and SMFP follow the same template. This framing is becoming the de facto off-policy standard in 2026.
- On-policy work has converged on two options. One is to make each denoising step stochastic (DPPO, ReinFlow, DA-MDP). The other is to use an ELBO/CFM-loss ratio (FPO, ReFPO). Path-measure KL (GSB-MDPO) is emerging as a principled bridge between the two, since the path KL upper-bounds the marginal KL.
- Clear trend toward **one-step / few-step** policies (FPMD, SMFP, MFPO, ReFPO one-step, ReinFlow 1-step). Inference cost is now treated as a first-class metric.

### Gaps
- "Q-Flow: Stable and Expressive RL with Flow-Based Policy" (arXiv 2605.13435) showed up in a search listing, but no abstract could be retrieved. Its method is unverified.
- No "DACER2 / enhanced DACER" paper could be confirmed in this session.
- Direct arXiv and OpenReview fetches failed (DNS) in this environment, so method details are abstract-level only.

---

## Q2. Core obstacles

### Takeaway
Five recurring obstacles appear across the papers:
1. **No tractable log π(a|s)**, which breaks PPO ratios, SAC entropy, and KL trust regions.
2. **BPTT through the sampling chain** is costly and unstable, and it behaves like an RNN.
3. **Deterministic ODE samplers give no controllable stochasticity** for exploration or entropy.
4. **A step-count trade-off**: more steps means more expressiveness and stable likelihoods, but higher compute and harder gradients.
5. **Importance-weight collapse and bias** in weighted-matching surrogates, especially in high-dimensional action spaces.

Each method family is essentially a different trade between bias, variance, and compute on these axes.

### Cited Findings
- **Intractable likelihood/entropy**
  - The lack of explicit log-probabilities is the central challenge for efficient RL with diffusion/flow policies (FlowRL taxonomy). — [arXiv 2603.27450](https://arxiv.org/html/2603.27450v2)
  - DACER: no closed-form entropy, so it is estimated with a GMM. — [arXiv 2405.15177](https://arxiv.org/html/2405.15177v1)
  - QVPO: entropy is "nontrivial" because the log-likelihood is inaccessible. — [arXiv 2405.16173](https://arxiv.org/html/2405.16173v1)
  - GenPO notes on-policy diffusion RL is underexplored precisely for this reason and uses exact inversion to fix it. — [arXiv 2505.18763](https://arxiv.org/pdf/2505.18763)
- **ELBO-ratio bias (FPO)**
  - The FPO ratio equals the true likelihood ratio times an inverse KL-gap correction.
  - With few Monte Carlo samples, the single-sample ratio is, by Jensen, only an *upper bound* in expectation on the true ratio. The number of MC draws is a tuning knob. — [arXiv 2507.21053](https://arxiv.org/pdf/2507.21053); [OpenReview PDF](https://openreview.net/pdf/376ea41422d8f63f58a4916b9a9783de90e8e461.pdf)
  - ReFPO reports that FPO suffers "policy-ratio spikes", which its reflow regularizer reduces. — [arXiv 2606.21086](https://arxiv.org/pdf/2606.21086)
- **BPTT instability and cost**
  - SAC Flow: multi-step flow rollout is algebraically equivalent to an RNN, so off-policy training of multi-step flows is unstable from vanishing/exploding gradients. The fix is GRU/Transformer-style velocity reparameterization. — [arXiv 2509.25756](https://arxiv.org/pdf/2509.25756)
  - QSM avoids full-chain differentiation by only differentiating the denoiser. — [arXiv 2312.11752](https://arxiv.org/abs/2312.11752v4)
  - PolicyFlow avoids "costly backpropagation through the flow". — [arXiv 2602.01156](https://arxiv.org/abs/2602.01156)
  - RLDT uses expected-target estimation to avoid BPTT. — [arXiv 2606.08602](https://arxiv.org/abs/2606.08602)
  - AMDP uses adjoint matching to avoid both likelihoods and "costly backpropagation through the diffusion process". — [arXiv 2606.22630](https://arxiv.org/abs/2606.22630)
- **Exploration for deterministic samplers**
  - ReinFlow injects *learnable* noise to make the ODE path a stochastic Markov chain. — [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/98d6f928497d9eaad9f320fb2db3040d-Abstract-Conference.html)
  - SAC Flow uses a "noise-augmented rollout". — [arXiv 2509.25756](https://arxiv.org/pdf/2509.25756)
  - DACER uses α-scaled additive output noise. — [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract.html)
  - SMFP adds Gaussian reparameterization to a one-step MeanFlow to get an entropy surrogate. — [arXiv 2605.21282](https://arxiv.org/html/2605.21282v2)
  - PolicyFlow uses a Brownian-inspired entropy regularizer. — [arXiv 2602.01156](https://arxiv.org/abs/2602.01156)
  - QVPO uses a separate behavior policy to control variance during interaction. — [arXiv 2405.16173](https://arxiv.org/abs/2405.16173v1)
- **Tanh squashing**
  - FMER derives a correction for density distortion from tanh squashing in the entropy objective. This is a detail most diffusion-SAC methods ignore. — [arXiv 2603.17685](https://arxiv.org/html/2603.17685v3)
- **Steps vs. performance / inference cost**
  - FPMD reaches parity with diffusion baselines on MuJoCo at "hundreds of times fewer" NFEs, through a variance/discretization-error link in straight flows. — [arXiv 2507.23675](https://arxiv.org/abs/2507.23675v1)
  - MFPO substantially cuts training and inference time. — [arXiv 2604.14698](https://arxiv.org/abs/2604.14698)
  - ReinFlow works down to one step. — [arXiv 2505.22094](https://arxiv.org/pdf/2505.22094v6)
  - FPO is sampler-agnostic, so the step count can differ between training and inference. — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d43cc5692bf68944ee7cd31b97d0c11-Abstract-Conference.html)
- **Importance-weight collapse**
  - FLAG argues that importance-weighted supervised generative-policy learning (the Q-weighted / reweighted-matching family) suffers weight collapse that limits scaling to high-dimensional actions. — [arXiv 2605.30749](https://arxiv.org/abs/2605.30749)
  - RLDT argues that approximation and distillation approaches bias gradients or lose multimodality. — [arXiv 2606.08602](https://arxiv.org/abs/2606.08602)
- **Sample efficiency vs. Gaussian**
  - DIME claims competitiveness with *smaller* UTD ratios than SOTA non-diffusion methods. — [mlanthology](https://mlanthology.org/icml/2025/celik2025icml-dime)
  - DPMD reports more than 120% over SAC on Humanoid and Ant. — [PMLR](https://proceedings.mlr.press/v267/ma25d.html)
  - FPO reports gains over Gaussian mainly in under-conditioned settings, which implies smaller gains where Gaussian is adequate. — [arXiv 2507.21053](https://arxiv.org/abs/2507.21053)

### Inferences
- Exploration in ODE-based flow policies is almost always handled with an ad-hoc Gaussian (output noise, per-step noise, or a reparameterized head). This makes the "expressive multimodal policy" partly a Gaussian policy in disguise for exploration purposes. A principled entropy definition for the ODE map itself (e.g., via divergence-of-velocity / instantaneous change of variables) is used rarely, with PolicyFlow and FMER as partial exceptions.
- The weighted-matching family (QVPO, RSM, GeMPO, FMER) and the Boltzmann-sampler family (DIME, AMDP) face the same core issue as diffusion samplers for unnormalized densities: mode coverage vs. weight degeneracy. FLAG's critique suggests this will be a central 2026–2027 battleground.

### Gaps
- No source found that gives a controlled, apples-to-apples wall-clock and sample-efficiency comparison of all families. The FlowRL taxonomy benchmark (2603.27450) may contain it, but its numbers were not retrieved.
- No quantitative ablation of number of steps vs. return across families was retrieved.

---

## Q3. Most recent (2025–2026) papers: claims, benchmarks, gains vs SAC/PPO

### Takeaway
Since mid-2025 the frontier has moved in four directions:
1. On-policy flow PG at ICLR 2026 (FPO) and NeurIPS 2026 (ReFPO).
2. Stable off-policy flow SAC at ICLR 2026 (SAC Flow).
3. One-step MeanFlow / mirror-descent policies (FPMD, SMFP, MFPO).
4. MaxEnt methods using SOC/sampler tools (AMDP via adjoint matching, FLAG, RLDT via SVGD, GSB-MDPO via Schrödinger bridges).

Benchmarks have expanded from MuJoCo-v4 to DMC-hard, HumanoidBench, MuJoCo Playground, and IsaacLab. Headline gains vs SAC are largest on Humanoid/Ant-type tasks. Most 2026 papers compare mainly against other diffusion baselines rather than strong modern Gaussian baselines.

### Cited Findings
| Paper | Venue / date | Setting | Benchmarks | Claimed result |
|---|---|---|---|---|
| DPPO | ICLR 2025 | on-policy fine-tune | Gym, Robomimic, Furniture-Bench [P] | strongest among diffusion-RL fine-tuning methods — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html) |
| DIME | ICML 2025 | off-policy MaxEnt | 13 high-dim control tasks (CrossQ base) | competitive with SOTA non-diffusion, smaller UTD — [arXiv](https://www.arxiv.org/pdf/2502.02316) |
| MaxEntDP | ICML 2025 | off-policy MaxEnt | MuJoCo | > Gaussian & other generative MaxEnt; ≈ SOTA diffusion — [mlanthology](https://mlanthology.org/icml/2025/dong2025icml-maximum) |
| DPMD/SDAC | ICML 2025 | off-policy PMD/MaxEnt | MuJoCo | DPMD >120% over SAC on Humanoid & Ant — [PMLR](https://proceedings.mlr.press/v267/ma25d.html) |
| ReinFlow | NeurIPS 2025 | on-policy fine-tune | Gym locomotion, Franka Kitchen, Robomimic [P] | exact likelihood via learnable noise; ≤1 step — [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/98d6f928497d9eaad9f320fb2db3040d-Abstract-Conference.html) |
| FlowRL (Lv) | NeurIPS 2025 | off-policy | DMControl, HumanoidBench | competitive — [NeurIPS](https://neurips.cc/virtual/2025/poster/119311) |
| GenPO | NeurIPS 2025 | on-policy (PPO) | 8 IsaacLab tasks | beats RL baselines — [NeurIPS](https://neurips.cc/virtual/2025/poster/119348) |
| DSRL | CoRL 2025 | latent-noise RL | sim + real robot (frozen policy) | adapts without weight change — [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html) |
| FPO | ICLR 2026 | on-policy from scratch | continuous control; [P] MuJoCo Playground / humanoid | > Gaussian esp. under-conditioned — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3d43cc5692bf68944ee7cd31b97d0c11-Abstract-Conference.html) |
| SAC Flow | ICLR 2026 | off-policy, scratch + off2on | continuous control & manipulation | SOTA claimed — [ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66a3f3d3317da8d4b9c1579e2bc97d5-Abstract-Conference.html) |
| FPMD | arXiv Jul/Oct 2025 | off-policy PMD, 1-step | MuJoCo | ≈ diffusion baselines at ~100× fewer NFEs — [arXiv](https://arxiv.org/abs/2507.23675v1) |
| DA-MDP | arXiv Dec 2025 | on-policy MaxEnt | StackCube, PushT | > Gaussian MaxEnt — [arXiv](https://arxiv.org/abs/2512.02019) |
| PolicyFlow | arXiv Feb 2026 | on-policy CNF+PPO | (not retrieved) | — [arXiv](https://arxiv.org/abs/2602.01156) |
| GeMPO | arXiv Mar 2026 | online, reweighted FM | (not retrieved) | — [arXiv](https://arxiv.org/html/2603.10250v2) |
| FMER | arXiv Mar 2026 | off-policy PMD+entropy | FrankaKitchen (sparse), MuJoCo | strong on sparse — [arXiv](https://arxiv.org/html/2603.17685v3) |
| GSB-MDPO | arXiv Mar 2026 | on-policy path-space MD | 14 tasks: Playground + Gym-MuJoCo | — [arXiv](https://arxiv.org/abs/2603.21621) |
| FlowRL taxonomy | RLC 2026 | survey + JAX codebase | Gym-Loco, DMC, IsaacLab | standardized benchmark — [arXiv](https://arxiv.org/html/2603.27450v2) |
| MFPO | arXiv Apr 2026 | MeanFlow | 5 MuJoCo, 6 hard DMC, 3 HumanoidBench | ≥ diffusion baselines, less time — [arXiv](https://arxiv.org/abs/2604.14698) |
| SMFP | arXiv May 2026 | off-policy entropic MD, 1-step | 7 MuJoCo | strong vs Gaussian & generative — [arXiv](https://arxiv.org/html/2605.21282v2) |
| FLAG | arXiv May 2026 | MaxEnt, latent-augmented | "challenging benchmarks" | SOTA claimed — [arXiv](https://arxiv.org/abs/2605.30749) |
| RLDT | arXiv Jun 2026 | online fine-tune, SVGD transport | dense/sparse; state & vision manipulation | beats baselines — [arXiv](https://arxiv.org/abs/2606.08602) |
| ReFPO | arXiv Jun 2026; NeurIPS 2026 | on-policy, FPO+reflow | GridWorld, MuJoCo Playground, Humanoid Control | fewer ratio spikes, 1-step — [arXiv](https://arxiv.org/pdf/2606.21086) |
| AMDP | arXiv Jun 2026 | MaxEnt via adjoint matching | (not retrieved) | scalable, BPTT-free — [arXiv](https://arxiv.org/abs/2606.22630) |

- **Version discrepancy:** DPMD's gain over SAC is stated as more than 120% in the ICML final version and more than 100% in arXiv v3. — [PMLR](https://proceedings.mlr.press/v267/ma25d.html) vs [arXiv v3](https://arxiv.org/html/2502.00361v3)
- **Out of scope but related:** "Scaling Reinforcement Learning for Diffusion Models via Velocity Matching" (arXiv 2608.23664) appeared in search. Its target domain (image generation vs control) was not verified. — [arXiv 2608.23664](https://arxiv.org/pdf/2608.23664)

### Inferences
- Gaussian baselines in these papers are usually vanilla SAC/PPO. Comparisons against strong modern Gaussian agents (CrossQ, BRO, SimBa, TD-MPC2) are rare. DIME is a partial exception, since it builds on CrossQ. "Gains vs SAC" headline numbers should be read with this caveat.
- Humanoid and Ant are where diffusion/flow gains are most often highlighted. That fits the hypothesis that expressiveness helps most in high-dimensional, multimodal action spaces. FLAG, however, argues that weighted-matching methods fail exactly there.

### Gaps
- Exact numeric results (returns, % gains) for FPO, SAC Flow, MFPO, SMFP, FLAG, AMDP, ReFPO, and GSB-MDPO could not be retrieved, because full-text fetches were blocked.
- No accepted ICML 2026 online diffusion/flow-policy paper could be confirmed in this session. ICML 2026 venues for the 2026 arXiv papers above are unknown.

---

## Q4. Theory: convergence, policy improvement, links to SOC / entropy-regularized RL

### Takeaway
Theory is mostly of three kinds:
1. **Policy-iteration convergence under exact function classes**: DIME proves convergence of its diffusion MaxEnt policy iteration.
2. **Surrogate-tightness results**: QVPO's tight lower bound, RSM's preservation of the DSM optimum, FPO's ELBO-gap decomposition, FLAG's consistent proxy objective.
3. **Path-measure / SOC arguments**: GSB-MDPO's path-KL ≥ marginal-KL bound, AMDP's adjoint-matching reverse-KL.

The deep link is that entropy-regularized policy improvement π_new ∝ π_old·exp(Q/α) is exactly a reward-tilted sampling or SOC problem, the same one solved by adjoint matching, iDEM-style samplers, and Schrödinger bridges.

### Cited Findings
- DIME: lower bound on the MaxEnt objective via diffusion approximate inference, with a policy-iteration scheme *proven to converge to the optimal diffusion policy*. — [mlanthology](https://mlanthology.org/icml/2025/celik2025icml-dime)
- QVPO: the Q-weighted variational loss is a *tight lower bound* of the policy objective "under certain conditions". — [arXiv 2405.16173](https://arxiv.org/abs/2405.16173v1)
- RSM: the reweighted loss preserves the optimal solution of denoising score matching while targeting the PMD/MaxEnt optimum without sampling from it. — [PMLR](https://proceedings.mlr.press/v267/ma25d.html)
- FPMD: theoretical link between distribution variance and one-step discretization error in straight-interpolation flow matching. Low-variance (late-training) policies can be sampled in one step. — [arXiv 2507.23675](https://arxiv.org/abs/2507.23675v1)
- FPO: the ratio decomposes as the true likelihood ratio times an inverse ELBO-KL-gap correction. Maximizing it raises likelihood and shrinks the gap. With a single MC sample the ratio is biased (an upper bound in expectation). — [arXiv 2507.21053](https://arxiv.org/pdf/2507.21053)
- ReFPO: FPO updates are interpreted as implicit advantage-weighted Reflow. — [arXiv 2606.21086](https://arxiv.org/pdf/2606.21086)
- GSB-MDPO: on-policy generative optimization as a Generalized Schrödinger Bridge. The path-measure KL upper-bounds the terminal-action marginal KL, which justifies a tractable path-space proximal term. — [arXiv 2603.21621](https://arxiv.org/abs/2603.21621)
- AMDP: Adjoint Matching (from SOC reward fine-tuning) yields simulation-free reverse-KL minimization to the MaxEnt target. — [arXiv 2606.22630](https://arxiv.org/abs/2606.22630)
- FLAG: a "provably consistent" proxy MaxEnt objective on a latent-augmented state space. — [arXiv 2605.30749](https://arxiv.org/abs/2605.30749)
- DA-MDP: casting the reverse chain as MDP steps gives a MaxEnt variational bound over denoising steps. — [arXiv 2512.02019](https://arxiv.org/abs/2512.02019)
- FMER: tunes the mirror-descent temperature by effective sample size to keep a trust region. — [arXiv 2603.17685](https://arxiv.org/html/2603.17685v3)

### Inferences
- No retrieved paper gives finite-sample or global convergence rates for diffusion/flow actor-critic with function approximation and approximate (few-step) samplers. Existing results assume exact policy classes or exact target fitting. The generic PMD linear-convergence theory (e.g., [arXiv 2301.13139](https://arxiv.org/pdf/2301.13139); [arXiv 2607.11626](https://arxiv.org/pdf/2607.11626)) has not been specialized to errors from reweighted flow matching.
- How sampler discretization error (number of steps) propagates into policy-improvement guarantees is mostly open. FPMD's variance–error link is the only explicit result found.

### Gaps
- No OpenReview review text was retrievable to see whether reviewers challenged these theoretical claims.

---

## Q5. Open problems and stated limitations (papers + reviews)

### Takeaway
The stated limitations cluster around five themes:
1. Bias and variance of likelihood surrogates (FPO's MC-sample bias and ratio spikes).
2. BPTT instability (SAC Flow).
3. Importance-weight collapse in high-dimensional action spaces (FLAG).
4. Loss of multimodality or gradient bias from approximation and distillation (RLDT).
5. Ad-hoc exploration noise and entropy estimates (DACER's GMM, tanh-squash distortion in FMER).

Other open items: inference cost, the lack of a standardized benchmark (motivating the FlowRL taxonomy), and weak comparisons against strong Gaussian baselines.

### Cited Findings
- FPO: fewer MC samples introduce bias, and the single-sample ratio is an upper bound in expectation. — [arXiv 2507.21053](https://arxiv.org/pdf/2507.21053)
- FPO's policy-ratio spikes and multi-step inference are the motivating problems for ReFPO. — [arXiv 2606.21086](https://arxiv.org/pdf/2606.21086)
- SAC Flow: multi-step flow off-policy training is unstable (vanishing/exploding gradients) without architectural changes. — [arXiv 2509.25756](https://arxiv.org/pdf/2509.25756)
- FLAG: importance weight collapse limits scalability in high-dimensional action spaces. — [arXiv 2605.30749](https://arxiv.org/abs/2605.30749)
- RLDT: prior methods either approximate the policy distribution or distill, giving biased gradients or lost multimodality. — [arXiv 2606.08602](https://arxiv.org/abs/2606.08602)
- FMER: tanh squashing distorts density, which entropy objectives must correct. — [arXiv 2603.17685](https://arxiv.org/html/2603.17685v3)
- FlowRL taxonomy: the field lacks a unified perspective and standardized benchmarks. — [arXiv 2603.27450](https://arxiv.org/html/2603.27450v2)
- GenPO: on-policy diffusion RL remains underexplored compared to offline and off-policy work. — [arXiv 2505.18763](https://arxiv.org/pdf/2505.18763)
- DSRL: fine-tuning diffusion weights directly is challenging, so it avoids weight changes entirely. This implicitly limits performance to what the base policy's noise manifold can reach. — [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html)

### Inferences (candidate research directions)
1. **Principled entropy for ODE flow policies.** Use the instantaneous change of variables (−∫∇·v dt) with Hutchinson estimators, or SMFP-style Gaussian reparameterization, and check how both interact with tanh squashing (FMER).
2. **Unbiased or low-variance likelihood-ratio estimators for FPO-style PPO.** Ideas include control variates, antithetic noise, and path-KL proximal terms (GSB-MDPO) to replace the clipped ELBO ratio. Analyze spike behavior theoretically (ReFPO).
3. **Robust weighting for reweighted matching** that addresses FLAG's weight collapse. Options: tempered or ESS-adaptive weights (FMER), SMC/annealed resampling borrowed from diffusion samplers, adjoint-matching targets (AMDP).
4. **Convergence theory with sampler error**, i.e., PMD with approximate flow-matching projections and K-step discretization. Generalize FPMD's variance–error link into rate guarantees.
5. **One-step expressive policies without distillation** (FPMD, MeanFlow, SMFP, ReFPO). Open question: does one-step inference keep multimodality on genuinely multimodal tasks, or does it collapse to near-Gaussian behavior?
6. **Fair benchmarking against strong Gaussian agents** (CrossQ/BRO/SimBa) at matched UTD and wall-clock, using standardized suites (FlowRL taxonomy codebase; MuJoCo Playground/IsaacLab for on-policy).
7. **Hybrid steering plus fine-tuning:** combine DSRL-style latent-noise RL (cheap, stable) with weight-space updates when the latent manifold saturates.

### Gaps
- OpenReview reviews for FPO (forum eoEmoKoQpJ, per search), SAC Flow, PolicyFlow (YETCQLcKtn), DIME, and DPPO could not be retrieved. openreview.net was unreachable from this environment, so no reviewer-stated weaknesses are included. A follow-up should read those forums directly.
- Limitations sections of the 2026 arXiv papers were not accessible beyond abstracts.
