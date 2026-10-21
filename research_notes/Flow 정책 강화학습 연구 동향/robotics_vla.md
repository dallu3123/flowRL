# RL for Diffusion / Flow-Matching Robot Policies and Flow-Head VLAs (2023–2026)

> Research note, compiled 2026-10-21. Method note: arxiv.org, huggingface.co, github.io and PMLR were blocked by this session's egress proxy, so full PDFs could not be fetched. Every finding below comes from search-engine snippets of primary pages (arXiv abstract/HTML, conference proceedings, official blogs, RLinf docs). Numbers marked "(secondary)" come from aggregators and should be checked against the PDF. Papers I know of but could not verify in this session are listed under Gaps, not Findings.

---

## Q1. Which methods fine-tune diffusion/flow robot policies with RL, and how? Which target flow-matching action heads specifically?

### Takeaway
The field has split into five families: (a) **likelihood-based policy gradient through the denoising chain** (DPPO → ReinFlow → πRL Flow-Noise/Flow-SDE, which scales PPO/GRPO to π0/π0.5/GR00T); (b) **frozen-base steering/residual methods** (DSRL latent-noise RL, Policy Decorator, ResiP/ResFiT, PLD, GR-RL's latent noise predictor), which do not touch the big model's weights; (c) **critic-guided sample selection or regression** (V-GPS, PA-RL, Q-VGM, value-gradient matching); (d) **advantage-conditioned / weighted offline-style RL at foundation-model scale** (RECAP/π*0.6, ARFM-type, Hierarchical Advantage Weighting) and preference methods (FlowPRO); (e) **world-model RL** (WMPO, VLA-RFT). Since late 2025, flow-matching VLAs (π0, π0.5, GR00T N1.5) have been the main RL targets in (a), (b), (c) and (d).

### Cited Findings

**(a) Policy gradient through the denoising chain (the denoising chain is treated as an MDP)**
- **DPPO, "Diffusion Policy Policy Optimization"** (Ren et al., **ICLR 2025**, arXiv 2409.00588). Fine-tunes pre-trained Diffusion Policy with policy-gradient RL (PPO over the denoising steps). The abstract reports the best overall performance and efficiency on common benchmarks among RL methods for diffusion policies, strong training stability, pixel-based simulated tasks, and zero-shot sim-to-real transfer on a long-horizon multi-stage manipulation task — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html); [arXiv](https://arxiv.org/pdf/2409.00588); [TRI page](https://www.tri.global/research/diffusion-policy-policy-optimization)
- **ReinFlow, "Fine-tuning Flow Matching Policy with Online RL"** (Zhang, Yu, Su, Wang; **NeurIPS 2025**, arXiv 2505.22094). Adds a learnable noise injection to a flow policy so that the few-step flow becomes a discrete Markov chain with tractable likelihood, then applies policy gradient. Reported results: Rectified Flow policies get an average +135.36% episode reward on legged locomotion while using fewer denoising steps and **82.63% less wall time than DPPO**. Shortcut-Model policies on state and visual manipulation get **+40.34% average success**, at four or even one denoising step, with ~23.20% less compute than fine-tuned DDIM policies. It beats DPPO on almost all Gym and Franka Kitchen tasks while cutting denoising steps from 10 (DPPO) to 4 — [NeurIPS 2025 proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/98d6f928497d9eaad9f320fb2db3040d-Abstract-Conference.html); [arXiv](https://arxiv.org/pdf/2505.22094); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/119473)
- **πRL, "Online RL Fine-tuning for Flow-based VLA Models"** (Chen et al., arXiv 2510.25889, Oct 2025; v1–v3; code in RLinf). This is the main flow-VLA RL method and the most common baseline in 2026 papers. Its premise is that RL needs action log-likelihoods, which iterative denoising makes hard to compute, and that deterministic ODE sampling "precludes exploration". It has two variants:
  - **Flow-Noise**: models denoising as a discrete-time MDP with a learnable noise network, which gives exact log-likelihoods.
  - **Flow-SDE**: converts ODE sampling into an SDE with the same marginals and builds a **two-layer MDP** (inner denoising MDP nested in the environment MDP).
  - Results: on LIBERO, few-shot SFT π0 goes from 57.6% to **97.6%** and π0.5 from 77.1% to **98.3%**. On ManiSkill (4,352 pick-and-place tasks, 320 parallel envs), π0 goes from 41.6% to 85.7% and π0.5 from 40.0% to 84.8%. v3 adds zero-shot sim-to-real via a 3D-Gaussian-Splatting-rendered simulator — [arXiv HTML v3](https://arxiv.org/html/2510.25889v3); [v2](https://arxiv.org/html/2510.25889v2); [PDF](https://arxiv.org/pdf/2510.25889)
- **πRL on GR00T N1.5** (flow-matching DiT action head). PPO with Flow-SDE raised LIBERO average success from **52.5% to 89.9%**. Only the action expert was updated, and dropout layers were replaced with identity because dropout destabilized online RL / corrupted log-prob computation — [πRL arXiv](https://arxiv.org/pdf/2510.25889); [RLinf GR00T example](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html)
- **RLinf / RLinf-VLA** (arXiv 2510.06710; listed as **RSS 2026** on the project docs). A unified VLA+RL training system. OpenVLA-OFT on LIBERO goes from 65.43% to **98.11%** average with RLinf-GRPO (LIBERO-Long 93.95%). It reports a 1.34–2.27× speedup over SimpleVLA-RL, and finds that **PPO consistently outperforms GRPO and is more stable on ManiSkill** — [RLinf-VLA docs](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_vla.html); [arXiv](https://arxiv.org/pdf/2510.06710)
- **SAC Flow** (Zhang, Yu, Zhang et al., **ICLR 2026**, arXiv 2509.25756). Observes that a flow rollout is algebraically a **residual recurrent computation**, so backprop-through-sampling suffers RNN-style vanishing/exploding gradients. It reparameterizes the velocity network as Flow-G (gated) or Flow-T (decoded/transformer) and uses a noise-augmented rollout so SAC can train end to end. Reports SOTA on continuous control and robotic manipulation, from scratch and offline-to-online, with no distillation or surrogate objective — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66a3f3d3317da8d4b9c1579e2bc97d5-Abstract-Conference.html); [arXiv](https://arxiv.org/pdf/2509.25756)
- **Reparameterization Flow Policy Optimization (RFO)** (**ICML 2026** poster). Backpropagates jointly through flow generation and (differentiable) environment dynamics, so it never computes log-likelihoods. Evaluated on locomotion (headline: soft-body quadruped) and manipulation — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63464)
- **Uncertainty-Gated Exploration Noise** (arXiv 2609.28838, 2026). Online RL of a flow-matching VLA: a 10-step velocity-field integration produces an action chunk, and Flow-SDE adds Gaussian noise σ at each step. Evaluation uses a stochastic sampler (σ_eval = 0.25) because the deterministic sampler is never run during online RL. It targets "task collapse" in multi-task online RL; the gating mechanism and numbers were not retrievable — [arXiv](https://arxiv.org/pdf/2609.28838)
- **SA-VLA: Spatially-Aware Flow-Matching for VLA RL** (arXiv 2602.00743, 2026). Combines dense geometric rewards with an annealed exploration strategy for flow-matching VLA RL — [arXiv](https://arxiv.org/abs/2602.00743)
- **Reinforcement Learning for Real-Time VLA Policies** (arXiv 2609.18207, 2026). Its reference list cites 2026 works "EXPO-FT" and "FASTER". Content not retrievable — [arXiv](https://arxiv.org/pdf/2609.18207)
- "Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models" (arXiv 2510.09976) exists; its details were not retrieved — [arXiv](https://arxiv.org/pdf/2510.09976)

**(b) Frozen-base steering and residual RL (no gradient into the large generative model)**
- **DSRL, "Steering Your Diffusion Policy with Latent Space RL"** (Wagenmaker et al., **CoRL 2025**, PMLR v305, arXiv 2506.15799). Runs RL over the **initial latent noise** of a frozen BC diffusion/flow policy, which needs only black-box access to the base policy. It is sample-efficient, works for real-world improvement, and adapts generalist policies (π0) — [PMLR](https://proceedings.mlr.press/v305/wagenmaker25a.html); [arXiv](https://arxiv.org/abs/2506.15799)
  - Variants: DSRL-SAC (SAC in noise space) and **DSRL-NA (noise-aliased)**, which distills an action-space Q-function into noise space. The maintainers say NA is more sample-efficient, while SAC is somewhat cheaper computationally — [GitHub ajwagen/dsrl](https://github.com/ajwagen/dsrl)
  - The π0 steering code covers LIBERO, Aloha sim and a real Franka — [GitHub dsrl_pi0](https://github.com/nakamotoo/dsrl_pi0). The RLinf version trains a ~500K-parameter SAC agent whose output replaces π0's starting noise — [RLinf DSRL docs](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html)
- **Policy Decorator** (Yuan et al., **ICLR 2025**, arXiv 2412.13630). A model-agnostic residual policy trained with online RL on top of a frozen BC model (Behavior Transformer or Diffusion Policy). It uses a bounded residual action, controlled/progressive exploration and sparse reward. Evaluated on 8 tasks in ManiSkill and Adroit, it improves the base policies while keeping smooth IL motion — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html); [slides](https://iclr.cc/media/iclr-2025/Slides/28950.pdf)
- **ResiP, "From Imitation to Refinement – Residual RL for Precise Assembly"** (Ankile et al., arXiv 2407.16677). Residual RL on top of a chunked diffusion BC policy for precise assembly — [arXiv](https://arxiv.org/html/2407.16677v4)
- **ResFiT, "Residual Off-Policy RL for Finetuning BC Policies"** (Ankile, Jiang, Duan, Shi, Abbeel, Nagabandi; Amazon FAR; **ICLR 2026**, arXiv 2509.19301). Freezes the base BC policy and trains a residual with off-policy RL; the critic scores the full action (base + residual). Needs only sparse binary rewards. Claims the first real-world RL on a humanoid with dexterous hands. Success goes from ~14–23% to ~64% with ~15–76 minutes of real interaction (secondary summary) — [arXiv](https://arxiv.org/pdf/2509.19301); [ICLR 2026](https://iclr.cc/virtual/2026/10012529); [summary (secondary)](https://liner.com/review/residual-offpolicy-rl-for-finetuning-behavior-cloning-policies)
- **PLD, "Self-Improving VLA Models with Data Generation via Residual RL"** (Xiao et al., NVIDIA/UT Austin, **ICLR 2026**, arXiv 2511.00091). Probe–Learn–Distill:
  - Probe: freeze the VLA and train lightweight residual actors off-policy.
  - Learn: hybrid rollouts that bias interventions toward states the base policy visits.
  - Distill: SFT the VLA on the curated trajectories. Works for both flow-matching and autoregressive heads.
  - Results: ~99% on LIBERO, >50% gains on SimplerEnv, and 100% on real Franka and YAM-arm tasks. Argues that RL-generated, policy-aligned data can beat teleop-only demonstrations — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/163a875ab0c2631ed30b7435c4c9b8f1-Abstract-Conference.html); [UT RPL](https://rpl.cs.utexas.edu/publications/2026/04/01/xiao-iclr26-pld/)
- **GR-RL** (ByteDance Seed, arXiv 2512.01801, Dec 2025). Turns a generalist VLA into a specialist for long-horizon dexterous tasks:
  - Learns a VL-conditioned task-progress estimate (Q-values from sparse-reward offline RL) and uses it to filter noisy demonstrations.
  - Applies morphological symmetry augmentation.
  - **Online RL learns a latent-space noise predictor** (DSRL-like steering) to align the policy with deployment behavior.
  - Reports the first learning-based policy to lace a shoe through multiple eyelets, at **83.3% success** — [arXiv](https://arxiv.org/html/2512.01801v2); [Seed blog](https://seed.bytedance.com/en/blog/gr-rl-a-breakthrough-in-dexterous-manipulation-achieving-the-first-real-robot-shoe-lacing-with-reinforcement-learning)
- **Flow Reversal Steering** (arXiv 2606.13675, 2026) improves generalist policies and uses DSRL as a baseline on a DROID Franka setup — [arXiv](https://arxiv.org/pdf/2606.13675)

**(c) Critic-guided selection, regression or value-gradient matching (likelihood-free)**
- **V-GPS, "Steering Your Generalists"** (Nakamoto et al., **CoRL 2024**, arXiv 2410.13816). At deployment, re-ranks a generalist policy's sampled actions with an offline-RL value function. Needs no weights and no fine-tuning. One value function improved **five** generalist policies across 12 tasks on multiple platforms — [PMLR v270](https://proceedings.mlr.press/v270/nakamoto25a.html); [arXiv](https://arxiv.org/pdf/2410.13816)
- **SeeQ** (arXiv 2609.22085, Sept 2026). Generalist value functions for long-horizon manipulation; learns subtask-local Q-values to shorten the prediction horizon for V-GPS-style steering — [arXiv](https://arxiv.org/pdf/2609.22085)
- **PA-RL, "Policy Agnostic RL"** (Mark, Gao, Sampaio, Srirama, Sharma, Finn, Kumar; CMU/Stanford; **ICLR 2025**, arXiv 2412.06685). Policy improvement is supervised learning on Q-optimized actions: sample several actions from the base policy, re-rank them with Q (global step), take gradient steps on the action (local step), then distill. Works with diffusion and autoregressive transformer policies. Fine-tunes **OpenVLA-7B with Cal-QL from 40% to 70% in 40 minutes of real-world interaction**. Caches 16 sampled OpenVLA actions per state. Up to 2× better performance/sample efficiency than prior offline/online methods. Notes that SAC reparameterization is unstable for diffusion policies — [arXiv](https://arxiv.org/pdf/2412.06685); [ICLR 2025](https://iclr.cc/virtual/2025/10000497)
- **Q-VGM** (arXiv 2606.08015, June 2026; titles differ across v1–v3, latest "Q-Guided Value-Gradient Matching for Offline-to-Online RL of Flow-Matching VLA Policies"). Off-policy critic-based training of the flow action expert. It turns the Q-gradient into a denoising-time value-gradient field (building on VGG-Flow) and avoids both likelihoods and backprop through the denoising chain. Latest version on LIBERO:
  - Offline: 4-suite average 84.6% → 90.7% with only 150 noisy rollouts per suite.
  - Online: **98.4 ± 0.2%** over 3 seeds, vs PPO with Flow-SDE (πRL) at 97.4%.
  - PPO needs **4.43–6.14× more episodes** (mean 5.35×).
  - Earlier versions report 75.0→92.5% and 79.0→92.5%, so the baselines changed between versions — [arXiv latest](https://arxiv.org/html/2606.08015); [v1](https://arxiv.org/html/2606.08015v1); [v2](https://arxiv.org/html/2606.08015v2)

**(d) Advantage-conditioned, weighted and preference-based post-training at VLA scale**
- **π\*0.6 / RECAP** (Physical Intelligence, arXiv 2511.14759, Nov 2025). "RL with Experience and Corrections via Advantage-conditioned Policies":
  - A multi-task value function predicts outcome from observation and language.
  - The π0.6 VLA (flow action expert) is conditioned on a **binarized advantage indicator**; at inference it is conditioned on "good".
  - Data: demonstrations, autonomous rollouts and expert teleop interventions.
  - Pipeline: offline-RL pre-training of a generalist, then on-robot specialization.
  - Tasks: laundry folding in real homes, box assembly, espresso making — [arXiv](https://arxiv.org/abs/2511.14759); [PI blog](https://pi.website/blog/pistar06)
  - Adding on-robot data **more than doubles throughput** on diverse laundry and espresso and roughly **halves the failure rate**. The paper's comparison is offline-RL+SFT vs final π\*0.6; press coverage words it as vs "a base model trained without RL", so the baselines conflict. Box-assembly failures are mostly time-outs — [arXiv PDF](https://arxiv.org/pdf/2511.14759); [AIM Media House (secondary)](https://aimmediahouse.com/ai-startups/robots-learn-from-doing-physical-intelligence)
  - Demo-video run lengths (secondary, not paper metrics): espresso ~13 h; laundry ~3 h at ~3 min/item; box ~1 h at ~2.5 min/box. No detailed intervention logs were published — [AIM Media House](https://aimmediahouse.com/ai-startups/robots-learn-from-doing-physical-intelligence)
- **Hierarchical Advantage Weighting for Online RL Fine-Tuning of VLAs from Sparse Episode Outcomes** (arXiv 2606.17043, 2026). Exists; details not retrieved — [arXiv](https://arxiv.org/pdf/2606.17043)
- **FlowPRO** (arXiv 2606.05468, June 2026). Reward-free offline RFT of flow-matching VLAs. Its RPRO loss combines a contrastive (DPO-style) term with a proximal regularizer that anchors the implicit reward magnitude, removing the reward-hacking failure mode of plain Flow-DPO. Preference pairs come from teleoperated **intervention-and-rollback** on a real robot. On 4 long-horizon bimanual tasks with two π0-family bases it gets the best success and the shortest completion time against DAgger, π\*0.6-style advantage conditioning and TPO. Secondary figures: 92–99% vs DAgger 80–88%, π0.6\*-style 90–94%, TPO 87–95%; another aggregator says 98–99%, so the figures conflict. Limitations: single platform, rollback timing set by a human — [arXiv](https://arxiv.org/pdf/2606.05468); [EmergentMind (secondary)](https://www.emergentmind.com/papers/2606.05468)
- **ConRFT** (Chen et al., **RSS 2025**, arXiv 2502.05450). Two stages: an offline stage of BC + Q-learning, then an online stage that fine-tunes the VLA (Octo-based) through a **consistency-policy** head with human interventions (HIL). Results: **96.3% average success on 8 real tasks within 45–90 minutes** of online fine-tuning, +144% over SFT baselines, 1.9× shorter episodes; HG-DAgger 65%, PA-RL 71.3%. Reward-design sensitivity is a stated limitation (secondary) — [RSS proceedings](https://roboticsproceedings.org/rss21/p019.html); [arXiv](https://arxiv.org/pdf/2502.05450)
- **RL-100** (Lei, Li, Yu, Wei, ..., Huazhe Xu; arXiv 2510.14830, Oct 2025, v1–v4). Three-stage real-world RL on diffusion visuomotor policies:
  - IL, then offline RL with PPO-style updates *inside the denoising process* gated by off-policy evaluation, then online RL — all under one PPO-style objective.
  - A **consistency-distillation head** compresses multi-step diffusion into a 1-step policy for high-frequency control.
  - Reports **100% success on 900/900 episodes** (up to 250/250 consecutive). Zero-shot to novel dynamics at 92.5% average; few-shot adaptation to 86.7% after 1–3 h.
  - Tasks: Push-T, bowling, pouring, cloth folding, unscrewing, orange juicing. 7 tasks in v1/v3 vs 8 in v4. Supports single-step and chunked control — [arXiv](https://arxiv.org/html/2510.14830v1); [v4](https://arxiv.org/pdf/2510.14830v4)

**(e) Autoregressive/tokenized VLAs (context, not flow heads)**
- **VLA-RL** (Lu et al., arXiv 2505.18719). Online PPO for autoregressive OpenVLA-7B with a VLM-based robotic process reward model. +4.5% over the strongest SFT baseline on 40 LIBERO tasks, roughly matching π0-FAST; claims early test-time scaling — [arXiv](https://arxiv.org/html/2505.18719v1)
- **SimpleVLA-RL** (**ICLR 2026**, arXiv 2509.09674). GRPO-style RL with exploration enhancements on OpenVLA-OFT; reaches SoTA on LIBERO and beats π0 on RoboTwin — [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html)
- **RIPT-VLA** (arXiv 2505.17016). Interactive post-training with sparse binary reward, dynamic rollout sampling and leave-one-out advantages. OpenVLA-OFT reaches 97.5%; a model starts at 4% after SFT with a single demo and reaches 97% within 15 iterations — [arXiv](https://arxiv.org/pdf/2505.17016)

**World-model RL**
- **WMPO** (arXiv 2511.09515, **ICLR 2026**). On-policy GRPO inside a pixel-space world model aligned with VLA pretrained features. Allows repeated rollouts from the same start state (needed for GRPO). Improves sample efficiency and self-correction on MimicGen and a real robot — [arXiv](https://arxiv.org/pdf/2511.09515); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10007263)
- **VLA-RFT** (arXiv 2510.00406). A world model serves as the simulator, with verified rewards from goal references. GRPO on a **flow-matching head pre-trained on demos**, keeping a small flow-matching MSE term for stability. Strong results in **<400 fine-tuning steps**, robust under perturbation — [arXiv](https://arxiv.org/html/2510.00406v1)

### Inferences
- Methods that work with flow-matching heads specifically: ReinFlow, πRL (Flow-Noise, Flow-SDE), RLinf (π0/π0.5/GR00T), SAC Flow, RFO, Q-VGM, FlowPRO, Uncertainty-Gated noise, SA-VLA, VLA-RFT, DSRL (π0), GR-RL, PLD, RECAP (π0.6 has a flow action expert). DPPO, RL-100, Policy Decorator, ResiP and IDQL-style methods target DDPM/DDIM diffusion policies. ConRFT targets Octo with a consistency head.
- There is a clear split between **likelihood-requiring** methods (PPO/GRPO via injected stochasticity: DPPO, ReinFlow, πRL) and **likelihood-free** ones (Q-gradient/value matching, steering, reranking, advantage conditioning, preference, reparameterized BPTT). The likelihood-free side is growing fastest in 2026 (Q-VGM, FlowPRO, RFO), apparently because injected noise brings exploration/evaluation mismatch (σ_eval issue) and on-policy sample cost.
- A common recipe is emerging for real-robot deployment: **freeze the large VLA, run RL in a small module (noise policy, residual or critic), then distill back via SFT** (PLD, GR-RL, PA-RL, DSRL-NA). Direct full-parameter PPO on flow VLAs is mostly done in simulation (LIBERO, ManiSkill) with massive parallelism.

### Gaps
- Could not verify from primary sources (excluded from findings): Flow Q-Learning (Park, Li, Levine; believed ICML 2025), FPO "Flow Policy Optimization" (McAllister et al., 2025), IDQL, Diffusion-QL, EXPO, ARFM ("adaptive RL flow matching" for VLA), Hume, TGRPO, the GR00T official RL recipe, Octo/RDT-specific RL papers. Snippets cite "EXPO-FT" and "FASTER" (2026) without details.
- No full per-task tables for DPPO (Robomimic, Furniture-Bench) or DSRL (episode counts per task) were retrievable.
- There is no official NVIDIA RL recipe for GR00T; RL on GR00T comes from third parties (RLinf/πRL) — [RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html). I found **no RL-fine-tuning paper targeting RDT** specifically.

---

## Q2. How are action chunking, multi-step denoising and large model size handled? Sample efficiency on real robots?

### Takeaway
The open questions are (i) whether the MDP is over chunks or steps, (ii) how to get likelihoods and exploration out of a deterministic K-step ODE (noise injection or SDE versus likelihood-free critics), and (iii) how to avoid updating billions of parameters (update only the action expert, steer noise, train residuals, or distill). Real-robot RL on diffusion/flow policies now commonly reports **tens of minutes to a few hours** of interaction. Full on-policy PPO/GRPO on flow VLAs still mostly lives in massively parallel simulation or world models.

### Cited Findings
**Action chunking**
- **Q-chunking, "RL with Action Chunking"** (Li, Zhou, Levine; **NeurIPS 2025**, arXiv 2507.07969). Runs TD learning directly in chunked action space for long-horizon, sparse-reward offline-to-online RL, with gains over prior offline-to-online methods on manipulation benchmarks — [NeurIPS 2025 proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html); [arXiv](https://arxiv.org/html/2507.07969v1)
- In flow-VLA RL, one chunk is an action, and its likelihood is the product of the K denoising Gaussian transitions (Flow-SDE; K = 10 steps per chunk in 2609.28838) — [Uncertainty-gated paper](https://arxiv.org/pdf/2609.28838); [πRL](https://arxiv.org/html/2510.25889v3)
- RL-100 supports both single-action and chunked control in one framework — [arXiv](https://arxiv.org/html/2510.14830v1)
- "Adaptive Action Chunking at Inference-time for VLA Models" (arXiv 2604.04161) also exists — [arXiv](https://arxiv.org/pdf/2604.04161)

**Multi-step denoising**
- DPPO treats the K denoising steps as inner MDP steps — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html). ReinFlow reduces this to 4 or even 1 step with learned noise and saves 82.63% wall time vs DPPO — [arXiv](https://arxiv.org/pdf/2505.22094).
- πRL's two-layer MDP (Flow-SDE) or noise network (Flow-Noise) makes π0's ODE stochastic — [arXiv](https://arxiv.org/html/2510.25889v3).
- SAC Flow shows that backpropagating through K flow steps behaves like a residual RNN with vanishing/exploding gradients, and fixes it architecturally (gated or transformer velocity) — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66a3f3d3317da8d4b9c1579e2bc97d5-Abstract-Conference.html).
- Q-VGM avoids both likelihoods and BPTT through the chain by matching a denoising-time value-gradient field — [arXiv](https://arxiv.org/html/2606.08015).
- **Latency**: RL-100 distills multi-step diffusion into a 1-step consistency head for high-frequency control — [arXiv](https://arxiv.org/html/2510.14830v1). ConRFT uses a consistency policy — [RSS 2025](https://roboticsproceedings.org/rss21/p019.html).
- **Train/eval stochasticity mismatch**: online flow RL evaluates with the stochastic sampler (σ_eval = 0.25) because the deterministic ODE sampler never runs during RL — [2609.28838](https://arxiv.org/pdf/2609.28838).

**Large model size**
- πRL on GR00T updates only the action expert; dropout must be removed for stable log-probs — [πRL](https://arxiv.org/pdf/2510.25889); [RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html). RLinf lists LoRA RL for GR00T as "under test" — [RLinf docs](https://rlinf.readthedocs.io/en/release-v0.2/rst_source/examples/embodied/gr00t.html).
- DSRL trains a ~500K-parameter noise actor over a frozen π0 — [RLinf DSRL](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dsrl.html).
- PA-RL caches 16 OpenVLA samples per state so the critic trains at small-policy speed, and does not update OpenVLA during offline RL — [arXiv](https://arxiv.org/pdf/2412.06685).
- PLD freezes the VLA backbone for residual RL, then distills — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/163a875ab0c2631ed30b7435c4c9b8f1-Abstract-Conference.html).
- RECAP avoids policy gradient entirely: advantage-conditioned SFT on a binarized indicator — [arXiv](https://arxiv.org/pdf/2511.14759).

**Real-robot sample efficiency (as reported)**

| Method | Base policy | Real-robot result | Interaction budget | Source |
|---|---|---|---|---|
| PA-RL | OpenVLA-7B | 40% → 70% | 40 min | [arXiv](https://arxiv.org/pdf/2412.06685) |
| ConRFT | Octo + consistency head | 96.3% average, 8 tasks | 45–90 min, with human interventions | [RSS 2025](https://roboticsproceedings.org/rss21/p019.html) |
| ResFiT | BC policy, humanoid with dexterous hands | ~14–23% → ~64% (secondary) | ~15–76 min | [summary](https://liner.com/review/residual-offpolicy-rl-for-finetuning-behavior-cloning-policies) |
| RL-100 | Diffusion visuomotor policy | 900/900; few-shot adaptation to 86.7% | 1–3 h of extra training for adaptation | [arXiv](https://arxiv.org/html/2510.14830v1) |
| PLD | VLA (Franka, YAM) | 100% | Not stated | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/163a875ab0c2631ed30b7435c4c9b8f1-Abstract-Conference.html) |
| GR-RL | Generalist VLA | 83.3% shoe lacing | Not stated | [arXiv](https://arxiv.org/html/2512.01801v2) |
| π\*0.6 / RECAP | π0.6 | >2× throughput, ~½ failures | Not stated | [arXiv](https://arxiv.org/pdf/2511.14759) |

- Simulation sample efficiency: Q-VGM reports PPO (Flow-SDE) needs 4.43–6.14× more episodes on LIBERO — [arXiv](https://arxiv.org/html/2606.08015). VLA-RFT reports <400 fine-tuning steps in a world model — [arXiv](https://arxiv.org/html/2510.00406v1).

### Inferences
- Off-policy or likelihood-free methods (Q-VGM, PA-RL, DSRL-NA, residual SAC) give 5×-class sample-efficiency gains over on-policy PPO through the chain, which is why they dominate the real-robot results. On-policy methods dominate large-parallel simulation (πRL with 320 envs).
- Most real-robot wins come with human interventions (ConRFT, RECAP, FlowPRO rollback) or a frozen base. "Autonomous" interaction counts are rarely separated from human-correction time, which makes comparisons across papers unreliable.
- Chunk-level critics (Q-chunking) and chunk-level likelihoods (Flow-SDE) are both in use, but no paper I found compares chunk-level vs step-level credit assignment for flow VLAs head to head.

### Gaps
- Real-world interaction counts for π\*0.6, PLD and GR-RL were not in the retrievable snippets.
- I could not confirm how DPPO and ReinFlow handle chunk length vs execution horizon, or any ablations on chunk size during RL.

---

## Q3. Which 2025–2026 papers are state of the art, and what do they report?

### Takeaway
SOTA by setting:
- **Simulation (flow VLAs on LIBERO)**: πRL (π0.5 at 98.3%) and Q-VGM (98.4% with ~5× fewer episodes); LIBERO is close to saturated.
- **ManiSkill multi-task**: πRL at ~85%.
- **Real robot**: RL-100 (900/900, diffusion), π\*0.6/RECAP (company-scale deployment), GR-RL (shoe lacing at 83.3%), ConRFT (96.3% in under 90 minutes), PLD (100% on Franka/YAM), FlowPRO (preference RFT, 2026).

### Cited Findings

**Venue-verified (proceedings or virtual-site listing)**

| Paper | Venue | Key result | Source |
|---|---|---|---|
| DPPO | ICLR 2025 | Best overall among diffusion RL methods; zero-shot sim-to-real | [link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c0749c39aaff9e9e4c91f7118bf21b1e-Abstract-Conference.html) |
| Policy Decorator | ICLR 2025 | 8 tasks, ManiSkill/Adroit | [link](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html) |
| PA-RL | ICLR 2025 | OpenVLA 40→70% in 40 min real | [link](https://iclr.cc/virtual/2025/10000497) |
| ConRFT | RSS 2025 | 96.3%, 8 real tasks, 45–90 min | [link](https://roboticsproceedings.org/rss21/p019.html) |
| DSRL | CoRL 2025 | Latent-noise steering incl. π0, real world | [link](https://proceedings.mlr.press/v305/wagenmaker25a.html) |
| ReinFlow | NeurIPS 2025 | +40.34% manipulation success, 82.63% less wall time vs DPPO | [link](https://papers.nips.cc/paper_files/paper/2025/hash/98d6f928497d9eaad9f320fb2db3040d-Abstract-Conference.html) |
| Q-chunking | NeurIPS 2025 | Chunked TD for offline-to-online | [link](https://papers.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) |
| SimpleVLA-RL | ICLR 2026 | SoTA LIBERO; beats π0 on RoboTwin | [link](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html) |
| SAC Flow | ICLR 2026 | Stable off-policy flow RL | [link](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66a3f3d3317da8d4b9c1579e2bc97d5-Abstract-Conference.html) |
| PLD | ICLR 2026 | ~99% LIBERO, 100% real | [link](https://proceedings.iclr.cc/paper_files/paper/2026/hash/163a875ab0c2631ed30b7435c4c9b8f1-Abstract-Conference.html) |
| ResFiT | ICLR 2026 | Real humanoid dexterous RL | [link](https://iclr.cc/virtual/2026/10012529) |
| WMPO | ICLR 2026 | World-model GRPO | [link](https://iclr.cc/virtual/2026/poster/10007263) |
| RFO | ICML 2026 | Reparameterized flow PG | [link](https://icml.cc/virtual/2026/poster/63464) |
| RLinf-VLA | RSS 2026 (per project docs) | OpenVLA-OFT 65.4→98.1% LIBERO | [link](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_vla.html) |
| V-GPS (older baseline) | CoRL 2024 | — | [link](https://proceedings.mlr.press/v270/nakamoto25a.html) |

**arXiv-only as of retrieval (venue not confirmed)**
- πRL (2510.25889): LIBERO π0 57.6→97.6%, π0.5 77.1→98.3%; ManiSkill π0 41.6→85.7% — [arXiv](https://arxiv.org/html/2510.25889v3)
- π\*0.6/RECAP (2511.14759) — [arXiv](https://arxiv.org/abs/2511.14759)
- RL-100 (2510.14830) — [arXiv](https://arxiv.org/html/2510.14830v1)
- GR-RL (2512.01801) — [arXiv](https://arxiv.org/html/2512.01801v2)
- VLA-RL (2505.18719) — [arXiv](https://arxiv.org/html/2505.18719v1)
- RIPT-VLA (2505.17016) — [arXiv](https://arxiv.org/pdf/2505.17016)
- VLA-RFT (2510.00406) — [arXiv](https://arxiv.org/html/2510.00406v1)
- 2026 preprints:
  - Q-VGM (2606.08015) — [arXiv](https://arxiv.org/html/2606.08015)
  - FlowPRO (2606.05468) — [arXiv](https://arxiv.org/pdf/2606.05468)
  - Hierarchical Advantage Weighting (2606.17043) — [arXiv](https://arxiv.org/pdf/2606.17043)
  - SA-VLA (2602.00743) — [arXiv](https://arxiv.org/abs/2602.00743)
  - Self-Correcting VLA via Sparse World Imagination (2602.21633) — [arXiv](https://arxiv.org/pdf/2602.21633)
  - Flow Reversal Steering (2606.13675) — [arXiv](https://arxiv.org/pdf/2606.13675)
  - RL for Real-Time VLA Policies (2609.18207) — [arXiv](https://arxiv.org/pdf/2609.18207)
  - Uncertainty-Gated Exploration Noise (2609.28838) — [arXiv](https://arxiv.org/pdf/2609.28838)
  - SeeQ (2609.22085) — [arXiv](https://arxiv.org/pdf/2609.22085)
  - "The Gaussian Is Enough" (2609.27070) — [arXiv](https://arxiv.org/pdf/2609.27070)
  - Hy-Embodied-0.5-VLA real-world stack (2606.14409) — [arXiv](https://arxiv.org/pdf/2606.14409)
- **Counterpoint, "The Gaussian Is Enough"** (arXiv 2609.27070, 2026). When fine-tuning LBMs (LBM 1.0, π0.5, GR00T N1.5) on task demos, replacing the Gaussian source prior with closer-to-target non-Gaussian priors does **not** help, across >100k sim rollouts and 1,250 hardware rollouts (rollout counts from a secondary summary). This questions prior-shaping as a fine-tuning lever — [arXiv](https://arxiv.org/pdf/2609.27070); [Digg summary (secondary)](https://digg.com/ai/3r82pvvj)

### Inferences
- On LIBERO, top methods now score 97–99%, so differences are within noise. Recent papers differentiate on **sample efficiency** (Q-VGM's 5.35×) and real-robot or long-horizon tasks instead. A survey should treat LIBERO-average as saturated.
- Industrial labs (PI, ByteDance Seed, NVIDIA) favor offline/advantage-conditioned or steering approaches on real hardware. Academic labs favor on-policy PPO in simulation. This split itself is worth stating.

### Gaps
- I could not confirm whether πRL, RL-100, GR-RL or π\*0.6 were accepted at a 2026 venue.
- A full ICML 2026 / CoRL 2026 accepted list for this topic was not retrievable. CoRL 2026 (likely Nov 2026) papers may not be public yet.

---

## Q4. What benchmarks are used?

### Takeaway
- **Simulation**: LIBERO (the de facto flow-VLA RL benchmark, now saturated), ManiSkill (multi-task and large-scale parallel), RoboTwin (bimanual), SimplerEnv (real-to-sim for generalists), MimicGen, Robomimic, Franka Kitchen, Gym locomotion, Adroit, and Furniture-Bench (assembly).
- **Real robots**: Franka, bimanual (Aloha, YAM), dexterous humanoids, and long-horizon home tasks.

### Cited Findings
- **LIBERO**: πRL, RLinf-VLA, Q-VGM, PLD, VLA-RL (40 tasks), RIPT-VLA, SimpleVLA-RL, DSRL-π0 — [πRL](https://arxiv.org/html/2510.25889v3); [RLinf-VLA](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_vla.html); [Q-VGM](https://arxiv.org/html/2606.08015); [VLA-RL](https://arxiv.org/html/2505.18719v1)
- **ManiSkill**: πRL (4,352 pick-and-place tasks, 320 parallel envs), RLinf (PPO > GRPO), Policy Decorator — [πRL](https://arxiv.org/html/2510.25889v3); [RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_vla.html); [Policy Decorator](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html)
- **RoboTwin**: SimpleVLA-RL, RLinf-VLA — [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html)
- **SimplerEnv**: PLD (>50% gains) — [PLD](https://proceedings.iclr.cc/paper_files/paper/2026/hash/163a875ab0c2631ed30b7435c4c9b8f1-Abstract-Conference.html)
- **MimicGen**: WMPO — [arXiv](https://arxiv.org/pdf/2511.09515)
- **Franka Kitchen, Gym locomotion, visual manipulation**: ReinFlow — [arXiv](https://arxiv.org/pdf/2505.22094)
- **Adroit**: Policy Decorator — [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c361d4117d598d4bb6568b407e9ac9-Abstract-Conference.html)
- **Aloha sim**: DSRL-π0 — [GitHub](https://github.com/nakamotoo/dsrl_pi0)
- **Real-world tasks**:
  - Franka (DSRL, PLD, Flow Reversal Steering on DROID) — [DSRL repo](https://github.com/nakamotoo/dsrl_pi0); [2606.13675](https://arxiv.org/pdf/2606.13675)
  - YAM arm (PLD)
  - Humanoid with dexterous hands (ResFiT) — [arXiv](https://arxiv.org/pdf/2509.19301)
  - 8 ConRFT tasks
  - RL-100 tasks: Push-T, bowling, pouring, cloth folding, unscrewing, juicing — [arXiv](https://arxiv.org/html/2510.14830v1)
  - Laundry, box assembly, espresso (π\*0.6) — [PI blog](https://pi.website/blog/pistar06)
  - Shoe lacing (GR-RL) — [Seed blog](https://seed.bytedance.com/en/blog/gr-rl-a-breakthrough-in-dexterous-manipulation-achieving-the-first-real-robot-shoe-lacing-with-reinforcement-learning)
  - 4 bimanual long-horizon tasks (FlowPRO) — [arXiv](https://arxiv.org/pdf/2606.05468)
- **Sim-to-real via 3DGS simulator**: πRL v3 — [arXiv](https://arxiv.org/html/2510.25889v3)
- **Robustness benchmark**: LIBERO-Plus (2510.13626) gives in-depth robustness analysis of VLAs and includes RIPT-style post-training — [arXiv](https://arxiv.org/pdf/2510.13626)

### Inferences
- There is no standard **real-world RL benchmark** and no standard reporting protocol (autonomous minutes vs intervention minutes, number of resets). Real-robot numbers across papers are not comparable.
- LIBERO is saturated, and ManiSkill/RoboTwin/SimplerEnv are the next headroom. A harder multi-task, long-horizon flow-VLA RL benchmark with fixed compute/sample budgets is missing.

### Gaps
- I could not retrieve whether DPPO's Robomimic and Furniture-Bench results were reused as baselines by flow-VLA papers. Official DPPO task lists were not fetched (arXiv blocked).

---

## Q5. Open problems and limitations, especially ML-contribution ones suited to ICML/ICLR

### Takeaway
The open ML problems are: (1) principled likelihood/exploration for deterministic flow samplers and the resulting train/eval sampler mismatch; (2) on-policy sample inefficiency compared with off-policy critics, plus stability of TD with chunked, high-dimensional actions; (3) credit assignment across the two-level (denoising × environment) MDP and across chunks; (4) multi-task interference ("task collapse") in online RL of generalists; (5) reward/value specification from sparse episode outcomes, and reward hacking of implicit-reward (DPO-style) objectives; (6) preserving the pretrained generalist's capabilities under RL (forgetting, distillation from residuals); (7) real-time constraints (few-step and consistency distillation that keeps RL gains).

### Cited Findings
- **Deterministic ODE vs exploration/likelihood**: "deterministic ODE sampling of flow-based VLAs precludes exploration," and computing log-likelihood through iterative denoising is hard — [πRL](https://arxiv.org/html/2510.25889v2). Fixes so far are heuristics: a learned noise network, ODE→SDE, or a σ schedule.
- **Train/eval sampler mismatch**: evaluation has to use the stochastic sampler (σ_eval = 0.25) because the deterministic sampler never runs during RL — [2609.28838](https://arxiv.org/pdf/2609.28838).
- **Task collapse in multi-task online RL** of flow VLAs is serious enough that a 2026 paper's title is about suppressing it with uncertainty-gated noise — [2609.28838](https://arxiv.org/pdf/2609.28838).
- **Sample inefficiency of on-policy PPO through the chain**: 4.43–6.14× more episodes than Q-VGM on LIBERO — [Q-VGM](https://arxiv.org/html/2606.08015).
- **GRPO vs PPO**: PPO consistently beats GRPO and is more stable on ManiSkill — [RLinf-VLA](https://rlinf.readthedocs.io/en/latest/rst_source/publications/rlinf_vla.html).
- **Gradient pathology** when backpropagating through flow sampling (RNN-like vanishing/exploding gradients) — [SAC Flow](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c66a3f3d3317da8d4b9c1579e2bc97d5-Abstract-Conference.html).
- **SAC reparameterization is unstable for diffusion policies** and intractable for autoregressive VLAs — [PA-RL](https://arxiv.org/pdf/2412.06685).
- **Reward hacking in preference/implicit-reward RFT**: plain Flow-DPO drifts in implicit-reward magnitude; FlowPRO adds a proximal anchor. Its own limits are a single platform and a human-timed rollback — [FlowPRO](https://arxiv.org/pdf/2606.05468).
- **Suboptimal, noisy human demos** for dexterous tasks, and a train/deploy mismatch — [GR-RL](https://arxiv.org/html/2512.01801v2).
- **Reward design sensitivity** (ConRFT limitation, secondary) — [labellerr summary](https://www.labellerr.com/blog/reinforcement-learning-techniques-vla-models/amp/).
- **Infrastructure fragility**: dropout corrupts log-probs and LoRA-RL is still "under test" for GR00T — [RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html).
- **Reproducibility / transparency** of industrial claims: no published intervention logs or failure-case disclosure for π\*0.6 — [AIM Media House](https://aimmediahouse.com/ai-startups/robots-learn-from-doing-physical-intelligence).
- **Prior shaping doesn't help fine-tuning**, so improvements have to come from the objective or data, not the source distribution — [The Gaussian Is Enough](https://arxiv.org/pdf/2609.27070).
- **Real-world GRPO needs repeated resets from the same state**, which motivates world models — [WMPO](https://arxiv.org/pdf/2511.09515).
- **Visual RL survey challenges**: sample efficiency, generalization, safe deployment, stabilizing policy optimization under complex rewards, scalable long-horizon reward design — [Wu et al. survey, arXiv 2508.08189](https://arxiv.org/html/2508.08189v2).

### Inferences (candidate ICML/ICLR-grade directions, derived from the gaps above)

1. **Principled exploration for flow policies.** Learn or derive the noise schedule (σ per step, per state) as part of the policy, with a provable link between SDE noise and the deterministic-ODE deployment policy. Possible routes: entropy-regularized flow RL with tractable entropy bounds, or a KL-to-ODE regularizer. This targets both the σ_eval mismatch and task collapse.
2. **Unifying likelihood-free critics with chunked TD.** Q-VGM-style value-gradient matching combined with Q-chunking's chunk-level TD. Analyze bias/variance of chunk vs step critics for flow heads, and compare their sample efficiency against πRL at fixed budgets.
3. **Two-level MDP credit assignment.** Which denoising steps should receive gradient? Adaptive step selection, or learning only late or early steps, with theory for variance reduction in the denoising × environment MDP. This connects to ReinFlow's 1–4-step results.
4. **Multi-task online RL without collapse.** Per-task trust regions, gradient surgery or uncertainty-gated updates for generalist flow VLAs, evaluated on ManiSkill multi-task and RoboTwin instead of saturated LIBERO.
5. **Steering vs fine-tuning theory.** When is noise-space steering (DSRL) expressive enough? Characterize the reachable action distribution under a frozen flow map, and design hybrids that switch from steering to weight updates. "The Gaussian Is Enough" suggests source-prior changes alone are weak.
6. **Robust implicit-reward / advantage-conditioned objectives.** Compare and analyze RECAP-style binarized advantage conditioning, advantage weighting from sparse outcomes (2606.17043) and proximal preference (FlowPRO) as instances of one KL-regularized objective, with guarantees against reward hacking.
7. **Distillation that keeps RL gains.** Distill residual/steering policies into the base VLA (PLD, PA-RL), and keep the gains through consistency or one-step distillation (RL-100). Formalize forgetting of generalist skills and measure it on held-out tasks.
8. **World-model RL with calibrated uncertainty** for flow VLAs (WMPO, VLA-RFT): exploitation of model errors by expressive generative policies is a natural ML problem.

### Gaps
- I found **no dedicated survey on RL for diffusion/flow VLAs** in searchable form. The nearest is the general visual-RL survey (2508.08189) — [arXiv](https://arxiv.org/html/2508.08189v2).
- Limitation sections of πRL, π\*0.6, RL-100, GR-RL and DSRL could not be read in full (arXiv fetch blocked), so author-stated limitations for those papers are not quoted here.
- Safety during real-world RL exploration with flow VLAs was not covered by any retrieved source.
