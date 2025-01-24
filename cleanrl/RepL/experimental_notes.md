# Notes on Experiments

## SAC + contrastive learning

### 24/01/2025 - \<Exps 1\>

**Setting**
* env = __Hopper-v4__
* Contrastive Learning losses implemented in the cleanrl baseline:
    * Noise Contrastive Estimation (NCE) from [Making Linear MDPs Practical via Contrastive Representation Learning](https://arxiv.org/pdf/2207.07150) (1)
    * Spectral Contrastive Loss (SpectralCL) from Pie
    * Supervised Contrastive Loss (SupervisedCL) from Ruohan [Supervised Contrastive Learning](https://arxiv.org/pdf/2004.11362.pdf)
* Goal: See if Q-function can be linearized in $<\phi(s,a), \theta>$
* Important note: the Q-function we are considering is $Q^\pi(s,a) = r(s,a) + \gamma \mathbb{E}[V^\pi (s')]$ because of a (undemonstrated) claim in App. B of (1) in which they say " for function completeness, we should and one more nonlinear component for capturing $log \pi (a|s)$ " which is present in the original Q-definition in SAC. To make a fair comparison between the SAC baseline and SAC+CL, removed this entropy term from both.

**Experiments**
* General: the baseline touches reward of 3k but it seems to converge between 1k and 2k. ![Vanilla SAC - Seed 0!](plots/250124/sac_baseline.png "Vanilla SAC")
* General CL
    * All CL losses reaches a plateaux of maximum 1k
    * Sampling 25k samples instead of 1k before the learning start doesn't seem to help
    * Training the embeddings also during the Q updates increases a lot the perfromance
    * SCE is the only one not able to stay around 1k of reward
    * All CL losses produce noisy results
    * NCE and Spectral have a rew. prediction error of 0.00something, while Supervised has an error of 0.0something. It is very low error (I guess)

**Open questions**
* Do we need some other tricks during training? For example:
    * Pretrain embeddings
    * Pretraining and freezing embeddings values
    * More extra steps for embeddings (i.e. every 1 critic training step -> n embedding training steps)
    * Gradient Clipping
* Does it work with 2 layers in Q-net?
* What are the results of the original NCE code?
* Is hopper too complicated, what are the performance on other simpler envs?
* What about PPO?