# FaithfulSAE

## Overview
- FaithfulSAE (arXiv:2506.17673, UCL) is a data-centric approach to training more faithful sparse autoencoders
- Core idea: train explainers on the model's own synthetic outputs (unconditional generation from BOS token) instead of external datasets
- This eliminates "fake features" — spurious latent dimensions that activate on data outside the model's generalization capabilities
- Not an architectural or loss function change — purely about training data selection

---

## Method
- Generate synthetic data: unconditional sampling from BOS token, temperature=1.0, top_p=0.9, repetition_penalty=1.1
- Train explainer on synthetic data using same architecture and hyperparameters as standard training
- Evaluate the trained explainer on OOD data (no test-time adaptation — same as TERM pattern)

---

## As a Baseline in Our Framework
- FaithfulSAE is a training-time intervention (unlike GAE/SAEBoost which adapt at test time)
- Integration follows TERM pattern: load pre-trained checkpoint, evaluate directly
- Hypothesis: explainers trained on synthetic data capture genuine model-internal features → more robust under distribution shift

---

## Hyperparameters

### Generation (from paper code: generate_datasets.py)
| Parameter | Value |
|-----------|-------|
| Prompt | BOS token only |
| Temperature | 1.0 |
| top_p | 0.9 |
| repetition_penalty | 1.1 |
| max_tokens | 1024 |
| n_sequences | 400,000 |
| Seed | 42 |

### Training
| Parameter | GPT-2 | Pythia-410M | Pythia-1.4B |
|-----------|-------|-------------|-------------|
| Layer | 8 | 15 | 15 |
| Dict Size | 24,576 | 32,768 | 65,536 |
| Training Tokens | 100M | 100M | 100M |
| LR | 2e-5 | 2e-5 | 2e-5 |
| L1 Coeff | 6e-5 | 5.5e-5 | 2e-5 |
| Context Size | 128 | 128 | 128 |

Note: Architecture HPs (lr, l1, dict_size) match our existing ID training. Only the dataset changes.
Note: Pythia-410M is not in the original paper — parameters interpolated from GPT-2 and Pythia-1.4B settings.

### Paper Reference Hyperparameters (Table 5)
| Parameter | GPT-2 Small | Pythia-1.4B |
|-----------|------------|-------------|
| Layer | 8 | 18 |
| Dict Size | 12,288 | 14,336 |
| TopK | 48 | 48 |
| LR | 2e-4 | 2e-4 |
| Training Tokens | 100M | 100M |

Note: Paper uses TopK SAEs (sparsify library), we use SAELens L1 transcoders. Layer/dict size differ due to our pipeline constraints.

---

## Key Differences from Other Baselines
| Baseline | When it acts | What it does |
|----------|-------------|--------------|
| Fixed | - | Uses ID-trained explainer as-is |
| TERM | Training time | Trains with tilted ERM objective |
| Retrain | Test time | Retrains on OOD data |
| SAEBoost | Test time | Adds residual OOD explainer |
| GAE | Test time | Geometric adaptation of dictionary |
| **FaithfulSAE** | **Training time** | **Trains on model's own synthetic outputs** |

---

## Entry Points
- Generation: `python gen_synthetic_data.py --model MODEL`
- Training: `bash scripts/train_id/train_faithfulsae.sh DEVICE MODEL`
- Evaluation: `bash scripts/timeshift_ood/faithfulsae.sh DEVICE`
- Pipeline: `python run_experiment.py timeshift --baselines faithfulsae --faithfulsae_checkpoint PATH`

---

## Future Scope
- Domain OOD evaluation (pipeline ready, needs eval script)
- Adversarial OOD evaluation (pipeline ready, needs eval script)
- SAE explainer type (needs separate training + evaluation)

---

## References
- Paper: arXiv:2506.17673
- Code: reference/FaithfulSAE/FaithfulSAE-main/
- HuggingFace: seonglae/faithful-saes (pre-trained SAEs, not transcoders)
