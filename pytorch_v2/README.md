# aa2codon PyTorch v2

Start with [Train in Colab](https://colab.research.google.com/github/pyypyyy/aa2codon/blob/pytorch-xielu-uncertainty/pytorch_v2/Train.ipynb), then [Analyze in Colab](https://colab.research.google.com/github/pyypyyy/aa2codon/blob/pytorch-xielu-uncertainty/pytorch_v2/Analyze.ipynb).

Choose a GPU runtime and specify CDS FASTA paths for one host. The notebooks store checkpoints and calibration reports in Google Drive. Local usage from this directory:

    pip install torch biopython captum plotly
    python -m unittest test_smoke
    python train.py path/to/cds.fasta --out models --ensemble 5

The model retains autoregressive encoder-decoder structure, uses xIELU (https://arxiv.org/abs/2411.13010), corrected attention dimensions, a hard synonymous mask, and the modern v3 architecture described below. ReLU and SwiGLU remain selectable; AdamW and cosine scheduling are new defaults. Legacy TensorFlow and initial PyTorch v2 architecture checkpoints are incompatible and must be retrained. Standard table 11 translation is used without alternative initiation-codon semantics; only standard amino acids and an optional terminal stop are supported.

Uncertainty is conditional on a common ensemble-selected prefix; it does not quantify uncertainty across all possible generated sequences. Predictive entropy is normalized by synonym count. Mean member entropy and Jensen-Shannon disagreement are proxies for aleatoric and epistemic uncertainty. Error bars are member standard deviations, not confidence intervals. Calibration uses held-out observed codons with teacher-forced prefixes, not experimental expression measurements or free-running generation. Deterministic positions are excluded from calibration metrics. Split groups exact protein sequences; near-homolog leakage remains possible unless externally computed homology cluster IDs are passed to run(). Train on a single host; multi-host conditioning is not yet implemented. Sequences exceeding max_length are excluded rather than fragmented.

Integrated gradients compare chosen and alternative synonymous codon logits with a zero embedding baseline, and include encoder and decoder-prefix contributions. Attribution is not a causal biological explanation. Attention maps are retained. In silico mutagenesis intentionally omitted.

This is an initial implementation; run smoke tests and Colab validation before relying on results. Full training and GPU benchmarks are not included.


## Focused tuning

Open [Tune in Colab](https://colab.research.google.com/github/pyypyyy/aa2codon/blob/pytorch-xielu-uncertainty/pytorch_v2/Tune.ipynb) before Train. The search starts with 3 layers on each side, 8 heads, d_model=256, d_ff=1024, dropout=.05, LR=.0003 and AdamW weight decay=.01. It varies five settings (LR, model width, FFN width, dropout, weight decay) with five values each, one coordinate at a time: at most 21 unique trials rather than 3125 combinations. Each trial uses the same fixed data split, seed, batch size and eight-epoch default budget. Optional midpoint pruning discards trials with NLL more than 15% above the current incumbent at the same epoch. The objective is teacher-forced validation NLL on degenerate codons. Calibration/test partitions are not used in tuning. This is a local, short-budget search, not proof of globally optimal parameters.

The notebook persists completed trials and best_config.json to Google Drive. Interrupted trials restart from the beginning. Keep the same FASTA files (including ordering), context length, batch size and data split when moving to Train; the notebook checks the tuning signature. The default ensemble size in Train is ten. The training loop does not yet support mid-training checkpoint/resume or adaptive ensemble expansion. A larger model can exceed GPU memory; lower batch size and restart tuning in a separate directory if needed. No tuning runs or Colab execution have been performed as part of this change.


## Modern model architecture (v3)

The default model remains a three-layer encoder and three-layer autoregressive decoder with eight full attention heads and xIELU FFNs. The five-parameter search stays unchanged. New defaults are not extra search dimensions:

- Pre-RMSNorm with final encoder and decoder RMSNorm.
- Rotary positional embeddings (RoPE) inside all attention modules; cross-attention query positions are shifted by one to align each decoder prediction with its target amino acid.
- Per-head RMS normalization of Q and K.
- Query-dependent headwise sigmoid output gates (inspired by Gated Attention for Large Language Models, NeurIPS 2025, https://proceedings.neurips.cc/paper_files/paper/2025/hash/904e89bb4e632e75fb47f093b620b257-Abstract-Conference.html). Benefits for codon modelling are untested.
- Direct learned, gated projection of the aligned encoder state into each decoder input. The encoder sees the entire known protein; future codon labels are not included. This is a task-specific architectural hypothesis.
- PyTorch scaled-dot-product attention (SDPA) during normal training/inference. PyTorch selects supported kernels; FlashAttention is not guaranteed for every mask/GPU. When attention weights are requested, the explicit softmax path is retained.
- GPU mixed precision: BF16 where supported, otherwise FP16 with persistent gradient scaling. Fused AdamW on CUDA with decoupled weight decay on matrix weights only.
- Shuffled length buckets for training, cached token tensors, pinned CUDA input memory; stable order for validation and calibration. Loss/accuracy exclude deterministic and special positions under synonymous masking. Ensemble test probabilities are averaged incrementally instead of stacking all members.
- Existing uncertainty and interpretability interfaces remain intact; visualization code is unchanged. The direct alignment path and attention gates mean raw attention weights alone do not describe the entire model computation.

Tuning results from older architectures are rejected by the versioned signature. Retrain using the new tuning notebook and the same data. The test module covers causal isolation, padding and attention parity as well as uncertainty and xIELU. Tests and real Colab training still need execution. No measured improvement in predictive performance or throughput is claimed.
