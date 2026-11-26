# aa2codon PyTorch v2

Start with [Train in Colab](https://colab.research.google.com/github/pyypyyy/aa2codon/blob/pytorch-xielu-uncertainty/pytorch_v2/Train.ipynb), then [Analyze in Colab](https://colab.research.google.com/github/pyypyyy/aa2codon/blob/pytorch-xielu-uncertainty/pytorch_v2/Analyze.ipynb).

Choose a GPU runtime and specify CDS FASTA paths for one host. The notebooks store checkpoints and calibration reports in Google Drive. Local usage from this directory:

    pip install torch biopython captum plotly
    python -m unittest test_smoke
    python train.py path/to/cds.fasta --out models --ensemble 5

The model retains autoregressive encoder-decoder structure, uses xIELU (https://arxiv.org/abs/2411.13010), corrected attention dimensions, and a hard synonymous mask. ReLU and SwiGLU remain selectable; AdamW and cosine scheduling are new defaults. Legacy TensorFlow checkpoints are incompatible and must be retrained. Standard table 11 translation is used without alternative initiation-codon semantics; only standard amino acids and an optional terminal stop are supported.

Uncertainty is conditional on a common ensemble-selected prefix; it does not quantify uncertainty across all possible generated sequences. Predictive entropy is normalized by synonym count. Mean member entropy and Jensen-Shannon disagreement are proxies for aleatoric and epistemic uncertainty. Error bars are member standard deviations, not confidence intervals. Calibration uses held-out observed codons with teacher-forced prefixes, not experimental expression measurements or free-running generation. Deterministic positions are excluded from calibration metrics. Split groups exact protein sequences; near-homolog leakage remains possible unless externally computed homology cluster IDs are passed to run(). Train on a single host; multi-host conditioning is not yet implemented. Sequences exceeding max_length are excluded rather than fragmented.

Integrated gradients compare chosen and alternative synonymous codon logits with a zero embedding baseline, and include encoder and decoder-prefix contributions. Attribution is not a causal biological explanation. Attention maps are retained. In silico mutagenesis intentionally omitted.

This is an initial implementation; run smoke tests and Colab validation before relying on results. Full training and GPU benchmarks are not included.


## Focused tuning

Open [Tune in Colab](https://colab.research.google.com/github/pyypyyy/aa2codon/blob/pytorch-xielu-uncertainty/pytorch_v2/Tune.ipynb) before Train. The search starts with 3 layers on each side, 8 heads, d_model=256, d_ff=1024, dropout=.05, LR=.0003 and AdamW weight decay=.01. It varies five settings (LR, model width, FFN width, dropout, weight decay) with five values each, one coordinate at a time: at most 21 unique trials rather than 3125 combinations. Each trial uses the same fixed data split, seed, batch size and eight-epoch default budget. Optional midpoint pruning discards trials with NLL more than 15% above the current incumbent at the same epoch. The objective is teacher-forced validation NLL on degenerate codons. Calibration/test partitions are not used in tuning. This is a local, short-budget search, not proof of globally optimal parameters.

The notebook persists completed trials and best_config.json to Google Drive. Interrupted trials restart from the beginning. Keep the same FASTA files (including ordering), context length, batch size and data split when moving to Train; the notebook checks the tuning signature. The default ensemble size in Train is ten. The training loop does not yet support mid-training checkpoint/resume or adaptive ensemble expansion. A larger model can exceed GPU memory; lower batch size and restart tuning in a separate directory if needed. No tuning runs or Colab execution have been performed as part of this change.

