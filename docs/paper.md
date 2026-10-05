---
title: "The 2026 Fully Open LLM Training Guide: A Technical Blueprint for OpenEuroLLM-Scale Foundation Models"
author: "Birger Moëll"
date: "2026-10-05"
bibliography: "references.bib"
---

# Executive Summary

This field guide uses the public OpenEuroLLM artifacts checked on 5 October
2026 as examples for building and evaluating a fully open multilingual model.
It follows the project's stated commitment to open documentation, code,
evaluation, intermediate results and community involvement
[@openeurollm_official_2026; @ai_sweden_openeurollm_2026]. Released evidence,
active experiments and formal deliverables are kept distinct.

A model release should be an auditable system, not weights alone. The guide
covers data lineage, tokenizer design, architecture and scaling decisions,
distributed training, post-training, multilingual and long-context evaluation,
safety, and release engineering. For each consequential claim, it asks for the
relevant artifact: a manifest, configuration, checkpoint, log, raw output or
independent test. Section 20 closes the field guide with a research brief on
four active post-training streams and the evidence needed to judge them.

# 1. Scope And Operating Principles

## 1.1 What "Fully Open" Must Mean In 2026

A fully open LLM project should release more than model weights. Open weights
are useful, but they do not let the research community understand why a model
behaves as it does, whether a capability claim is reliable, whether a language
was genuinely supported, or whether a failure comes from data, architecture,
training instability, tokenizer fertility, evaluation leakage, or post-training
over-optimization.

The minimum release package should contain:

- Model weights for base, instruction, reasoning, and safety variants.
- Tokenizer files, tokenizer training data description, normalization rules, and
  fertility analyses.
- Architecture code and exact configuration files.
- Training code, optimizer settings, parallelism settings, launcher scripts, and
  restart procedures.
- Data recipe, source registry, data manifests, filtering code, deduplication
  code, and mixture weights.
- Clear legal and licensing metadata for each data source class.
- Data-quality reports and ablation evidence.
- Intermediate checkpoints and training logs at meaningful intervals.
- Evaluation harnesses, prompts, task definitions, parse logic, raw outputs, and
  result summaries.
- Safety and privacy reports, including known limitations.
- Model cards and dataset cards that separate verified claims from hypotheses.
- Energy, compute, and hardware descriptions sufficient for scientific audit.

OLMo and Dolma are important precedents because they explicitly frame openness
around data, code, intermediate artifacts, and logs, not only weights
[@groeneveld2024olmo; @soldaini2024dolma]. FineWeb is important because it
demonstrates that open data curation needs ablations and documented recipes, not
only a large token count [@penedo2024fineweb]. OpenEuroLLM should adopt the same
spirit, with an even stronger multilingual and EU-compliance emphasis.

## 1.2 Project Objectives

The project objective is not merely to produce a leaderboard model. A strong
OpenEuroLLM-style project should optimize for five objectives at once:

1. Scientific quality: stable training, ablations, calibrated evaluation, and
   reproducible artifacts.
2. Multilingual usefulness: reliable coverage across EU official languages and
   other socially and economically relevant languages.
3. Openness: auditable training data recipes, code, logs, and model behavior.
4. Compliance and safety: EU-facing legal, privacy, copyright, and deployment
   risk discipline.
5. Operational usefulness: models that can be fine-tuned, served, quantized,
   evaluated, and integrated into real applications.

These goals interact. A model can be highly capable but not open. It can be open
but too weak to matter. It can be multilingual in marketing copy while actually
being evaluated mostly in English. It can be safe in English but brittle in
lower-resource languages. It can score well on static benchmarks while failing
as a tool-using agent. The engineering task is therefore a systems problem.

## 1.3 OpenEuroLLM Artifact State, 5 October 2026

The public record now contains several distinct artifact classes. Do not infer
that every organization upload is a final project release. The official
[deliverables page](https://openeurollm.eu/deliverables) still lists the
initial dataset and first-model deliverables for December 2026.

| Public artifact | What it establishes | What it does not establish |
|---|---|---|
| [Training Data Catalogue](https://github.com/OpenEuroLLM/training-data-catalogue) and [Training Data Collection](https://github.com/OpenEuroLLM/training-data-collection) | Discoverable sources, cycle-specific `metadata.yaml`, counts, sampling and packing recipes | That every source can be redistributed under one license |
| [Tokenizer v2, 256k](https://huggingface.co/openeurollm/tokenizer-256k-v2) and [128k](https://huggingface.co/openeurollm/tokenizer-128k-v2) | Released SentencePiece tokenizer variants with multilingual and special-token tests | Interchangeability with v1 checkpoints; the exact tokenizer revision remains part of model identity |
| [Prelude](https://huggingface.co/openeurollm/prelude), [9B SFT experiment](https://huggingface.co/openeurollm/oellm-9b-256k-sft), and [32B repository](https://huggingface.co/openeurollm/oellm-32b) | Public model artifacts and experiment reports | A completed, uniformly documented final model family; the 32B repository currently has no model card |
| [OpenEuroLLM datasets](https://huggingface.co/openeurollm/datasets) | Published translated instruction, reasoning, tool-use, RLVR, and evaluation resources | Automatic evidence that each dataset was used in a given checkpoint |
| [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) and [post-training](https://github.com/OpenEuroLLM/post-training) | Public evaluation and configuration-driven SFT/DPO tooling | Comparable results without pinned task, prompt, shot count, metric and code revision |

Use a dated artifact inventory for release claims. Record the exact repository
revision, model commit, tokenizer revision, data recipe, and evaluation
configuration. A nominal context limit is configuration metadata until a
specific retrieval or reasoning test demonstrates useful behavior at that
length. The [9B SFT model card](https://huggingface.co/openeurollm/oellm-9b-256k-sft)
shows this distinction clearly: it reports 4K-sequence SFT and separate
long-context retrieval probes, with their limitations.

# 2. End-To-End Lifecycle

The lifecycle of a fully open LLM can be decomposed into stages:

1. Charter, governance, and success criteria.
2. Data source registry and legal review.
3. Data extraction, normalization, and language identification.
4. Deduplication, privacy filtering, quality filtering, contamination control.
5. Dataset composition and multilingual mixture design.
6. Tokenizer design and validation.
7. Architecture selection and scaling-law experiments.
8. Infrastructure and distributed training stack.
9. Pretraining.
10. Mid-training, data curriculum, and long-context adaptation.
11. Supervised instruction tuning.
12. Preference tuning and RLHF/DPO.
13. Reasoning training with verifiable rewards, GRPO, RLVR, and distillation.
14. Agent capability training.
15. Safety tuning and refusal/calibration behavior.
16. Evaluation, red teaming, release, and monitoring.

Every stage must produce artifacts. If a stage does not produce artifacts, it is
not auditable. The single most important management principle is therefore:
**make the pipeline observable**. Record what data entered, what transformations
were applied, what parameters were used, what was removed, what changed in
validation loss, and why a decision was made.

# 3. Data Governance, Legal Review, And Source Registry

## 3.1 Source Registry

The source registry is the root of reproducibility. It should be a versioned
table with one row per source collection:

- Source name and URL or storage path.
- Snapshot date and acquisition method.
- License class and terms.
- Jurisdictional considerations.
- Language/domain claims.
- Estimated documents, bytes, and tokens.
- Known risks: PII, copyright, toxicity, spam, templated content, malware,
  code-license issues, sensitive personal data, medical/legal content.
- Whether source text can be redistributed, redistributed as hashes/manifests,
  or only described as a recipe.
- Contact or owner.

For code, record license compatibility and provenance. The Stack and StarCoder
work illustrate why permissive-code filtering matters for open model training
[@kocetkov2022stack; @bigcode2023starcoder]. For web corpora, keep exact Common
Crawl snapshots, extraction versions, filtering versions, and source-domain
statistics. FineWeb's main contribution is not merely token count; it is a
systematic curation recipe with ablations [@penedo2024fineweb].

## 3.2 Legal And Privacy Review

Legal review should happen before expensive processing. The registry should
classify sources into at least:

- Public-domain or permissive.
- Openly licensed with attribution or share-alike conditions.
- Web-crawled public text with uncertain copyright status.
- Contracted or partner-provided data with restrictions.
- Sensitive or high-risk data requiring exclusion or special handling.

Privacy filtering must include deterministic detectors, probabilistic models,
and manual audit. At minimum:

- Email, phone, address, government ID, credit card, API keys, passwords.
- Health, legal, financial, and child-related sensitive content.
- Personal social media and forum content where identifiability is high.
- Memorization-risk analysis for rare sequences and unique documents.

The goal is not to pretend that automatic privacy filtering is perfect. The
goal is to document residual risk, make conservative source choices, and
provide mechanisms for removal requests and model updates.

# 4. Data Curation Pipeline

## 4.1 Extraction And Normalization

Raw data is not text. Web pages contain boilerplate, scripts, navigation,
comments, tables, malformed markup, language mixing, and hidden spam. PDF and
OCR content contain line breaks, hyphenation, repeated headers, and encoding
errors. Code repositories contain generated files and vendored dependencies.

Recommended pipeline:

1. Extract raw text with source-specific parsers.
2. Preserve document boundaries and source metadata.
3. Normalize Unicode carefully; do not erase meaningful script distinctions.
4. Remove boilerplate, navigation, cookie notices, and repeated template text.
5. Segment documents and paragraphs.
6. Run language identification at document and segment levels.
7. Keep uncertainty scores, not only labels.
8. Store immutable raw, normalized, and filtered versions separately.

Do not use ad hoc string manipulation for major source processing. Use
structured parsers where possible and keep parser versions fixed. Always sample
failures visually: a 0.1% extraction bug at trillion-token scale is a very
large dataset.

## 4.2 Language Identification

Multilingual LLMs fail quietly when language metadata is wrong. Language ID
should be performed at multiple granularities:

- Document-level primary language.
- Segment-level language labels.
- Script detection.
- Confidence and entropy.
- Mixed-language markers.

For European models, distinguish languages that share scripts and vocabulary.
Do not collapse regional variants too early. Use language tags such as
`eng_Latn`, `deu_Latn`, or other BCP-47/Glottolog-compatible formats if the
pipeline supports them.

Important metrics:

```text
language_confidence = max_l p(l | document)
language_entropy = - sum_l p(l | document) log p(l | document)
mixed_document = language_entropy > tau_entropy or second_language_share > tau_share
```

Language uncertainty should affect mixture sampling and evaluation confidence.
It should not simply be hidden.

## 4.3 Deduplication

Deduplication reduces memorization, improves data efficiency, and prevents
benchmark leakage. However, excessive deduplication can harm legitimate repeated
formats and minority-language data.

Use multiple deduplication levels:

- Exact document hashing.
- Near-deduplication using MinHash or SimHash.
- Line-level or paragraph-level deduplication for boilerplate.
- Cross-split deduplication between train, validation, and evaluation.
- Benchmark contamination checks.

Let \(D\) be a set of documents, \(h(d)\) a locality-sensitive hash signature,
and \(J(d_i,d_j)\) the Jaccard similarity of shingles. Near duplicates can be
defined as:

\[
\operatorname{dup}(d_i,d_j)=\mathbb{1}[J(d_i,d_j) > \tau].
\]

The threshold \(\tau\) is not universal. Web news may need a different
threshold than code, legislation, or educational material. For multilingual
data, deduplication should not accidentally remove translations unless that is
explicitly intended.

## 4.4 Quality Scoring

Quality scoring should combine interpretable filters and learned signals:

- Length and token ratio filters.
- Character distribution, punctuation, and script sanity.
- Repetition and boilerplate metrics.
- Language-ID confidence.
- Perplexity or surprisal from reference models.
- Classifiers for educational value, code quality, toxicity, spam, and
  domain-specific usefulness.
- LLM-as-judge sampling only for audit, not as the sole filter.

FineWeb-Edu showed that educational filtering can improve knowledge and
reasoning benchmarks [@penedo2024fineweb]. But classifiers transfer poorly
across languages and domains. For OpenEuroLLM, educational filtering must be
validated per language. A classifier trained mostly on English educational text
may under-score high-quality Finnish, Bulgarian, Maltese, or legal-domain text.

Recommended score:

\[
q(d)=w_\ell q_\ell(d)+w_r q_r(d)+w_e q_e(d)+w_s q_s(d)+w_p q_p(d)-w_b b(d),
\]

where \(q_\ell\) is language confidence, \(q_r\) readability/extraction quality,
\(q_e\) educational/domain value, \(q_s\) safety, \(q_p\) privacy confidence,
and \(b\) is boilerplate/spam risk. The weights should be learned or tuned
through ablations, not guessed once.

## 4.5 Contamination Control

Evaluation contamination is especially severe for open models because benchmark
data often appears in web crawls, GitHub repositories, educational sites, and
model-generated examples. The pipeline should maintain:

- Hashes and fuzzy hashes of benchmark prompts and answers.
- N-gram overlap checks.
- Semantic retrieval checks for paraphrased contamination.
- Split locks: validation and eval sets must be frozen before major training
  runs.
- Public reporting of contamination methodology.

Do not claim benchmark superiority without contamination analysis. For
reasoning models, also check solution contamination, not only problem text.

# 5. Dataset Composition And Multilingual Mixture Design

## 5.1 Mixture As An Optimization Problem

Dataset mixture is the core of model behavior. A training mixture is not just a
list of sources; it is a schedule:

\[
P_t(s,l,d) = P_t(s)P_t(l|s)P_t(d|s,l),
\]

where \(s\) is source, \(l\) is language, \(d\) is domain, and \(t\) is training
time or curriculum phase.

The mixture should optimize:

- Validation loss across domains and languages.
- Downstream task performance.
- Robustness and safety.
- Coverage of target languages.
- Data freshness and legal reliability.
- Tokenizer efficiency and fertility.
- Avoidance of overfitting repeated high-quality subsets.

Sampling probability can be temperature-smoothed:

\[
p_i = \frac{n_i^\alpha}{\sum_j n_j^\alpha},
\]

where \(n_i\) is available token count for bucket \(i\), and \(\alpha \in [0,1]\)
controls upsampling. \(\alpha=1\) samples proportional to size; \(\alpha=0\)
samples uniformly across buckets. Low-resource languages usually need
temperature smoothing, but aggressive upsampling risks memorization and
overfitting.

## 5.2 Multilingual Fairness

Language coverage has at least six dimensions:

1. Pretraining token volume.
2. Tokenizer fertility.
3. Domain diversity.
4. Instruction data.
5. Preference/reasoning data.
6. Evaluation quality.

A language with many pretraining tokens but no post-training data may be
fluent but bad at following instructions. A language with synthetic evaluation
only may look covered but remain unvalidated. A language with high tokenizer
fertility consumes more context window per word, reducing effective context.

Let \(F_l\) be tokenizer fertility for language \(l\):

\[
F_l = \frac{\text{tokens}_l}{\text{normalized words or characters}_l}.
\]

Higher \(F_l\) means less text capacity per fixed token window. Long-context
claims should therefore report results by language, token length, and
approximate text length.

## 5.3 OpenEuroLLM Data Workstream

The public [Training Data Collection](https://github.com/OpenEuroLLM/training-data-collection)
separates a searchable source catalogue from the exact, cycle-specific
collection used to train a model. Its `metadata.yaml` files and tokenizer-based
source counts are the natural anchors for a reproducible mixture. For each
cycle, publish the source revision, subset rule, quality and privacy flags,
language/domain counts, tokenizer revision, packed-token counts, random seed,
and final manifest. Where raw source redistribution is restricted, publish the
complete acquisition and transformation recipe within those terms.

Treat sampling as a measured intervention. Report both document shares and
token shares: long examples can dominate tokens even when they are a minority
of rows. The [9B SFT card](https://huggingface.co/openeurollm/oellm-9b-256k-sft)
provides a concrete example: its Nemotron math slice was 18.77% of loaded rows
but 64.41% of the tokenized pool. Record prefix selection, packing, and
post-concatenation deduplication explicitly; these details change the effective
training distribution.

# 6. Tokenizer Design

## 6.1 Goals

The tokenizer must support target languages, code, math, structured data, and
tool-use formats without pathological fragmentation. In 2026, common choices
include byte-level BPE, unigram, BBPE variants, or tokenizer families inherited
from strong open models. Qwen3 reports broad multilingual support through a
shared tokenizer, which reflects the practical value of a single tokenizer for
model families [@qwen3_2025].

Tokenizer requirements:

- Stable normalization.
- Byte fallback.
- Good fertility across all target languages.
- Robust handling of code and mathematical notation.
- Reserved tokens for chat, tools, system messages, reasoning controls, and
  safety metadata.
- No accidental collision between special tokens and ordinary text.
- Versioned training corpus and training script.

## 6.2 Fertility And Coverage Tests

For each language and domain, compute:

- Mean tokens per character.
- Mean tokens per word.
- Percent of single-character tokens.
- Unknown/byte fallback rate.
- Compression ratio versus UTF-8 bytes.
- Tokenization of named entities, inflections, compounds, and diacritics.
- Tokenization of code identifiers and common APIs.

Example:

```text
Language: Finnish
Domain: legal text
Characters: 10,000,000
Words: 1,420,000
Tokens: 2,100,000
Tokens/word: 1.48
Bytes/token: 4.8
Fallback rate: 0.03%
```

The tokenizer should be evaluated before pretraining. Retrofitting a tokenizer
after large-scale training is extremely expensive.

OpenEuroLLM's [256k v2 card](https://huggingface.co/openeurollm/tokenizer-256k-v2)
documents a retrain that restored Georgian coverage, adds chat and reasoning
special tokens, and reports held-out results across prose, code, math, chat and
PDF text. Its published comparison also shows a code trade-off. This is the
right reporting pattern: publish per-language deltas and domain weaknesses,
then verify that the chosen tokenizer and special-token IDs match every model,
packed dataset and chat template in the run.

# 7. Architecture Strategy For 2026

## 7.1 Baseline: Dense Decoder-Only Transformer

The default baseline remains a decoder-only Transformer [@vaswani2017attention].
It is simple, robust, well-supported by serving stacks, and easier to debug than
MoE or experimental sparse attention. A strong dense baseline is essential even
if the frontier architecture becomes MoE.

Recommended dense choices:

- Pre-normalization with RMSNorm.
- SwiGLU/GeGLU-style feed-forward networks.
- RoPE or improved positional encodings.
- Grouped-query attention or multi-query attention for inference efficiency.
- QK-norm or other stabilization if validated.
- FlashAttention-compatible attention kernels [@dao2022flashattention;
  @flashattention2_2023].
- BF16 or FP8 training only after numerical validation.

Dense models are ideal for:

- Early scaling-law experiments.
- Reference checkpoints.
- Debugging data mixtures.
- Distillation teachers or students.
- Smaller public releases.

## 7.2 Mixture Of Experts

MoE is the most important architectural lever for increasing total parameter
count without proportional inference cost. Switch Transformers showed the
scaling promise of sparse expert routing [@fedus2022switch]. Mixtral
demonstrated a practical open-weight sparse MoE model with strong cost/performance
trade-offs [@mixtral2024]. DeepSeek-V3 pushed MoE further with 671B total
parameters and 37B activated per token, using DeepSeekMoE, auxiliary-loss-free
load balancing, MLA, FP8 training, and co-designed infrastructure
[@deepseekv3_2025]. Qwen3 also includes dense and MoE variants
[@qwen3_2025].

MoE benefits:

- More total capacity for a fixed active compute budget.
- Better specialization across domains/languages if routing is stable.
- Potentially strong inference cost/performance.
- Natural family scaling: dense small models, MoE frontier models.

MoE risks:

- Routing instability.
- Expert collapse or load imbalance.
- More complex distributed training.
- Harder checkpoint portability.
- Harder inference serving and quantization.
- Multilingual routing pathologies where low-resource languages route to
  under-trained experts.

MoE router equations:

\[
r(x)=\operatorname{softmax}(W_r h_x),
\]

\[
E(x)=\operatorname{TopK}(r(x), k),
\]

\[
y = \sum_{e \in E(x)} r_e(x) f_e(h_x).
\]

Load balancing should be measured per batch, per language, per domain, and over
time. A multilingual MoE must report whether experts specialize by language,
domain, syntax, or artifact. Specialization is useful only if it does not starve
low-resource languages.

2026 recommendation: train a dense baseline and a MoE candidate. Use MoE for
frontier scaling only after the dense pipeline is stable. Do not let MoE become
the first system you debug.

## 7.3 Multi-Head Latent Attention And KV Efficiency

Inference cost is dominated by KV cache for long contexts. Multi-head latent
attention, used in DeepSeek-V2/V3, compresses key-value representations and
reduces memory pressure [@deepseekv3_2025]. Grouped-query attention and
multi-query attention are simpler alternatives. The right choice depends on
serving constraints, training code support, and quality.

Decision rule:

- If the model must serve long-context chat at scale, optimize KV cache early.
- If the model is a research baseline, prefer simpler attention.
- If adopting MLA, validate against full attention at small scale and test
  export/inference support before large-scale training.

## 7.4 Sparse Attention

Sparse attention is attractive because full self-attention has quadratic
prefill cost:

\[
\operatorname{cost}_{attn}=O(n^2 d),
\]

where \(n\) is sequence length and \(d\) is hidden size. Sparse attention can
reduce this toward \(O(n k d)\), where \(k \ll n\) is the attended subset.

2025-2026 work includes native trainable sparse attention, efficient
long-sequence serving, dynamic sparse attention, and adaptive pruning
[@native_sparse_attention_2025; @lserve2025; @twilight2025]. These methods are
promising but operationally risky.

Sparse attention trade-offs:

- Fixed sparse masks are simple but may miss task-relevant tokens.
- Dynamic masks adapt better but are harder to train and serve.
- Inference-only sparse attention can degrade quality if the model was trained
  with full attention.
- Trainable sparse attention is cleaner but requires deep kernel and framework
  support.

2026 recommendation: keep full attention as the gold baseline. Use sparse
attention for long-context frontier experiments only after measuring retrieval,
lost-in-the-middle, multilingual behavior, short-context regression, and serving
compatibility. Do not claim long-context capability from nominal context length.

## 7.5 State-Space And Hybrid Models

Mamba-like state-space models and hybrid attention/state-space architectures
remain relevant for efficient long-context modeling, but for a fully open
general-purpose multilingual LLM in 2026, the safest baseline is still
Transformer-first because tooling, transfer recipes, evaluation, quantization,
and serving are more mature. Hybrid models are worth a research track, not the
first production-scale OpenEuroLLM backbone unless the team has strong internal
expertise.

## 7.6 Recommended Architecture Portfolio

Train a family:

- 0.1B-0.5B: data and tokenizer smoke tests.
- 1B-3B dense: scaling-law and mixture ablations.
- 7B-9B dense: high-quality open baseline and teacher/student anchor.
- 20B-40B dense or efficient dense: serious general model if compute allows.
- 50B-200B+ MoE total parameters: frontier candidate with lower active
  parameters.

Each family member should share tokenizer, data recipe lineage, evaluation
harness, and model-card structure.

# 8. Scaling Laws And Experiment Design

## 8.1 Compute-Optimal Training

Scaling laws guide trade-offs between parameter count \(N\), token count \(D\),
and compute \(C\). A simplified training compute estimate for decoder-only
Transformers is:

\[
C \approx 6ND,
\]

where \(C\) is FLOPs, \(N\) is non-embedding parameters, and \(D\) is training
tokens. Chinchilla-style results showed that many earlier LLMs were undertrained
relative to parameter count and that compute-optimal models use more tokens per
parameter than older recipes [@hoffmann2022chinchilla]. Data-constrained scaling
adds another wrinkle: if high-quality unique data is limited, repeated data and
curriculum design matter [@muennighoff2024scaling].

For a 9B model trained on 10T tokens:

\[
C \approx 6 \times 9 \times 10^9 \times 10 \times 10^{12}
  = 5.4 \times 10^{23} \text{ FLOPs}.
\]

At sustained 300 TFLOP/s per GPU:

\[
\text{GPU seconds} =
\frac{5.4 \times 10^{23}}{3 \times 10^{14}}
= 1.8 \times 10^9,
\]

\[
\text{GPU hours} \approx 500{,}000.
\]

This simplified estimate excludes overhead, failed runs, evaluation, tuning,
post-training, and lower utilization. Real allocations must include overhead
and reruns.

## 8.2 Scaling-Law Protocol

For every candidate mixture:

1. Train small models at multiple sizes.
2. Keep tokenizer and architecture fixed.
3. Train each to enough tokens to compare loss curves.
4. Evaluate on frozen validation sets by language and domain.
5. Fit loss curves:

\[
L(N,D)=L_\infty + aN^{-\alpha}+bD^{-\beta}.
\]

6. Compare predicted frontier performance.
7. Validate predictions with one larger run.

Do not use a single validation loss. Use a dashboard:

- General web.
- Educational.
- Code.
- Math.
- Legal/public sector.
- News.
- Low-resource languages.
- High-resource European languages.
- Instruction-like text.
- Long documents.

Scaling-law results are useful only if validation is locked, representative,
and reproducible. Publish the validation manifest and the exact
[oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) task configuration.

# 9. Infrastructure And Distributed Training

## 9.1 HPC Target Reality

OpenEuroLLM-like training happens across heterogeneous EuroHPC systems.
Public [training-data catalogue](https://github.com/OpenEuroLLM/training-data-catalogue)
documentation and [post-training tooling](https://github.com/OpenEuroLLM/post-training)
describe workflows spanning LUMI, Leonardo and MareNostrum 5. Hardware
differences matter:

- GPU memory: 64GB vs 80GB vs 96GB changes feasible model parallelism.
- Interconnect: affects tensor, pipeline, expert, and data parallelism.
- Filesystem performance: affects dataloader throughput and checkpointing.
- Queue policy: affects run length, failure recovery, and experiment cadence.
- Software stack: ROCm vs CUDA, compiler versions, kernel availability.

The goal is not to pretend all systems are identical. The goal is to create a
portable experiment specification that compiles into system-specific launchers.

## 9.2 Parallelism

Large-scale training composes:

- Data parallelism (DP): replicate model, split batch.
- Tensor parallelism (TP): split matrix operations.
- Pipeline parallelism (PP): split layers across devices.
- Sequence/context parallelism (SP/CP): split sequence dimension.
- Expert parallelism (EP): distribute MoE experts.
- Fully sharded data parallelism (FSDP/ZeRO): shard parameters, gradients, and
  optimizer states.

Total world size:

\[
W = DP \times TP \times PP \times CP \times EP.
\]

The right decomposition depends on model size, sequence length, GPU memory,
network, and framework support.

Dense 7B on 8x80GB GPUs may fit with TP=1, PP=1, DP=8 using activation
checkpointing. A 70B dense model may need TP, PP, and FSDP. A large MoE model
may need EP and careful routing all-to-all optimization.

## 9.3 Checkpointing And Fault Tolerance

At frontier scale, failure is normal. Checkpointing must be designed before the
run:

- Save model, optimizer, scheduler, RNG, dataloader state, and consumed-token
  count.
- Validate restart determinism at small scale.
- Keep rolling checkpoints and milestone checkpoints.
- Store checksums and metadata.
- Monitor checkpoint time and filesystem load.
- Test partial-node failure recovery if framework supports it.

Never start a major run before a restart test has succeeded. Never assume a
checkpoint is valid until it has been loaded and used to continue training.

## 9.4 Observability

Track:

- Training loss and validation loss.
- Gradient norm.
- Learning rate.
- Tokens/sec and samples/sec.
- Model FLOP utilization (MFU).
- GPU memory and utilization.
- Dataloader time.
- Communication time.
- Straggler nodes.
- NaNs/infs.
- Expert load balance.
- Activation checkpoint overhead.
- Evaluation job latency.
- Checkpoint save/load time.

Stable training is a product feature. DeepSeek-V3 explicitly emphasizes stable
training without irrecoverable loss spikes or rollbacks [@deepseekv3_2025].
Open projects should report stability, not only final benchmarks.

# 10. Pretraining

## 10.1 Objective

The base objective remains autoregressive next-token prediction:

\[
\mathcal{L}_{LM}(\theta)=
-\sum_{t=1}^{T}\log p_\theta(x_t|x_{<t}).
\]

Use packed sequences to reduce padding waste. But packing creates document
boundary issues. For long-context training, efficient inter-document masking may
matter because compute depends on actual document length statistics, not merely
maximum context length. If documents are packed without
masking, the model may learn cross-document artifacts. If masking is naive, it
may waste attention compute. If masking is efficient and framework-supported,
it can improve both quality and cost.

## 10.2 Hyperparameters

Key hyperparameters:

- Global batch size in tokens.
- Sequence length.
- Learning rate peak.
- Warmup tokens.
- Decay schedule.
- Weight decay.
- AdamW betas and epsilon.
- Gradient clipping.
- Dropout, often zero for large-scale pretraining.
- Initialization scale.
- Loss scaling for mixed precision.

Common schedule:

\[
\eta(t)=
\begin{cases}
\eta_{max}\frac{t}{T_w}, & t<T_w\\
\eta_{min} + \frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi(t-T_w)}{T-T_w}\right), & t\ge T_w
\end{cases}
\]

Use smaller experiments to tune learning rate and batch size. Do not copy
hyperparameters blindly from a different architecture or tokenizer.

## 10.3 Data Curriculum

Modern strong models often use staged data curricula: broad general pretraining,
then higher-quality or domain-targeted phases, then long-context or reasoning
heavy phases. OLMo 2 reports a two-stage pretraining/mid-training regime and a
late-stage data curriculum [@olmo2furious2025]. Qwen3 reports staged
pretraining with general, reasoning, and long-context phases [@qwen3_2025].

Recommended phases:

1. General multilingual web/books/reference/code foundation.
2. Higher-quality educational, code, math, public-sector, and multilingual
   curated data.
3. Long-document and long-context adaptation.
4. Optional domain-balanced cooldown to reduce over-specialization.

The curriculum should be reflected in data manifests and training logs:

```yaml
phase: midtrain_quality_v2
start_token: 7_500_000_000_000
end_token: 9_000_000_000_000
mixture:
  hplt4_clean: 0.35
  fineweb_edu_like: 0.20
  code_permissive: 0.15
  math_science: 0.10
  eu_public_sector: 0.10
  multilingual_low_resource_upsample: 0.10
```

## 10.4 Validation

Evaluate during pretraining:

- Perplexity/loss by language and domain.
- Downstream zero-shot and few-shot tasks at sparse checkpoints.
- Code benchmarks.
- Math benchmarks.
- Long-context probes.
- Toxicity and refusal calibration probes.
- Memorization probes.
- Regression against previous checkpoint.

Never wait until the end to evaluate. Late discovery of data mixture failure is
expensive.

# 11. Mid-Training And Long-Context Adaptation

## 11.1 Why Mid-Training Exists

Mid-training bridges raw next-token pretraining and instruction following. It
can improve reasoning, code, multilingual balance, and long-context behavior
without the brittleness of post-training alone.

Mid-training data can include:

- High-quality educational text.
- Textbook-like explanations.
- Code and documentation.
- Math derivations.
- Synthetic but verified reasoning traces.
- Long documents.
- Multilingual parallel and comparable corpora.
- Public-sector, legal, and administrative text.

Mid-training should still use next-token prediction unless using a specific
auxiliary objective. It is not yet chat SFT.

## 11.2 Long Context

Long context is a system capability, not a RoPE number. Effective context length
requires:

- Positional encoding support.
- Training or adaptation on long sequences.
- Attention/kernel support.
- Serving support and KV-cache capacity.
- Evaluation on retrieval, aggregation, reasoning, and lost-in-the-middle.
- Multilingual validation.

YaRN and LongRoPE are practical context-extension techniques
[@peng2023yarn; @longrope2024]. The local OpenEuroLLM wiki recommends treating
long context as a system-level capability and separating cheap local screening
from final GPU/HPC validation.

Recommended path:

1. Baseline at original context.
2. YaRN to 32K as cheap extension baseline.
3. LongRoPE/LongRoPE2-style candidate if search and conversion overhead are
   manageable.
4. Evaluate short-context regression.
5. Evaluate long-context retrieval and reasoning by language.
6. Validate serving stack.

## 11.3 Inter-Document Masking

For packed long-context training, document boundaries matter. Suppose a packed
sequence contains documents \(d_1,\ldots,d_m\). Standard causal attention allows
tokens in \(d_i\) to attend to previous tokens from \(d_{i-1}\), which may be
semantically unrelated. Inter-document masking blocks attention across document
boundaries:

\[
A_{ij}=1 \quad \text{iff} \quad j \le i \text{ and } doc(i)=doc(j).
\]

Efficient implementation is non-trivial. If efficient inter-document masking
is supported, compute can depend on document length statistics such as mean
\(\mu\) and variance \(\sigma^2\), not only maximum context. Benchmark this
with the actual attention kernel, packing distribution, and hardware.

# 12. Post-Training Overview

Post-training converts a base model into useful assistants, reasoners, coders,
and agents. It also risks damaging the base model if done carelessly. A good
post-training stack has stages:

1. Instruction supervised fine-tuning (SFT).
2. Preference optimization: RLHF, DPO, IPO/KTO-style variants.
3. Reasoning training with verifiable rewards.
4. Tool-use and agent training.
5. Safety tuning and policy calibration.
6. Distillation into smaller models.
7. Multilingual repair and regression recovery.

Llama 3, Tulu 3, DeepSeek-V3, DeepSeek-R1, and Qwen3 all show that
post-training is not a minor appendix; it is central to final model behavior
[@llama3herd2024; @lambert2024tulu3; @deepseekv3_2025; @deepseekr1_2025;
@qwen3_2025].

# 13. Supervised Instruction Tuning

## 13.1 Data

Instruction SFT data should include:

- General chat.
- Multilingual instruction following.
- Coding.
- Math.
- Summarization.
- Information extraction.
- Structured output.
- Long-context tasks.
- Tool-call demonstrations.
- Refusal and safe-completion examples.
- Domain-specific tasks for public services and industry.

Every example should have metadata:

```json
{
  "language": "deu_Latn",
  "domain": "public_service",
  "task": "form_explanation",
  "source": "human_written",
  "license": "cc-by-4.0",
  "safety_class": "low",
  "requires_tool": false,
  "quality_score": 0.94
}
```

## 13.2 Objective

For chat SFT:

\[
\mathcal{L}_{SFT}(\theta)=
-\sum_{(x,y)\in \mathcal{D}}\sum_t
\log p_\theta(y_t|x,y_{<t}).
\]

Mask user tokens and train on assistant tokens unless explicitly training
transcription or multi-role modeling. Preserve system-message formatting and
tool-call schema exactly.

## 13.3 Trade-Offs

Too much SFT can:

- Reduce creativity.
- Overfit chat style.
- Harm multilingual fluency.
- Damage code or math performance.
- Increase verbosity.
- Teach brittle refusal patterns.

Use small learning rates, monitor base benchmarks, and keep multilingual
regression tests.

# 14. Preference Training: RLHF, DPO, And Friends

## 14.1 RLHF

RLHF, popularized for instruction-following models by InstructGPT, uses human
preferences to train a reward model and then optimizes the policy with PPO
[@ouyang2022instructgpt; @schulman2017ppo].

Reward model:

\[
\mathcal{L}_{RM} =
-\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l)),
\]

where \(y_w\) is preferred and \(y_l\) is rejected.

Policy objective with KL control:

\[
\max_\theta
\mathbb{E}_{y\sim \pi_\theta}
\left[
r_\phi(x,y) -
\beta D_{KL}(\pi_\theta(\cdot|x)\|\pi_{ref}(\cdot|x))
\right].
\]

RLHF strengths:

- Flexible preference modeling.
- Can optimize nuanced human judgments.
- Useful for style, helpfulness, harmlessness, and interaction quality.

RLHF risks:

- Reward hacking.
- Expensive human labeling.
- Reward model bias.
- Instability.
- Multilingual preference gaps.
- Over-optimization and loss of diversity.

## 14.2 DPO

Direct Preference Optimization removes the explicit reward model and optimizes
preferences directly [@rafailov2023dpo]:

\[
\mathcal{L}_{DPO}(\pi_\theta;\pi_{ref}) =
-\mathbb{E}_{(x,y_w,y_l)}
\log \sigma\left(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right).
\]

DPO strengths:

- Simpler than PPO.
- Stable.
- Good for open post-training pipelines.
- Easy to reproduce.

DPO risks:

- Depends heavily on preference data quality.
- Less flexible for multi-step environment rewards.
- Can overfit preference style.

2026 recommendation: use SFT + DPO as the default reliable open pipeline. Use
RLHF/PPO or GRPO/RLVR when the reward is verifiable, environment-based, or when
the project has the infrastructure to monitor policy optimization carefully.

# 15. Reasoning Training

## 15.1 Why Reasoning Needs Its Own Stage

Reasoning training is not just "more chat data". It needs tasks with delayed
credit assignment, verifiable answers, chain-of-thought or latent reasoning
behavior, and careful evaluation. DeepSeekMath introduced GRPO for mathematical
reasoning in open models [@deepseekmath2024]. DeepSeek-R1 demonstrated the
importance of reinforcement learning for reasoning behavior, including
long-form thinking and self-correction [@deepseekr1_2025]. Qwen3's
thinking/non-thinking modes indicate a 2026 trend: models should know when to
reason deeply and when to answer efficiently [@qwen3_2025].

## 15.2 Reasoning Data Types

- Math problems with exact answers.
- Code problems with unit tests.
- Logic puzzles.
- Scientific QA with derivable answers.
- Formal proof steps.
- Data-analysis tasks with executable checks.
- Multi-hop retrieval with cited evidence.
- Tool-augmented tasks with environment verification.

Each example should define whether the reward is:

- Exact match.
- Unit test pass.
- Symbolic equivalence.
- Verifier-model judgment.
- Human preference.
- Hybrid.

Prefer verifiable rewards where possible.

## 15.3 GRPO

Group Relative Policy Optimization avoids a separate value model by sampling a
group of completions for each prompt and normalizing rewards within the group.
For prompt \(x\), sample completions \(y_1,\ldots,y_G\), compute rewards
\(r_i\), then:

\[
\bar{r}=\frac{1}{G}\sum_{i=1}^{G} r_i,
\quad
\sigma_r = \sqrt{\frac{1}{G}\sum_i(r_i-\bar{r})^2+\epsilon},
\]

\[
A_i=\frac{r_i-\bar{r}}{\sigma_r}.
\]

A simplified GRPO-style objective:

\[
\mathcal{L}_{GRPO}(\theta)=
-\frac{1}{G}\sum_{i=1}^{G}
\min\left(
\rho_i A_i,
\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i
\right)
\ + \beta D_{KL}(\pi_\theta||\pi_{ref}),
\]

where

\[
\rho_i =
\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}.
\]

GRPO strengths:

- No value model.
- Good for verifiable reasoning tasks.
- Efficient group-relative signal.
- Works naturally with multiple sampled solutions.

GRPO risks:

- Group size affects variance and cost.
- Reward hacking.
- Format overfitting.
- Long reasoning traces can become verbose or self-indulgent.
- Multilingual reasoning may lag if rewards are English-centric.

## 15.4 RLVR

Reinforcement learning with verifiable rewards (RLVR) should be the default for
math/code/science reasoning where possible. The reward function should be
transparent:

```python
def reward(problem, answer):
    parsed = parse_final_answer(answer)
    if not parsed.valid:
        return -0.2
    if equivalent(parsed.value, problem.gold):
        return 1.0
    return 0.0
```

For code:

```python
def reward(problem, completion):
    program = extract_code(completion)
    result = run_tests_in_sandbox(program, problem.tests)
    return result.passed / result.total
```

Reward design must penalize invalid formatting, unsafe tool calls, and
non-termination. But do not over-penalize alternative reasoning paths if the
answer is correct.

## 15.5 Thinking Budgets

Qwen3-style thinking/non-thinking modes suggest a practical interface: users or
systems can allocate a reasoning budget [@qwen3_2025]. Train the model with
explicit control tokens or system instructions:

```text
<mode>fast</mode> Answer directly.
<mode>think</mode> Use a hidden scratchpad, then provide the concise answer.
<budget>2048</budget>
```

For open models, be careful with public chain-of-thought. A useful compromise is
hidden reasoning during training and concise rationales at inference. If the
project releases reasoning traces, filter for privacy, hallucinated citations,
and unsafe procedural content.

# 16. Agent Capability Training

## 16.1 What Agent Capability Means

Agent capability means the model can plan, use tools, observe results, recover
from errors, and complete tasks across multiple steps. It is not the same as
chat helpfulness.

Agent tasks include:

- Search with citation.
- Code editing.
- Running tests.
- Browser navigation.
- Spreadsheet/document manipulation.
- API use.
- Database querying.
- Multi-step research.
- Workflow automation.

ReAct and Toolformer are early foundations for reasoning-plus-acting and tool
use [@yao2022react; @schick2023toolformer]. Llama 3 explicitly includes tool
usage as a native capability target [@llama3herd2024]. In 2026, agent training
should be a first-class post-training stage.

## 16.2 Agent Data

Agent training data:

- Human tool-use traces.
- Synthetic tool-use tasks with verified outcomes.
- Code repair trajectories.
- Browser tasks.
- Document editing tasks.
- API schema following.
- Failure recovery examples.
- Permission-denial examples.
- Cost-aware examples.

Trajectory format:

```json
{
  "task": "Fix the failing unit test and summarize the patch.",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "I will inspect the failure."},
    {"role": "assistant", "tool_call": {"name": "run_tests", "args": {}}},
    {"role": "tool", "content": "1 failing test..."},
    {"role": "assistant", "tool_call": {"name": "edit_file", "args": {...}}},
    {"role": "assistant", "content": "Fixed..."}
  ],
  "outcome": "tests_pass",
  "reward": 1.0
}
```

## 16.3 Agent Evaluation

Metrics:

- Task completion.
- Tool-call validity.
- Number of unnecessary tool calls.
- Recovery from tool errors.
- Permission safety.
- Sandbox compliance.
- Citation correctness.
- Patch correctness.
- Latency and cost.
- Human preference.

Agent models must learn to ask for permission when needed, avoid destructive
actions, and keep working through recoverable failures.

## 16.4 Training Methods

Stages:

1. Tool schema SFT.
2. Demonstration trajectory SFT.
3. Environment rollouts with reward.
4. Preference tuning on trajectory quality.
5. Safety tuning for permissions and irreversible actions.
6. Regression tests on non-agent chat tasks.

Use RL for environments with objective success signals. Use preference training
for style, helpfulness, and judgment. Do not use free-form model-judge rewards
alone for high-stakes agent behavior.

# 17. Safety, Compliance, And Alignment

## 17.1 Safety Is Multilingual

Safety behavior must be evaluated across languages. A refusal policy that works
in English may fail in Polish, Swedish, Maltese, Bulgarian, or code-switched
prompts. Safety data must include:

- Direct harmful requests.
- Translated harmful requests.
- Code-switched harmful requests.
- Indirect prompt injection.
- Tool-use abuse.
- Public-sector sensitive scenarios.
- Medical/legal/financial boundary cases.
- Privacy extraction attempts.

## 17.2 Refusal Calibration

Good safety is not maximum refusal. It is calibrated helpfulness:

- Answer benign questions.
- Redirect dangerous instructions.
- Provide safe alternatives.
- Avoid moralizing verbosity.
- Preserve multilingual quality.

Track:

- False refusal rate.
- False compliance rate.
- Helpfulness after refusal.
- Language-specific refusal disparities.
- Jailbreak robustness.

## 17.3 Model And Dataset Cards

Every release should include:

- Intended uses.
- Out-of-scope uses.
- Data summary.
- Training stages.
- Evaluation results.
- Safety results.
- Known limitations.
- Language coverage.
- License and compliance notes.
- Contact and removal process.

Model cards should not overclaim. If a language has synthetic evaluation only,
say so.

# 18. Evaluation

## 18.1 Evaluation Taxonomy

Evaluate:

- Base language modeling.
- Knowledge.
- Reasoning.
- Math.
- Code.
- Multilingual tasks.
- Translation and cross-lingual transfer.
- Long context.
- Instruction following.
- Tool use.
- Agent tasks.
- Safety.
- Bias and representational harms.
- Robustness.
- Memorization and privacy.

## 18.2 Multilingual Evaluation

The public [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) workflow
supports reproducible task groups across clusters. Multilingual benchmarks
still need language-specific review before broad claims. Make evaluation
status explicit:

- Native validated.
- Professional translation.
- Community reviewed.
- Synthetic fallback.
- Missing.

Report parse failure rate:

\[
\text{parse failure rate} =
\frac{\#\text{unparseable outputs}}{\#\text{total outputs}}.
\]

This matters because models often "know" the answer but fail the format, or
produce unparseable multilingual output.

For every reported score, pin the checkpoint and tokenizer revisions, task
dataset revision, prompt, shot count, answer normalization, metric, decoding
configuration and harness commit. Small changes to these settings can change
rankings. Keep a stable headline panel and a clearly labeled experimental
panel; publish raw predictions so language-specific parse and length bias can
be diagnosed.

## 18.3 Long-Context Evaluation

Long-context benchmarks:

- Single needle retrieval.
- Multi-needle retrieval.
- Lost-in-the-middle curves.
- Variable tracking.
- Long-document QA.
- Cross-lingual context/instruction.
- Long-context perplexity.
- Multi-document synthesis with citations.

Report performance by context length and position:

\[
\text{accuracy}(l,p,n)
\]

where \(l\) is language, \(p\) is evidence position, and \(n\) is context length.

## 18.4 Reasoning Evaluation

Reasoning evaluation should include pass@k:

\[
\operatorname{pass@k} =
1 - \frac{\binom{n-c}{k}}{\binom{n}{k}},
\]

where \(n\) completions are sampled and \(c\) are correct.

But pass@k can hide verbosity, invalid formatting, and compute cost. Report:

- pass@1, pass@8, pass@32.
- Tokens per solution.
- Verifier pass rate.
- Invalid output rate.
- Self-correction success.
- Language-specific reasoning performance.

## 18.5 Agent Evaluation

Agent evals should run in sandboxes and measure actual outcomes. For coding,
use tests. For browser tasks, use DOM state. For research tasks, use citation
precision and answer faithfulness. For document tasks, render outputs and verify
layout.

# 19. Release Engineering

## 19.1 Release Tiers

Release:

- Base model.
- Instruct model.
- Reasoning model.
- Agent model if trained.
- Safety classifier or guard model.
- Smaller distilled variants.
- Quantized variants.

Each release should have a reproducible build path.

## 19.2 Artifact Manifest

Illustrative schema (not a real OpenEuroLLM checkpoint):

```yaml
model_name: example-base-v1
architecture: dense_decoder_transformer
tokenizer: tokenizer-revision-and-commit
training_tokens: measured_total
context_length: 8192
long_context_variant: measured-variant-or-none
data_recipe: immutable-mixture-manifest
code_commit: full-git-sha
checkpoint: exact-step-and-hash
license: verified-weight-license
eval_report: evals/pinned-run.json
model_card: README.md
```

## 19.3 Intermediate Checkpoints

Intermediate checkpoints are scientifically valuable. Release them at:

- Early stable checkpoint.
- 25%, 50%, 75%, 100% of training.
- Before and after mid-training.
- Before and after post-training.

This supports mechanistic interpretability, data studies, and reproducibility.

# 20. OpenEuroLLM-Specific Technical Roadmap

This is a **release-readiness sequence**, not a claim that these tasks are
complete or an internal schedule. The project's [official deliverables](https://openeurollm.eu/deliverables)
remain the authority for formal dates.

## 20.1 Freeze the Evidence Chain

- Pin source and collection revisions in the public [Training Data Collection](https://github.com/OpenEuroLLM/training-data-collection), including `metadata.yaml`, counts, selection rules and packing configuration.
- Pin the tokenizer revision in every data and model manifest. Compare v1 and [v2](https://huggingface.co/openeurollm/tokenizer-256k-v2) using the same multilingual, code, math and chat test sets; verify special-token round trips and document boundaries.
- Publish a model lineage graph linking base, annealed, context-extended and post-trained checkpoints. Mark experimental and formal-release status separately.

## 20.2 Make Comparisons Reproducible

- Lock multilingual evaluation suites, prompt templates, few-shot counts, normalization, decoding settings and metric versions in [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval). Keep raw predictions for error analysis.
- Evaluate long context at multiple lengths, depths and languages. Report exact retrieval separately from answer format, multi-document synthesis and short-context regression.
- Publish per-language and per-domain data-ablation results before changing the production mix; compare token shares as well as row shares.

## 20.3 Turn Post-Training Runs into Releases

- Use versioned YAML in the public [post-training framework](https://github.com/OpenEuroLLM/post-training) for SFT and preference runs, with code revision, data revisions, packing, loss mask, optimizer, seed, hardware and checkpoint lineage.
- Use the released [translated SFT](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated), [reasoning](https://huggingface.co/datasets/openeurollm/reasoning-traces-multilingual), [tool-use](https://huggingface.co/datasets/openeurollm/oellm-eu-tooluse-v1), [math RLVR](https://huggingface.co/datasets/openeurollm/oellm-math-rlvr) and [code RLVR](https://huggingface.co/datasets/openeurollm/oellm-code-rlvr) resources as candidates with explicit licensing, contamination and fit checks. Dataset publication alone does not prove inclusion in a checkpoint.
- Run a small end-to-end rehearsal of each stage before large runs, then compare capability gains and regressions on the same frozen evaluation suite.

## 20.4 Publish with Clear Limits

- Complete model and dataset cards before announcing a checkpoint as a release. Explain missing cards or incomplete lineage in experimental repositories.
- Release weights, tokenizer, code, manifests, run logs, evaluation configurations and raw results where terms permit. State any unavailable inputs and provide executable reconstruction steps.
- Maintain a correction process for dataset issues, benchmark contamination and model-card errors. Date every status claim in this living guide.

## 20.5 Repository Field Guide

OpenEuroLLM is a collection of connected tools, not one monolithic training
repository. Start with the question you are trying to answer. The table below
describes publicly documented roles as of 5 October 2026; it is a navigation
guide, not a claim that every tool was used for every model.

**Data and pretraining.**

| Question | Public resource | What it does | Inspect first |
|---|---|---|---|
| Which source corpora are available? | [training-data-catalogue](https://github.com/OpenEuroLLM/training-data-catalogue) | Curates candidate pretraining datasets, versions, language references, corpus statistics and licensing notes. Inclusion does not certify legal suitability. | Source entry, version, terms and language/script coverage. |
| Which data went into one training cycle? | [training-data-collection](https://github.com/OpenEuroLLM/training-data-collection) | Organizes cycle-specific source subsets, `metadata.yaml`, counts, sampling and release recipes. | Cycle directory, tokenizer property, source counts and subset rules. |
| How were selected records packaged? | [training-data-packer](https://github.com/OpenEuroLLM/training-data-packer) | Reads collection metadata; applies sampling, decontamination and PII masking; merges files and summarizes metrics. | `metadata.yaml`, lint result, release manifest and transformation logs. |
| How are pretraining experiments launched? | [oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp) | Uses declarative configurations to plan sweeps, launch and monitor Slurm jobs. | Experiment config, code revision, container, data revision and restart record. |

**Post-training and evaluation.**

| Question | Public resource | What it does | Inspect first |
|---|---|---|---|
| How are SFT and DPO runs configured? | [post-training](https://github.com/OpenEuroLLM/post-training) | Provides configuration-driven SFT/DPO with TRL or LlamaFactory backends and multi-node support. | YAML, data transforms, packing, loss mask and checkpoint lineage. |
| How is benchmark overlap removed? | [post-training-decontamination](https://github.com/OpenEuroLLM/post-training-decontamination) | Indexes post-training datasets and searches benchmark overlap with n-gram matching. | Benchmark list, n-gram settings, removed IDs and output revision. |
| How are standard tasks run on clusters? | [oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) | Runs reproducible model/task evaluations across EuroHPC environments. | Task group, shot count, prompt, metric, harness commit and raw outputs. |
| How are open-ended answers compared? | [JudgeArena](https://github.com/OpenEuroLLM/JudgeArena) | Runs pairwise and arena-style evaluations with swappable local or remote judges, including multilingual tasks. | Judge identity, baseline, task version, generation settings and resolved config. |

These tools answer different questions. The Catalogue is a discovery layer;
the Collection is a cycle-specific recipe; the Packer executes data
transformations. `oellm-eval` runs task-based measurement, while JudgeArena
compares generated answers through a judge. Neither result can be interpreted
without its pinned configuration and raw outputs.

## 20.6 Worked Paths Through Repositories and Data

**Example A: from candidate corpus to base-model evidence.** Begin with the
[Catalogue](https://github.com/OpenEuroLLM/training-data-catalogue) entry for a
multilingual source such as HPLT or a domain-specific source such as FineMath.
Review its origin, version, license notes and language coverage. In the
[Collection](https://github.com/OpenEuroLLM/training-data-collection), specify
the cycle, tokenizer, sampling rule and token budget. Run the
[Packer](https://github.com/OpenEuroLLM/training-data-packer) with linting and
record the resulting manifest. Use [AutoExp](https://github.com/OpenEuroLLM/oellm-autoexp)
to launch a pinned training config, then evaluate the checkpoint with
[oellm-eval](https://github.com/OpenEuroLLM/oellm-eval). The evidence chain is
source version -> selected records -> packed tokens -> run config -> checkpoint
-> evaluation output. This is an illustrative path, not an assertion about a
specific published checkpoint.

**Example B: from base checkpoint to an assistant.** Choose a base checkpoint
and log its exact revision. Candidate SFT data include the published
[translated Dolci instruction set](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated);
tool-use examples are available in
[oellm-eu-tooluse-v1](https://huggingface.co/datasets/openeurollm/oellm-eu-tooluse-v1).
The latter card currently labels its rows English, so its name alone is not
evidence of multilingual tool-use coverage. Check dataset terms and benchmark
overlap with [post-training-decontamination](https://github.com/OpenEuroLLM/post-training-decontamination),
then configure SFT or DPO in [post-training](https://github.com/OpenEuroLLM/post-training).
Compare the resulting model with its base and a fixed baseline using
[JudgeArena](https://github.com/OpenEuroLLM/JudgeArena), and run the frozen task
suite in `oellm-eval`. The published [9B SFT model card](https://huggingface.co/openeurollm/oellm-9b-256k-sft)
is an example of reporting actual source and token shares, packing, hardware,
loss mask, export checks and long-context probes. It does not imply that the
candidate datasets above were included in that run.

**Example C: from a reasoning dataset to a verifiable test.** The published
[math RLVR](https://huggingface.co/datasets/openeurollm/oellm-math-rlvr)
dataset contains multilingual prompts, ground truth, verifier metadata and
contamination groups. The [code RLVR](https://huggingface.co/datasets/openeurollm/oellm-code-rlvr)
dataset includes programming problems, tests and verifier fields; its card
currently labels the prompt language English. Split by semantic or
contamination group before training. Validate the verifier on a held-out set,
track reward hacking and invalid outputs, and compare both capability gain and
regression on non-reasoning tasks. Dataset availability is not proof that a
specific model was RL-trained on it.

## 20.7 Research Brief: Active Post-Training Work on LUMI

The released [OpenEuroLLM 9B SFT card](https://huggingface.co/openeurollm/oellm-9b-256k-sft)
provides a useful starting diagnosis. Nemotron-v2 math accounted for 18.77% of
source rows but 64.41% of the tokenized pool, so token exposure matters more
than row counts alone. The SFT used sequences up to 4,096 tokens although the
model is configured for 262,144; its 40/40 single-needle retrieval result and
0/5 exact-format compliance at 128K show why long-context claims need
task-specific probes. Its SFT sources had no held-out validation split, so
training loss alone cannot establish generalization.

Against that backdrop, the following are four **distinct experiments led by
the author on LUMI**, using public OpenEuroLLM checkpoints, datasets and code.
Statuses reflect LUMI accounting and run artifacts checked on 5 October 2026.
They are a time-stamped research record, not formal project deliverables.

### Reasoning SFT: improve reasoning without losing the assistant

The published [reasoning-v1 checkpoint](https://huggingface.co/birgermoell/oellm-9b-256k-reasoning-v1)
is a completed 2,000-step full-weight SFT continuation. Its model card reports
gains on ARC-Challenge and flexible-answer multilingual MGSM, but regressions
on GSM8K, IFEval and MMLU college computer science. It is therefore an
experimental artifact rather than an accepted upgrade. The current
[reasoning-training recipe](https://github.com/BirgerMoell/oellm-reasoning-training)
tests a separate Anneal-300B 9B parent with 524.3 million rendered tokens,
translated Dolci reasoning, and instruction replay. Its latest submitted
production attempt failed at chat-template validation **before an optimizer
update**. The immediate work is to repair and qualify the assistant loss mask,
then compare reasoning gains with instruction, language and loop regressions.

### DPO: learn preferences from three differently shaped sources

A full-weight 9B DPO run was **training** on four LUMI nodes at this check.
It starts from the [Anneal-300B instruct SFT model](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft)
and stages approximately 1.48 million preference pairs from
[translated Dolci DPO](https://huggingface.co/datasets/openeurollm/Dolci-Instruct-DPO-translated),
[HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) and
[UltraFeedback](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned).
The multilingual Dolci component numerically dominates this natural-size
mixture; it is not a balanced three-way sample. A step-9,000 intermediate
checkpoint had been saved, but its training preference accuracy is **not** a
held-out quality result. The decision gate is paired evaluation against the
unchanged SFT parent, with per-language preference and general-capability
regressions. The public [post-training framework](https://github.com/OpenEuroLLM/post-training)
shows the configuration-driven SFT/DPO stack; exact run code and artifacts
still need a reproducible release.

### Verifiable-reward RL: test whether math gains transfer

The public [math RLVR dataset](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr)
and [RLVR run records](https://github.com/BirgerMoell/oellm-rlvr) support a
sequence of bounded LUMI experiments. The released
[Phase A checkpoint](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-phase-a-32step-experimental)
completed 32 additional updates. A corrected 64-update Phase B qualification
also completed and was evaluated. Its step-64 export scored **45.12%** on a
frozen 512-prompt multilingual procedural-math holdout, but only **4.00%** on
a 1,024-prompt harder external-family diagnostic. That mixed result does not
establish broad reasoning improvement. A 10,000-problem difficulty survey,
quality gate, 256-update run and paired evaluation were queued at this check;
no result from that chain is claimed. The next decision depends on verifier
quality, disjoint semantic groups, reward variation, language slices and
external-family regressions, not rising training reward alone.

### Tool calling: separate valid calls from unnecessary calls

The tool-use continuation combines the public
[oellm-eu-tooluse-v1 dataset](https://huggingface.co/datasets/birgermoell/oellm-eu-tooluse-v1)
with an equal rendered-token budget of instruction replay. Its checked build
contains 36,642 tool training rows and 31,695 replay rows. A five-update
LUMI smoke test completed after fixing a Qwen3 chat-template and assistant-mask
compatibility issue; checkpoint validation, four-node SFT and paired evaluation
were still queued. The evaluation compares the parent and candidate on valid
call syntax, function name, exact arguments and abstention when an irrelevant
tool is available. The source tool data are labeled English; success on this
test would not by itself establish multilingual tool use. The
[run record](https://github.com/BirgerMoell/oellm-reasoning-training) documents
the training path and its gates.

Across all four streams, the release standard is the same: pin the parent and
tokenizer, publish data lineage and transformations, pass a small executable
gate, retain intermediate checkpoints, freeze evaluation before looking at
results, and report failures beside gains. Current LUMI-only observations
should be treated as provisional until the corresponding configs, model cards,
raw outputs and independent comparisons are public.

# 21. Trade-Off Summary

| Decision | Best 2026 Default | Upside | Risk |
|---|---|---|---|
| Dense vs MoE | Dense baseline plus MoE candidate | Reliable baseline and scalable frontier | MoE complexity |
| Full vs sparse attention | Full baseline; sparse as long-context track | Quality and debuggability | Sparse serving mismatch |
| SFT vs DPO/RLHF | SFT + DPO default | Stable open post-training | May underperform RL on reasoning |
| GRPO/RLVR | Use for verifiable reasoning | Strong reasoning gains | Reward hacking |
| Agent training | Separate tool/agent stage | Real task capability | Safety and eval complexity |
| Long context | System-level validation | Real long-doc utility | Nominal context overclaims |
| Data openness | Manifests + recipes + filters | Auditability | Legal complexity |

# 22. The "World-Class" Checklist

A world-class fully open LLM project in 2026 should be able to answer:

- What exact data was used?
- What was removed and why?
- Which languages are genuinely covered?
- How was tokenizer quality measured?
- What architecture alternatives were tested?
- What scaling-law evidence justified the final run?
- What was the sustained MFU?
- How many restarts happened?
- What were the largest training incidents?
- How did validation loss evolve by language and domain?
- What post-training data changed behavior?
- Which rewards were verifiable?
- Did reasoning training harm normal chat?
- Did agent training create unsafe tool behavior?
- Which benchmarks may be contaminated?
- Which claims are native-speaker validated?
- What are the known limitations?
- Can another lab reproduce the recipe?

If the answer to any of these is "we do not know", the release can still be
valuable, but the uncertainty must be disclosed.

# 23. Conclusion

Training a fully open LLM in 2026 is not a single training run. It is a
multi-year scientific infrastructure project. The model is only one artifact.
The real deliverable is an open system of evidence: data lineage, architecture
decisions, training stability, post-training methodology, evaluation discipline,
safety analysis, and release reproducibility.

For OpenEuroLLM, the opportunity is unusually important. The project can define
what openness means for European multilingual AI: not merely open weights, but
open science, open evaluation, open recipes, and honest multilingual coverage.
The strongest path is pragmatic: build a stable dense baseline, run disciplined
data and scaling ablations, standardize HPC recipes, evaluate multilingual
coverage rigorously, add post-training in carefully measured stages, use
GRPO/RLVR where rewards are verifiable, train agent capabilities explicitly, and
release the artifacts needed for others to inspect and improve the work.

The central rule is simple: if a claim matters, make it measurable; if a result
matters, make it reproducible; if a risk matters, make it visible.

# References

Use `references.bib` with Pandoc, Quarto, or another citation-aware renderer to
format the bibliography.
