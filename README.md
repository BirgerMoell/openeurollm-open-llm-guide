# The 2026 Fully Open LLM Training Guide

Public website and PDF for Birger Moëll's technical field guide to training fully open, multilingual foundation models in 2026, using OpenEuroLLM public artifacts as worked examples.

The October 2026 edition uses the shock-pink editorial cover and matching web theme. It distinguishes public experimental artifacts from formal deliverables and links directly to the relevant OpenEuroLLM repositories and Hugging Face cards.

The site opens with a research brief built around the public 9B SFT model card and a proposed controlled multilingual RLVR experiment. The interactive ecosystem explorer in `index.html` and `ecosystem.js` covers 16 public repositories, datasets, tokenizers and models. It offers three example workflows, stage filters, search and source-linked details. The PDF includes the same research brief and a repository field guide in `docs/paper.md`.

## Read

- Website: GitHub Pages after deployment
- PDF: [`docs/fully-open-llm-guide-2026.pdf`](docs/fully-open-llm-guide-2026.pdf)
- Markdown source: [`docs/paper.md`](docs/paper.md)
- Bibliography: [`docs/references.bib`](docs/references.bib)

## Scope

The guide covers data governance, corpus curation, multilingual mixture design, tokenizer validation, architecture choices, scaling laws, HPC training, pretraining, long-context adaptation, post-training, reasoning training, GRPO/RLHF/DPO/RLVR, agent capability training, evaluation, safety, release engineering, and reproducibility.

## Publication Note

The manuscript grounds project-status claims in dated public sources and separates released evidence from proposed experiments and formal deliverables. This is an independent guide, not an official OpenEuroLLM project deliverable. The PDF is generated from the manuscript in `../openeurollm-fully-open-llm-guide-2026/` and copied into `docs/`.

## License

No project license has been selected in this generated package. Add a `LICENSE` file before encouraging reuse.
