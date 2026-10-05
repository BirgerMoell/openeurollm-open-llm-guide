const artifacts = [
  {
    id: "catalogue", name: "Training Data Catalogue", stage: "data", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/training-data-catalogue",
    summary: "A curated map of candidate pretraining corpora, with versions, language references, statistics and licensing notes.",
    input: "Public source datasets such as HPLT, FineMath, FinePDFs and code corpora.",
    output: "Entries that help researchers decide what to inspect or acquire.",
    example: "Start with an HPLT entry to check its version, language coverage and source terms before proposing a multilingual mixture.",
    check: "Catalogue inclusion is discovery, not proof of legal suitability or use in a model.",
    next: "collection"
  },
  {
    id: "collection", name: "Training Data Collection", stage: "data", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/training-data-collection",
    summary: "The cycle-specific recipe for selecting, counting and sampling data that may enter a training run.",
    input: "Candidate sources, tokenizer revision and source-level selection rules.",
    output: "Versioned metadata.yaml files, counts and collection recipes.",
    example: "Inspect a cycle directory to see which source parts were selected and how their token counts were measured.",
    check: "Pin the collection revision, tokenizer and sampling rules; avoid treating a catalogue entry as the final mixture.",
    next: "packer"
  },
  {
    id: "packer", name: "Training Data Packer", stage: "data", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/training-data-packer",
    summary: "Executes collection metadata to sample, mask PII, decontaminate and package training files.",
    input: "A collection directory and metadata.yaml with processing rules.",
    output: "Release-ready files, merged shards and processing metrics.",
    example: "Lint a source's metadata, run packaging, then inspect the release files and metrics before tokenization.",
    check: "Keep the input manifest, lint result, logs and output hashes together.",
    next: "autoexp"
  },
  {
    id: "multisynt", name: "MultiSynt", stage: "data", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/MultiSynt",
    summary: "A project entry point for multilingual synthetic data intended for LLM pretraining.",
    input: "Language and domain gaps identified in the pretraining corpus.",
    output: "A synthetic-data research track; inspect linked releases for actual usable data.",
    example: "Explore whether translated or generated material can improve coverage for a lower-resource language.",
    check: "The repository description alone is not a downloadable dataset or evidence of inclusion in a model.",
    next: "catalogue"
  },
  {
    id: "tokenizer", name: "Tokenizer 256k v2", stage: "training", type: "Hugging Face model",
    url: "https://huggingface.co/openeurollm/tokenizer-256k-v2",
    summary: "A released SentencePiece BPE tokenizer with multilingual and special-token evaluation.",
    input: "Text from target languages plus code, math, chat and document formats.",
    output: "Token IDs, vocabulary and special-token mappings used by compatible runs.",
    example: "Compare tokens per word by language and test chat/reasoning markers before freezing a training mixture.",
    check: "The v2 card reports a code-domain trade-off; never silently swap v1 and v2 in a checkpoint.",
    next: "autoexp"
  },
  {
    id: "autoexp", name: "oellm-autoexp", stage: "training", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/oellm-autoexp",
    summary: "Declarative experiment configuration for planning, launching and monitoring Slurm training runs.",
    input: "A pinned model, data, tokenizer, compute and launcher configuration.",
    output: "Training jobs, logs and checkpoints tied to an experiment config.",
    example: "Run a small data-mix sweep before committing the selected recipe to a larger pretraining run.",
    check: "Record the config, code commit, container, data revision, restart events and GPU-hour budget.",
    next: "oellm-eval"
  },
  {
    id: "decontamination", name: "Post-training Decontamination", stage: "post-training", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/post-training-decontamination",
    summary: "An n-gram overlap pipeline for finding benchmark contamination in post-training datasets.",
    input: "Candidate post-training data and benchmark definitions.",
    output: "Overlap findings and a cleaned dataset revision.",
    example: "Index an SFT mixture, search for benchmark overlap, then document which examples were removed.",
    check: "Pin the benchmark list, match settings and removed IDs; overlap detection has limits.",
    next: "posttraining"
  },
  {
    id: "posttraining", name: "Post-training Framework", stage: "post-training", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/post-training",
    summary: "Configuration-driven SFT and DPO training through TRL or LlamaFactory backends.",
    input: "A base checkpoint, curated datasets and a versioned YAML run config.",
    output: "Instruction or preference-tuned checkpoints plus run metadata.",
    example: "Run a small SFT rehearsal with a translated instruction slice, inspect packing and loss masks, then compare the checkpoint.",
    check: "Log exact data revisions, transforms, token shares, loss mask, optimizer and checkpoint lineage.",
    next: "JudgeArena"
  },
  {
    id: "dolci", name: "Dolci Instruct SFT translated", stage: "post-training", type: "Hugging Face dataset",
    url: "https://huggingface.co/datasets/openeurollm/Dolci-Instruct-SFT-translated",
    summary: "Translated instruction conversations organized into language subsets for SFT research.",
    input: "Source instruction data and translation process described by the dataset card.",
    output: "Conversation rows that can be evaluated as a candidate SFT slice.",
    example: "Sample Czech and German rows, inspect response quality, then measure token shares in a proposed SFT mix.",
    check: "Review provenance and terms for all components; publication does not prove use in a checkpoint.",
    next: "posttraining"
  },
  {
    id: "tooluse", name: "OELLM EU Tool Use v1", stage: "post-training", type: "Hugging Face dataset",
    url: "https://huggingface.co/datasets/openeurollm/oellm-eu-tooluse-v1",
    summary: "Function-calling conversations for tool-use SFT experiments.",
    input: "Tool schemas, user requests, calls and responses.",
    output: "Conversation examples for learning valid tool calls and follow-up behavior.",
    example: "Check whether a model selects the right function and arguments, then recovers from tool errors.",
    check: "The current card labels the rows English; the dataset name is not evidence of multilingual coverage.",
    next: "posttraining"
  },
  {
    id: "math-rlvr", name: "OELLM Math RLVR", stage: "post-training", type: "Hugging Face dataset",
    url: "https://huggingface.co/datasets/openeurollm/oellm-math-rlvr",
    summary: "Multilingual math prompts with ground truth, verifier metadata and contamination groups.",
    input: "Generated math problems across languages and difficulty levels.",
    output: "Verifiable training and held-out examples for reasoning experiments.",
    example: "Split by semantic group, validate the exact-answer verifier, and measure pass rate by language.",
    check: "Keep verifier version and holdout policy fixed; dataset availability is not evidence of model training.",
    next: "oellm-eval"
  },
  {
    id: "code-rlvr", name: "OELLM Code RLVR", stage: "post-training", type: "Hugging Face dataset",
    url: "https://huggingface.co/datasets/openeurollm/oellm-code-rlvr",
    summary: "Programming tasks with reference solutions, tests and verifier fields for RLVR research.",
    input: "Code problems and executable test specifications.",
    output: "Problems whose proposed solutions can be checked by tests.",
    example: "Hold out generator families, run hidden tests and track pass rate alongside invalid-code rate.",
    check: "The current card labels prompt language English; isolate test execution and check leakage.",
    next: "oellm-eval"
  },
  {
    id: "oellm-eval", name: "oellm-eval", stage: "evaluation", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/oellm-eval",
    summary: "A CLI for reproducible task-based model evaluation across EuroHPC clusters.",
    input: "A checkpoint and a pinned set of tasks, prompts, metrics and decoding settings.",
    output: "Task scores and raw results for controlled comparisons.",
    example: "Compare a base and SFT checkpoint on the same multilingual task group with identical shot count.",
    check: "Record harness commit, dataset revision, prompt, shot count and parse-failure rate.",
    next: "JudgeArena"
  },
  {
    id: "JudgeArena", name: "JudgeArena", stage: "evaluation", type: "GitHub repository",
    url: "https://github.com/OpenEuroLLM/JudgeArena",
    summary: "Pairwise and arena-style comparison of generated answers with swappable local or remote judges.",
    input: "Candidate models, baseline, evaluation prompts and a specified judge.",
    output: "Comparative results, cached generations and a resolved run configuration.",
    example: "Compare a base and SFT model on m-Arena-Hard with a fixed judge and inspect disagreements by language.",
    check: "Judge choice, baseline, task version and generation settings can change the result.",
    next: "oellm-eval"
  },
  {
    id: "prelude", name: "Prelude", stage: "models", type: "Hugging Face model",
    url: "https://huggingface.co/openeurollm/prelude",
    summary: "A public OpenEuroLLM model artifact to inspect for model lineage and evaluation evidence.",
    input: "A specific checkpoint and its associated tokenizer and model card.",
    output: "A frozen model revision suitable for controlled evaluation or further study.",
    example: "Record its commit and tokenizer before comparing it with an instruction-tuned checkpoint.",
    check: "Read the current card for exact status and scope; do not treat every artifact as the formal first-model deliverable.",
    next: "oellm-eval"
  },
  {
    id: "sft-9b", name: "OELLM 9B 256K SFT", stage: "models", type: "Hugging Face model",
    url: "https://huggingface.co/openeurollm/oellm-9b-256k-sft",
    summary: "An experimental 9B instruction checkpoint with unusually detailed training and long-context reporting.",
    input: "Its published base checkpoint and the card's exact SFT mixture.",
    output: "A BF16 SFT checkpoint, training details and retrieval probe results.",
    example: "Compare the card's row shares and token shares: long reasoning traces occupy far more tokens than row counts suggest.",
    check: "It trained on 4K SFT sequences; the 256K configuration and retrieval probes are separate claims.",
    next: "JudgeArena"
  }
];

const workflows = {
  base: [
    ["catalogue", "Find candidate corpora"], ["collection", "Freeze a cycle recipe"],
    ["packer", "Package selected records"], ["autoexp", "Train from a pinned config"],
    ["oellm-eval", "Measure the checkpoint"]
  ],
  assistant: [
    ["prelude", "Choose a base revision"], ["dolci", "Inspect instruction data"],
    ["decontamination", "Check benchmark overlap"], ["posttraining", "Run SFT or DPO"],
    ["JudgeArena", "Compare open-ended answers"]
  ],
  evaluation: [
    ["sft-9b", "Pin the candidate model"], ["oellm-eval", "Run fixed task scores"],
    ["JudgeArena", "Compare answers with a judge"]
  ]
};

const labels = {data: "Data", training: "Training", "post-training": "Post-training", evaluation: "Evaluation", models: "Models"};
const byId = new Map(artifacts.map((item) => [item.id, item]));
const grid = document.querySelector("#artifact-grid");
const detail = document.querySelector("#artifact-detail");
const count = document.querySelector("#result-count");
const search = document.querySelector("#artifact-search");
const steps = document.querySelector("#workflow-steps");
let activeFilter = "all";
let selectedId = "catalogue";

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value) node.textContent = value;
  return node;
}

function renderDetail() {
  const item = byId.get(selectedId);
  detail.replaceChildren();
  detail.append(element("p", "micro-label", "Selected resource"));
  detail.append(element("p", "detail-stage", `${labels[item.stage]} · ${item.type}`));
  detail.append(element("h3", "", item.name));
  detail.append(element("p", "detail-summary", item.summary));
  for (const [heading, value] of [["Input", item.input], ["Produces", item.output], ["Example", item.example], ["Check before using", item.check]]) {
    const block = element("div", "detail-row");
    block.append(element("h4", "", heading), element("p", "", value));
    detail.append(block);
  }
  const link = element("a", "detail-link", "Open the public source ↗");
  link.href = item.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  detail.append(link);
  const related = byId.get(item.next);
  if (related) {
    const next = element("button", "related-link", `Explore connected resource: ${related.name} →`);
    next.type = "button";
    next.addEventListener("click", () => {
      search.value = "";
      setFilter("all");
      selectArtifact(related.id, true);
    });
    detail.append(next);
  }
}

function filteredArtifacts() {
  const query = search.value.trim().toLocaleLowerCase();
  return artifacts.filter((item) =>
    (activeFilter === "all" || item.stage === activeFilter) &&
    (!query || [item.name, item.summary, item.type, item.example, item.check].join(" ").toLocaleLowerCase().includes(query))
  );
}

function renderCards() {
  const visible = filteredArtifacts();
  count.textContent = `${visible.length} of ${artifacts.length} resources shown`;
  grid.replaceChildren();
  if (!visible.length) {
    grid.append(element("p", "empty-state", "No matching resources. Try a broader search or another stage."));
    return;
  }
  for (const item of visible) {
    const card = element("button", `artifact-card${selectedId === item.id ? " selected" : ""}`);
    card.type = "button";
    card.setAttribute("aria-pressed", selectedId === item.id ? "true" : "false");
    card.append(element("span", "artifact-kind", `${labels[item.stage]} · ${item.type}`));
    card.append(element("strong", "", item.name));
    card.append(element("span", "artifact-summary", item.summary));
    card.append(element("span", "artifact-more", "Explore resource →"));
    card.addEventListener("click", () => selectArtifact(item.id));
    grid.append(card);
  }
}

function selectArtifact(id, scroll = false) {
  if (!byId.has(id)) return;
  selectedId = id;
  renderCards();
  renderDetail();
  if (scroll) detail.scrollIntoView({behavior: "smooth", block: "center"});
}

function setFilter(value) {
  activeFilter = value;
  document.querySelectorAll("[data-filter]").forEach((button) => {
    const active = button.dataset.filter === value;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  renderCards();
}

function renderWorkflow(name) {
  const tab = document.querySelector(`[data-workflow="${name}"]`);
  document.querySelectorAll("[data-workflow]").forEach((button) => button.setAttribute("aria-selected", button === tab ? "true" : "false"));
  steps.setAttribute("aria-labelledby", tab.id);
  steps.replaceChildren();
  workflows[name].forEach(([id, caption], index) => {
    const item = byId.get(id);
    const button = element("button", "workflow-step");
    button.type = "button";
    button.append(element("span", "workflow-number", String(index + 1).padStart(2, "0")));
    button.append(element("strong", "", item.name));
    button.append(element("span", "", caption));
    button.addEventListener("click", () => {
      search.value = "";
      setFilter("all");
      selectArtifact(id, true);
    });
    steps.append(button);
  });
}

document.querySelectorAll("[data-filter]").forEach((button) => button.addEventListener("click", () => setFilter(button.dataset.filter)));
document.querySelectorAll("[data-workflow]").forEach((button) => button.addEventListener("click", () => renderWorkflow(button.dataset.workflow)));
document.querySelector(".workflow-tabs").addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  const tabs = [...document.querySelectorAll("[data-workflow]")];
  const index = tabs.indexOf(document.activeElement);
  if (index < 0) return;
  const next = tabs[(index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length];
  next.focus();
  renderWorkflow(next.dataset.workflow);
  event.preventDefault();
});
search.addEventListener("input", renderCards);

renderWorkflow("base");
renderCards();
renderDetail();
