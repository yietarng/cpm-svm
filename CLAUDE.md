# CLAUDE.md

## Project Overview

`cpm-svm` is a C++ implementation of a linear binary Support Vector Machine trained via the **BMRM** (Bundle Methods for Regularized Risk Minimization) algorithm, also known as the Cutting Plane Method (CPM). It accepts data in LIBSVM sparse format, performs a train/test split, and reports accuracy.

The repository previously contained a Python layer called **NOC** (New OpenClaw) — a context-management and multi-agent framework for LLM workloads — which was removed in commit `734eb85`.

---

## Repository Layout

```
cpm-svm/
├── cpm-svm/              # C++ source (the only active code)
│   ├── main.cpp          # CLI entry point
│   ├── svm.h / svm.cpp   # SVM training (BMRM loop), prediction, error
│   ├── data.h / data.cpp # Data loading (LIBSVM format), split, shuffle
│   ├── solve_qp.h / solve_qp.cpp  # QP sub-problem solver (cutting plane)
│   ├── linear_algebra.h  # Type aliases and timing utility
│   └── cpm-svm.pro       # Qt project file (build config)
├── requirements.txt      # Currently empty
├── .gitignore            # Currently empty
└── CLAUDE.md             # This file
```

---

## Algorithm

Training uses the BMRM / Cutting Plane Method:

1. Start with `w = 0`.
2. Each iteration:
   - Compute the empirical risk subgradient of the hinge loss over training data.
   - Append the linearisation `(a_t, b_t)` to the cutting-plane model.
   - Solve the resulting QP via a pairwise Frank–Wolfe method (`SolveQP`).
   - Update `w = -∑ αᵢ aᵢ / λ`.
3. Stop when `ε_current ≤ ε_abs` OR `ε_current ≤ ε_tol · J(w)` OR `t ≥ tMax`.

The objective is `J(w) = λ · ½||w||² + empirical_hinge_loss(w)`.

---

## Build

The project uses a **Qt `.pro` file** and requires:

- **Boost** (headers only — `boost/numeric/ublas` is used for `Vec` and `Mat`)
- **Qt** build tools (`qmake`)

```bash
cd cpm-svm
qmake cpm-svm.pro
make
```

The `.pro` file has references to a local Mosek path (`/home/sergei/DevTools/mosek/...`) that are active but may not be present on your machine. If Mosek is unavailable, comment out those `LIBS` lines — the QP solver does not use Mosek; it uses a hand-written pairwise Frank–Wolfe solver.

---

## CLI Usage

```
./cpm-svm <data_file> <train_portion> <mix_seed> <lambda> <epsilon_abs> <epsilon_tol> <max_iter>
```

| Argument | Type | Description |
|---|---|---|
| `data_file` | path | LIBSVM-formatted dataset |
| `train_portion` | float (0–1) | Fraction of samples used for training |
| `mix_seed` | int | Seed for shuffling (`< 0` = no shuffle) |
| `lambda` | float | Regularisation strength |
| `epsilon_abs` | float | Absolute convergence threshold |
| `epsilon_tol` | float | Relative convergence threshold |
| `max_iter` | int | Hard cap on BMRM iterations |

Example:

```bash
./cpm-svm data/train.libsvm 0.8 42 0.01 1e-4 1e-3 100
```

---

## Data Format

LIBSVM sparse format — one sample per line:

```
<label> <index>:<value> <index>:<value> ...
```

- Labels must be `-1` or `+1`.
- Feature indices are 1-based.
- The parser infers `varNumber` from the highest index seen.

---

## Key Types (`linear_algebra.h`)

| Alias | Underlying type |
|---|---|
| `Real` | `double` |
| `Vec` | `boost::numeric::ublas::vector<double>` |
| `Mat` | `boost::numeric::ublas::matrix<double>` |
| `SparseMat` | `std::vector<std::list<Pair>>` |
| `Pair` | `{Real value; int idx;}` (1-based index) |

`gettimeus()` returns microseconds via `gettimeofday`.

---

## Code Conventions

- **Language**: C++03/C++11 with Boost.uBLAS; no STL algorithms beyond `std::fill`, `std::max_element`.
- **Naming**: `CamelCase` for classes and methods; `lower_snake_case` for local variables; `camelCase` for private members (`betta`, `trainCount`).
- **Error handling**: exceptions (`SVM::Exception`, `Data::Exception`) for unrecoverable states; boolean returns for I/O failures.
- **Comments**: sparse, mostly Russian-language inline notes — preserve existing style; do not add English comment blocks.
- **No test suite**: correctness is verified by observing train/test accuracy output.
- **Build macro**: `#define BMRM_INFO` in `svm.h` enables per-iteration console logging. Comment it out for silent training.

---

## Git Workflow

- Active development branch: `claude/add-claude-documentation-YE2q2`
- Main branch: `master`
- Push: `git push -u origin <branch-name>`
- Commit messages follow imperative mood, short summary line.

---

## Historical Context (NOC Layer)

A Python package `noc/` was added in PR #1 (commit `dc87a1f`) and subsequently deleted (commit `734eb85`). It implemented:

- **`session.py`** — `NOCSession`: orchestrates plan→execute→replan loop.
- **`agent.py`** — `PlannerAgent`, `ExecutorAgent`, `ReplannerAgent`, `SummarizerAgent`.
- **`context_manager.py`** — `ContextDecomposer`, `SemanticIndex`, `ContextComposer`.
- **`kv_cache.py`** — `KVCacheManager`: position-independent KV cache per SubContext.
- **`local_llm.py`** — `OllamaClient`, `LlamaCppClient`, `VLLMClient`, `TransformersClient`, `LocalLLMRouter`.
- **`router.py`** — `RequestRouter`: rule-based request classification and agent dispatch.
- **`compressor.py`** — `ContextCompressor`: token-budget-aware context compression.
- **`subcontext.py`** — `SubContext`: atomic unit of session memory.

This code is no longer present but lives in git history. If revived, it requires the dependencies that were in the now-empty `requirements.txt`.
