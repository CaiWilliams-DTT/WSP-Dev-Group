# styleprofiler

Learn a user's **writing-style preferences** through **pairwise comparisons**,
then compile the learned profile into reusable AI writing instructions
(a Claude skill, a system-prompt block, or JSON).

This is **active preference learning** (adaptive choice-based conjoint analysis),
not reinforcement learning: we maintain a Bayesian posterior over a latent
utility function on a multi-attribute style space, and actively pick each next
comparison to learn the *whole* profile efficiently.

## How it works

- A **style configuration** is a point in a discrete space — each dimension
  (verbosity, formality, warmth, …) has ordered levels.
- Utility is `u(x) = wᵀφ(x)`, with `φ` a one-hot encoding (one reference level
  dropped per dimension for identifiability).
- A pairwise choice `a ≻ b` is modelled with a logistic (Bradley–Terry)
  likelihood; we keep a **Laplace-approximated posterior over `w`**.
- Each next comparison is chosen to **maximise information gain** about `w`
  (predictive-variance / BALD-style), so the full profile is characterised in
  fewer comparisons than random questioning.

The learning loop is **headless**: a driver pulls a query and pushes a result.
The CLI is one driver; a web/GUI front-end would be another, using the same
[`ProfilingSession`](src/styleprofiler/engine.py) API.

## Install

This project uses [uv](https://docs.astral.sh/uv/).

```sh
uv sync                # core (offline template generator, no API key needed)
uv sync --extra llm    # also install the optional Anthropic generator
```

## Quickstart

**Simulated run** (no human, no API key) — exercises the whole loop and prints a
profile:

```sh
uv run styleprofiler run --simulated --max-rounds 20
```

**Interactive run** in the terminal using the offline template generator:

```sh
uv run styleprofiler run --max-rounds 15
```

You'll be shown two samples (A/B) per round and asked which you prefer
(`a`, `b`, or `s` to skip). The profile is saved to `profile.json` and the
resumable session to `profile.session.json`.

**Real-prose run** with the Anthropic generator (needs `uv sync --extra llm` and
`ANTHROPIC_API_KEY`; falls back to the template generator with a warning if
unavailable):

```sh
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell;  export on bash
uv run styleprofiler run --generator llm --topic "Explain why CI pipelines flake."
```

**Compile** a saved profile into reusable instructions:

```sh
uv run styleprofiler compile profile.json --target skill   # SKILL.md
uv run styleprofiler compile profile.json --target prompt  # system-prompt block
uv run styleprofiler compile profile.json --target json    # structured profile
```

## Convergence evaluation

Show that information-gain acquisition recovers the profile in fewer comparisons
than random questioning:

```sh
uv run python scripts/eval_convergence.py --trials 40 --budgets 16 24 32
```

It reports, per comparison budget, the **preferred-level match accuracy** and the
**utility rank-correlation** against a simulated user's hidden ground truth, for
random vs. info-gain selection.

## Configuration

The style schema is **data**: edit [`src/styleprofiler/schema.yaml`](src/styleprofiler/schema.yaml)
to change dimensions/levels without touching code. Runtime knobs (prior
precision, cold-start rounds, candidate-pool size, stopping thresholds, default
base task, generator, Anthropic model) live in
[`settings.py`](src/styleprofiler/settings.py) and can be overridden via
`STYLEPROFILER_*` environment variables or a `.env` file.

## Using the engine directly (headless)

```python
from styleprofiler.factory import build_session
from styleprofiler.elicitation.cli import CLIFeedbackSource

session = build_session()
feedback = CLIFeedbackSource()
while not session.is_complete():
    q = session.next_query()
    session.submit_feedback(feedback.ask(q.sample_a, q.sample_b))
profile = session.get_profile()
print(profile.summary())
```

Any front-end implements the same three-call loop. No domain logic lives in the
CLI or the feedback sources.

## Architecture

```
schema.yaml -> StyleSchema -> engine builds candidate configs -> acquisition
picks a pair -> generator renders two samples (same base task) -> driver shows
them and returns the choice -> model updates posterior -> repeat until stopping
-> StyleProfile -> compiler -> outputs
```

| Module | Responsibility |
|---|---|
| `schema.py` / `schema.yaml` | Dimensions, levels, configs, feature encoding `φ` |
| `model/logistic.py` | Bayesian logistic preference model (Laplace) |
| `acquisition/` | Cold-start (space-filling) + info-gain selection |
| `generation/` | Template (offline) and LLM (Anthropic) sample generators |
| `elicitation/` | Feedback sources: CLI and simulated oracle (driver-side) |
| `engine.py` | `ProfilingSession` — the headless loop/state machine |
| `profile.py` | `StyleProfile` result (utilities + uncertainty) |
| `compiler/` | SKILL.md / system-prompt / JSON outputs |
| `persistence.py` | Save/resume session and profile |
| `cli/main.py` | Thin `typer` driver |

Everything swappable (model, acquisition, generator, feedback source, compiler)
sits behind an ABC. The model is hand-rolled on numpy/scipy for transparency;
GP preference models (BoTorch/GPyTorch) are the noted upgrade path.

## Development

```sh
uv run pytest          # full suite (offline; no API key required)
```

The whole loop is testable without a human or API key via the `SimulatedUser`
oracle and the offline template generator.
