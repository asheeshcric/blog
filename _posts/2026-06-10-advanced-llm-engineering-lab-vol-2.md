---
title: "Advanced LLM Engineering Lab — Vol. 2: The Agentic Frontier"
date: 2026-06-10 18:30:00 -0500
categories: [Tutorial]
tags: [machine_learning, large_language_models, ai_agents, agentic_ai, llm_engineering]
description: A second interactive, single-file web app — this time about engineering agents, not just understanding models. The agent loop, harness engineering, context & memory systems, building agentic APIs with MCP, generalist-vs-specialist orchestration, training agents with RL, and shipping them with real evals. Built for ML practitioners moving from "I understand LLMs" to "I can build reliable agents."
toc: true
---

[Volume 1]({% post_url 2026-06-05-interactive-field-guide-to-modern-llms %}) rebuilt the
mental model of modern LLM *research* — transformers, scaling laws, RLHF, reasoning, the
2026 frontier. But there's a different skill that's been eating the field this year:
actually **building agents**. Not "call the model in a loop and hope," but the real
engineering discipline around it.

2024 was the year of *prompt engineering*. 2025 was *context engineering*. **2026 is the
year of the harness.** So I built a second interactive lab — a companion volume focused
entirely on the engineering side of agents.

> **Who it's for:** ML practitioners and engineers who understand modern LLMs (or finished
> Vol. 1) and now want to build production-grade agents — covering harness engineering,
> memory, agentic APIs, orchestration, agentic RL, and evals.
{: .prompt-info }

The one idea that runs through the whole thing:

> **Agent = Model + Harness.** The model is the brain you mostly rent; the *harness* is
> everything you build around it — tools, context assembly, memory, permissions,
> orchestration, evals. The same frontier model in a great harness beats a better model in
> a poor one. The harness is your durable IP.
{: .prompt-tip }

## Try it right here

Like Vol. 1, it's a single self-contained HTML file — 8 tracks, 33 short lessons, a
hands-on widget in every lesson, with progress saved locally. No frameworks, no server, no
network calls. Use the sidebar to navigate, poke the widgets, and toggle the
**light/dark theme** from the top of the sidebar. (For the best experience — especially on
mobile — open it fullscreen.)

<p>
  <a href="{{ site.assets_url }}/apps/advanced-llm-lab.html" target="_blank" rel="noopener" class="btn btn-primary">
    ↗ Open the full lab in a new tab
  </a>
  &nbsp;
  <a href="{{ site.assets_url }}/apps/llm-catchup-lab.html" target="_blank" rel="noopener" class="btn">
    ← Vol. 1: LLM Catch-Up Lab
  </a>
</p>

<iframe
  src="{{ site.assets_url }}/apps/advanced-llm-lab.html"
  title="Advanced LLM Engineering Lab — Vol. 2: The Agentic Frontier"
  loading="lazy"
  style="width:100%; height:80vh; min-height:640px; border:1px solid var(--main-border-color, #d9e0ee); border-radius:12px; box-shadow:0 8px 30px rgba(0,0,0,.18);">
</iframe>

## What's inside

Nine progressive tracks. Each lesson pairs a tight conceptual explanation with an
interactive widget, a "what changed recently" note, and a quick self-check.

| # | Track | What you'll build intuition for |
|---|-------|---------------------------------|
| 1 | **The Agent Loop, Properly** | Agent = Model + Harness, the Think–Act–Observe loop, and tools as the action space |
| 2 | **Harness Engineering** | Permissions & sandboxing, hooks that gate/rewrite/transform tool calls, skills & progressive disclosure, self-improving harnesses |
| 3 | **Context Engineering** | Context as a budget, "context rot" & ordering, compaction/self-summarization, KV/prefix-cache reuse for agents |
| 4 | **Memory Systems** | Context ≠ memory (LLM-as-OS), the four memory types, vector vs knowledge-graph stores, the write/retrieve/update/forget lifecycle |
| 5 | **Building Agentic APIs (MCP)** | Why MCP exists, host/client/server & primitives, designing tools an LLM can use, wrapping a REST API, and agentic-API security |
| 6 | **Generalist vs Specialist Agents** | Single vs multi-agent tradeoffs, orchestration patterns, the "orchestration tax," and when specialization actually wins |
| 7 | **Training Agents (Agentic RL)** | SFT on traces, RL on multi-turn trajectories, the credit-assignment problem, and verifiable/checklist rewards |
| 8 | **Eval, Observability & Production** | Why agent eval is hard, outcome vs trajectory evals, OpenTelemetry tracing, debugging a real incident, shipping with guardrails |

Plus three reference views: a **timeline** of the agentic era (2022 → 2026), a
**glossary** of the agent-engineering lexicon, and a printable **cheat sheet** of the
recipes and mental models worth memorizing.

## A few of the interactive widgets

The widgets are the point — reading about the orchestration tax is one thing; sliding the
number of agents up and watching the token cost multiply is another. A sampling:

- **Agent loop stepper** — step a real task through Think–Act–Observe and watch the context
  grow as observations feed back, with each step labeled model-owned vs harness-owned.
- **Hook bench** — send a tool call through a PreToolUse/PostToolUse hook and watch it get
  blocked, rewritten (e.g. inject `--dry-run`), or have its output redacted.
- **Context budget visualizer** — allocate a 200K window across system prompt, retrieval,
  memory, and scratchpad, and watch headroom vanish when you over-stuff retrieval.
- **Memory router** — classify a fact into working / episodic / semantic / procedural
  memory, and see how a vector store vs a knowledge graph handle updates and multi-hop.
- **Tool-schema builder** — toggle best-practices on a tool definition and watch the
  modeled tool-call success rate climb (the description matters most).
- **Orchestration-tax calculator** — scale agents, turns, and per-turn context, and compare
  the total cost against a single agent doing the same job.
- **Turn-vs-token credit lab** — distribute an agentic-RL reward across a trajectory and see
  the "action bottleneck" appear under uniform token-level credit.
- **Trace explorer** — read a real-looking OpenTelemetry trace tree and find the one span
  dragging the whole task down.

## How it's built

Same philosophy as Vol. 1: deliberately low-tech and durable. A single static HTML file
with vanilla JavaScript and `<canvas>` for the visualizations. No build step, no
dependencies, no analytics. It works offline, loads instantly, and will still run years
from now. Theme preference and lesson progress persist via `localStorage`.

If you haven't been through [Volume 1]({% post_url 2026-06-05-interactive-field-guide-to-modern-llms %})
yet and want the model-side foundations first, start there — then come back for the
agentic stack. As always, if you spot something out of date or want a track that goes
deeper somewhere, let me know. The frontier moves fast, and this is meant to be a living
guide.
