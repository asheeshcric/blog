---
title: "Interactive Field Guide to Modern LLMs: From Transformers to the Agentic Frontier"
date: 2026-06-05 18:30:00 -0500
categories: [Tutorial]
tags: [machine_learning, large_language_models, deep_learning, transformers]
description: An interactive, single-file web app to brush up on modern LLM research — self-attention, pre-training and scaling laws, RLHF and the DPO/GRPO alignment zoo, reasoning and test-time compute, efficiency tricks (LoRA, quantization, KV-cache), alternative architectures, multimodal generation, and how to build production agents. Built for ML practitioners catching up after time away.
toc: true
---

The pace of LLM research is relentless. If you know deep learning fundamentals but
have been heads-down on other things for a year or two, the modern landscape can feel
like alphabet soup — GRPO, RLVR, MLA, MoE, dLLMs, test-time compute, agentic RAG.

So I built an **interactive field guide** to rebuild that mental model the fun way: a
self-contained web app with 9 learning tracks, 34 short lessons, and a hands-on widget
in every single lesson. No frameworks, no server, no network calls — it's one HTML file
that runs entirely in your browser, and your progress saves locally.

> **Who it's for:** ML researchers and engineers who understand the basics of deep
> learning but want to catch up on what changed in LLMs from ~2022 to 2026. It's a
> brush-up, not a beginner intro.
{: .prompt-info }

## Try it right here

The whole guide is embedded below. Use the sidebar to navigate tracks, poke the
interactive widgets to build intuition, and toggle the **light/dark theme** from the
top of the sidebar. (For the best experience — especially on mobile — open it
fullscreen.)

<p>
  <a href="{{ site.assets_url }}/apps/llm-catchup-lab.html" target="_blank" rel="noopener" class="btn btn-primary">
    ↗ Open the full guide in a new tab
  </a>
</p>

<iframe
  src="{{ site.assets_url }}/apps/llm-catchup-lab.html"
  title="LLM Catch-Up Lab — Interactive Field Guide to Modern LLMs"
  loading="lazy"
  style="width:100%; height:80vh; min-height:640px; border:1px solid var(--main-border-color, #d9e0ee); border-radius:12px; box-shadow:0 8px 30px rgba(0,0,0,.18);">
</iframe>

## What's inside

The guide is organized as 9 progressive tracks. Each lesson pairs a tight conceptual
explanation with an interactive widget, a "what changed recently" note, and a quick
self-check.

| # | Track | What you'll brush up on |
|---|-------|--------------------------|
| 1 | **Foundations: the Transformer** | Tokenization, self-attention, multi-head attention, RoPE positions, encoder vs decoder |
| 2 | **Pre-training at Scale** | The next-token objective, Chinchilla scaling laws (and why we now "overtrain"), data & synthetic data |
| 3 | **Post-training & Alignment** | SFT, the RLHF pipeline, and the DPO / PPO / ORPO / KTO / SimPO preference-optimization zoo |
| 4 | **Reasoning & Test-Time Compute** | Chain-of-thought, RLVR & GRPO, and spending compute at inference (best-of-N, self-consistency) |
| 5 | **Efficiency & Systems** | LoRA / QLoRA, quantization, the KV-cache (GQA/MLA), FlashAttention, speculative decoding |
| 6 | **Architectures Beyond Vanilla** | Mixture-of-Experts, long-context engineering, and Mamba / state-space-model hybrids |
| 7 | **Generative & Multimodal** | Diffusion for images, diffusion language models, vision-language models, knowledge distillation |
| 8 | **The Agentic Frontier** | Tool use & ReAct, agentic RAG, evaluation & LLM-as-judge, reading the 2026 leaderboard |
| 9 | **Building Agents in Practice** | The open-source agent stack, plus a worked Discovery Agent for a content feed you can replicate |

Plus three reference views: a **timeline** of milestones from 2017 to 2026, a
**glossary** that decodes the acronym soup, and a printable **cheat sheet** of the
formulas and decision recipes worth memorizing.

## A few of the interactive widgets

The widgets are the point — reading about attention is one thing, dragging a query
token around and watching the attention distribution reshape is another. A sampling:

- **Self-attention heat-flow** — pick a query token, adjust temperature, toggle a causal
  mask, and watch how attention weights redistribute.
- **Scaling-law planner** — set a compute budget and trade between the Chinchilla
  compute-optimal point and the "overtrained for cheap inference" regime that modern
  open models prefer.
- **RLHF over-optimization simulator** — crank the optimization pressure and watch the
  proxy reward keep rising while true quality peaks and collapses (the Goodhart curve).
- **GRPO group-advantage simulator** — sample a group of answers, score them with a
  verifier, and see how group-relative advantages reinforce or suppress each one.
- **KV-cache sizer** — compare how MHA, GQA, and MLA shrink the memory footprint at long
  context lengths.
- **Discovery Agent simulator** — assemble a retrieval pipeline stage by stage and watch
  the result quality degrade when you skip the quality filter or reranker.

## How it's built

It's deliberately low-tech and durable: a single static HTML file with vanilla
JavaScript and `<canvas>` for the visualizations. No build step, no dependencies, no
analytics. That means it works offline, loads instantly, and will still run years from
now. Theme preference and lesson progress persist via `localStorage`.

If you spot something that's out of date or want a track that goes deeper somewhere,
let me know — the frontier moves fast, and this is meant to be a living guide.
