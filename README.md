# Overlap-Aware Speaker-Attributed ASR with Adaptive Routing

This project extends XuTong's multi-speaker conversation management system by focusing on overlap-aware speaker-attributed ASR. Instead of applying speech separation to all audio, we evaluate different overlap intensities and test whether separated speaker tracks improve transcription quality selectively.

## Problem Statement

Multi-speaker audio becomes difficult to transcribe when speakers interrupt or overlap. This project studies when speech separation helps, when it may hurt, and how retrieval-enhanced LLM correction can improve terminology and speaker-attributed transcripts.

## Method Overview

The planned comparison includes:

1. Mixed audio to ASR.
2. Mixed audio to an alternative ASR baseline.
3. Separated speaker tracks to ASR to speaker-attributed transcript.
4. Duplicate suppression post-processing for separated transcripts.
5. Adaptive/oracle routing over mixed, separated, and cleaned separated outputs.

## Repository Structure

- `configs/`: project configuration.
- `resources/`: migrated audio and glossary resources.
- `references/`: manual references for evaluation.
- `src/`: experiment and utility code.
- `results/`: generated transcripts, tables, summaries, and figures.
- `demo/`: future Streamlit demo.
- `docs/`: project notes, contribution docs, and video script.

## Current Stage

Stage 1 only initializes the repository, migrates audio resources, creates configuration, and generates an audio manifest. No heavy models are run in this stage.

## Current Research Focus

The project currently focuses on:

1. Adaptive routing between mixed ASR, separated ASR, and cleaned separated ASR.
2. Error type analysis for repeated hallucinations and insertion errors.
3. Speaker-aware evaluation for speaker-attributed ASR.

LLM/RAG is treated as an optional future extension, not the core experimental line at present.

## Core Claim

Speech separation is useful, but not universally beneficial. It improves CER for clean or heavily overlapping cases, while it can hurt ASR under lighter overlap because of repeated hallucinations and insertion errors. The project therefore emphasizes when to separate, how to detect failure modes, and how to evaluate speaker-attributed output more carefully.
