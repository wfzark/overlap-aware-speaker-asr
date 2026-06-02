# Overlap-Aware Speaker-Attributed ASR with RAG-Enhanced LLM Correction

This project extends XuTong's multi-speaker conversation management system by focusing on overlap-aware speaker-attributed ASR. Instead of applying speech separation to all audio, we evaluate different overlap intensities and test whether separated speaker tracks, speaker-aware LLM correction, and glossary-based RAG improve transcription, attribution, and summary quality.

## Problem Statement

Multi-speaker audio becomes difficult to transcribe when speakers interrupt or overlap. This project studies when speech separation helps, when it may hurt, and how retrieval-enhanced LLM correction can improve terminology and speaker-attributed transcripts.

## Method Overview

The planned comparison includes:

1. Mixed audio to ASR.
2. Mixed audio to an alternative ASR baseline.
3. Separated speaker tracks to ASR to speaker-attributed transcript.
4. Speaker-aware LLM correction.
5. Glossary-RAG enhanced correction.

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
