from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .evaluate_cer import list_verified_cases, load_reference, normalize_text


CSV_COLUMNS = [
    "case_id",
    "query_text",
    "retrieved_context",
    "similarity_score",
    "retrieval_method",
]


def simple_text_similarity(text1: str, text2: str) -> float:
    """
    Compute simple character-level similarity between two texts.
    
    For production, use embedding-based similarity (e.g., sentence-transformers).
    """
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)
    
    if not norm1 or not norm2:
        return 0.0
    
    # Jaccard similarity on character n-grams
    def get_ngrams(text: str, n: int = 3) -> set[str]:
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    ngrams1 = get_ngrams(norm1)
    ngrams2 = get_ngrams(norm2)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    return round(intersection / union, 4) if union > 0 else 0.0


def build_reference_knowledge_base() -> dict[str, list[dict[str, Any]]]:
    """
    Build a knowledge base from verified references.
    
    Returns:
        Dict mapping case_id to list of reference segments
    """
    cases = list_verified_cases()
    kb = {}
    
    for case_id in cases:
        try:
            reference = load_reference(case_id)
            segments = reference.get("segments", [])
            kb[case_id] = [
                {
                    "speaker": seg.get("speaker", "UNKNOWN"),
                    "text": seg.get("text", ""),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                }
                for seg in segments
            ]
        except Exception:
            kb[case_id] = []
    
    return kb


def retrieve_relevant_segments(
    query_text: str,
    knowledge_base: dict[str, list[dict[str, Any]]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k most relevant segments from knowledge base.
    
    Args:
        query_text: ASR output text to match
        knowledge_base: Reference segments KB
        top_k: Number of segments to retrieve
    
    Returns:
        List of retrieved segments with similarity scores
    """
    candidates = []
    
    for case_id, segments in knowledge_base.items():
        for seg in segments:
            seg_text = seg["text"]
            similarity = simple_text_similarity(query_text, seg_text)
            candidates.append({
                "case_id": case_id,
                "segment": seg,
                "similarity": similarity,
            })
    
    # Sort by similarity and return top-k
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    return candidates[:top_k]


def format_rag_context(retrieved_segments: list[dict[str, Any]]) -> str:
    """Format retrieved segments as LLM context."""
    lines = []
    for item in retrieved_segments:
        seg = item["segment"]
        similarity = item["similarity"]
        lines.append(f"[{seg['speaker']}] {seg['text']} (similarity: {similarity:.2f})")
    return "\n".join(lines)


def demo_rag_retrieval() -> list[dict[str, Any]]:
    """
    Demo RAG retrieval for each verified case.
    
    Returns:
        List of retrieval results
    """
    kb = build_reference_knowledge_base()
    cases = list_verified_cases()
    results = []
    
    for case_id in cases:
        try:
            reference = load_reference(case_id)
            # 使用第一个 segment 作为查询
            segments = reference.get("segments", [])
            if not segments:
                continue
            
            query_seg = segments[0]
            query_text = query_seg.get("text", "")
            
            retrieved = retrieve_relevant_segments(query_text, kb, top_k=3)
            context = format_rag_context(retrieved)
            
            avg_similarity = sum(item["similarity"] for item in retrieved) / len(retrieved) if retrieved else 0.0
            
            results.append({
                "case_id": case_id,
                "query_text": query_text[:50] + "..." if len(query_text) > 50 else query_text,
                "retrieved_context": context[:100] + "..." if len(context) > 100 else context,
                "similarity_score": round(avg_similarity, 4),
                "retrieval_method": "simple_text_similarity",
            })
        
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    return results


def render_summary(rows: list[dict[str, Any]]) -> Path:
    output_path = PROJECT_ROOT / "results" / "figures" / "rag_retrieval_demo.md"
    
    avg_similarity = sum(row["similarity_score"] for row in rows) / len(rows) if rows else 0.0
    
    lines = [
        "# RAG Retrieval Demo",
        "",
        "This demo shows RAG-based reference retrieval for ASR repair.",
        "",
        "## Overview",
        "",
        f"- Total cases: {len(rows)}",
        f"- Average similarity: {avg_similarity:.4f}",
        f"- Retrieval method: simple character n-gram similarity",
        "",
        "## Retrieval Examples",
        "",
        "| case_id | query_text | similarity_score | retrieval_method |",
        "| --- | --- | ---: | --- |",
    ]
    
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['query_text']} | {row['similarity_score']:.4f} | {row['retrieval_method']} |"
        )
    
    lines.extend([
        "",
        "## Improvement Directions",
        "",
        "- Replace simple text similarity with embedding-based retrieval (e.g., sentence-transformers)",
        "- Use vector database (FAISS, Chroma) for efficient large-scale retrieval",
        "- Add speaker-aware retrieval to preserve attribution",
        "- Fine-tune retrieval model on domain-specific data",
        "",
        "## Integration with LLM Repair",
        "",
        "Retrieved context can be provided to LLM as additional hints:",
        "",
        "```",
        "User: Please correct this ASR output.",
        "System: Here are some verified reference examples:",
        "[Retrieved context]",
        "```",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    results = demo_rag_retrieval()
    
    csv_path = PROJECT_ROOT / "results" / "tables" / "rag_retrieval_demo.csv"
    json_path = PROJECT_ROOT / "results" / "tables" / "rag_retrieval_demo.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    
    summary_path = render_summary(results)
    
    print(f"Wrote RAG retrieval demo: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote RAG retrieval JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote summary: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
