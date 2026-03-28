#!/usr/bin/env python3

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.vector_store import get_vector_store


@dataclass
class EvalQuery:
    question: str
    expected_pdf: str
    expected_page: int | None
    keywords: list[str]


EVAL_DATASET: list[EvalQuery] = [
    EvalQuery(
        question="What is the core methodology of the DREAM paper?",
        expected_pdf="DREAM",
        expected_page=None,
        keywords=["methodology", "approach", "framework"],
    ),
    EvalQuery(
        question="How does the model handle multimodal inputs?",
        expected_pdf="",
        expected_page=None,
        keywords=["multimodal", "input", "fusion"],
    ),
    EvalQuery(
        question="What are the experimental results on ImageNet?",
        expected_pdf="",
        expected_page=None,
        keywords=["ImageNet", "accuracy", "results"],
    ),
]


def evaluate_retrieval(
    queries: list[EvalQuery],
    top_k: int = 5,
) -> dict[str, Any]:
    store = get_vector_store()

    results = []
    hit_count = 0
    keyword_match_count = 0

    for query in queries:
        search_results = store.similarity_search(
            query.question,
            k=top_k,
        )

        pdf_hit = False
        page_hit = False
        keyword_hits = 0

        for result in search_results:
            payload = result.get("payload", {})
            metadata = payload.get("metadata", {})

            if query.expected_pdf:
                pdf_name = metadata.get("pdf_name", "")
                if query.expected_pdf.lower() in pdf_name.lower():
                    pdf_hit = True

            if query.expected_page is not None:
                page_idx = metadata.get("page_idx")
                if page_idx == query.expected_page:
                    page_hit = True

            content = payload.get("page_content", "")
            for keyword in query.keywords:
                if keyword.lower() in content.lower():
                    keyword_hits += 1

        if pdf_hit or page_hit:
            hit_count += 1

        keyword_match_count += keyword_hits / max(len(query.keywords), 1)

        results.append(
            {
                "question": query.question,
                "pdf_hit": pdf_hit,
                "page_hit": page_hit,
                "keyword_hits": keyword_hits,
                "result_count": len(search_results),
            }
        )

    total = len(queries)
    metrics = {
        "mode": "hybrid-only",
        "total_queries": total,
        "top_k": top_k,
        "pdf_hit_rate": hit_count / total if total > 0 else 0.0,
        "keyword_match_rate": keyword_match_count / total if total > 0 else 0.0,
        "results": results,
    }

    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"Retrieve quality evaluation results - {metrics['mode']}")
    print(f"{'=' * 60}")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Top-K: {metrics['top_k']}")
    print(f"PDF hit rate: {metrics['pdf_hit_rate']:.2%}")
    print(f"Keyword matching rate: {metrics['keyword_match_rate']:.2%}")
    print("\nDetailed results:")
    for result in metrics["results"]:
        print(f"  - {result['question'][:50]}...")
        print(f"    PDFhit: {result['pdf_hit']}, Keyword hits: {result['keyword_hits']}")


def main() -> None:
    print("ScholarRAG Retrieval quality offline evaluation")
    print("=" * 60)

    store = get_vector_store()
    try:
        all_papers = store.get_all_papers()
        if not all_papers:
            print("\nwarn: There is no paper data in the vector library")
            print("Please run first: python main.py add <path/to/paper.pdf>")
            return
        print(f"\nCommonly found in the vector library {len(all_papers)} indivual chunks")
    except Exception as e:
        print(f"\nmistake: Unable to access vector library - {e}")
        return

    print("\n[1/1] Evaluating Hybrid-Only Mode...")
    hybrid_metrics = evaluate_retrieval(EVAL_DATASET)
    print_metrics(hybrid_metrics)

    output_file = Path(__file__).parent / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hybrid_only": hybrid_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResults have been saved to: {output_file}")


if __name__ == "__main__":
    main()
