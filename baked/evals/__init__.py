"""
baked.evals — retrieval evaluation toolkit

Typical workflow:

    1. Generate labeled pairs (query → expected doc) once and save them:

        from baked.evals.synthetic import generate_pairs, openai_llm
        from baked.evals import save_pairs

        pairs = generate_pairs(docs, llm=openai_llm(), questions_per_doc=5)
        save_pairs(pairs, "evals/my_pairs.json")

    2. Run eval against your store whenever embeddings or search params change:

        from baked.evals import load_pairs, run_eval, print_report

        pairs  = load_pairs("evals/my_pairs.json")
        result = run_eval(store, pairs, top_k=5)
        print_report(result, verbose=True)

Metrics:
    Recall@k  — fraction of queries where the correct doc appears in top-k results.
    MRR       — mean reciprocal rank; rewards finding the correct doc earlier.
"""

from .harness import EvalPair, EvalResult, QueryResult, run_eval, print_report, save_pairs, load_pairs
from .synthetic import generate_pairs, groq_llm, openai_llm, anthropic_llm

__all__ = [
    "EvalPair",
    "EvalResult",
    "QueryResult",
    "run_eval",
    "print_report",
    "save_pairs",
    "load_pairs",
    "generate_pairs",
    "groq_llm",
    "openai_llm",
    "anthropic_llm",
]
