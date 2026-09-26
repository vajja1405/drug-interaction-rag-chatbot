# Retrieval benchmark — 2026-09-26

1011 passages from 154 FDA labels; 600 queries (300 clean, 300 with a misspelled drug). Relevance: mention-based proxy: label A passages naming B, plus label B passages naming A.

| Mode | Slice | Hit@1 | Hit@5 | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| bm25 | clean | 0.697 | 0.890 | 0.897 | 0.782 | 0.787 | 0.3 | 0.4 |
| bm25 | typo | 0.090 | 0.347 | 0.408 | 0.197 | 0.238 | 0.3 | 0.4 |
| bm25 | all | 0.393 | 0.618 | 0.653 | 0.490 | 0.513 | 0.3 | 0.4 |
| dense | clean | 0.413 | 0.727 | 0.735 | 0.542 | 0.554 | 3.12 | 4.57 |
| dense | typo | 0.343 | 0.637 | 0.627 | 0.450 | 0.456 | 3.12 | 4.57 |
| dense | all | 0.378 | 0.682 | 0.681 | 0.496 | 0.505 | 3.12 | 4.57 |
| hybrid | clean | 0.703 | 0.857 | 0.871 | 0.773 | 0.765 | 3.7 | 5.19 |
| hybrid | typo | 0.270 | 0.543 | 0.601 | 0.377 | 0.403 | 3.7 | 5.19 |
| hybrid | all | 0.487 | 0.700 | 0.736 | 0.575 | 0.584 | 3.7 | 5.19 |
| hybrid_rerank | clean | 0.633 | 0.857 | 0.840 | 0.725 | 0.717 | 548.03 | 1034.49 |
| hybrid_rerank | typo | 0.470 | 0.730 | 0.736 | 0.579 | 0.584 | 548.03 | 1034.49 |
| hybrid_rerank | all | 0.552 | 0.793 | 0.788 | 0.652 | 0.651 | 548.03 | 1034.49 |
