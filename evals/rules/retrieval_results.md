# Retrieval benchmark (dev split)

35 answerable dev questions, searched with the rules agent's reformulated query (reformulate prompt da50e412); the reranker scores against the original question. 20 candidates, top 5 returned. Sorted by coverage@5, then MRR@5. Latency: mean per question on the CPU, in milliseconds.

| # | config | chunking | search | embedder | reranker | hit@5 | coverage@5 | MRR@5 | hit@20 first stage | latency ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `recursive-vector-gemma-gte` | recursive | vector | gemma | gte | 0.971 | 0.944 | 0.818 | 0.971 | 8149 |
| 2 | `structure-hybrid-gemma-gte` | structure | hybrid | gemma | gte | 0.943 | 0.943 | 0.825 | 0.971 | 9260 |
| 3 | `recursive-hybrid-gemma-gte` | recursive | hybrid | gemma | gte | 0.943 | 0.915 | 0.814 | 1.000 | 8137 |
| 4 | `structure-hybrid-granite-gte` | structure | hybrid | granite | gte | 0.914 | 0.914 | 0.793 | 0.943 | 9238 |
| 5 | `structure-vector-granite-gte` | structure | vector | granite | gte | 0.914 | 0.914 | 0.793 | 0.943 | 9337 |
| 6 | `structure-vector-gemma-gte` | structure | vector | gemma | gte | 0.914 | 0.914 | 0.789 | 0.943 | 9189 |
| 7 | `fixed-vector-granite-gte` | fixed | vector | granite | gte | 0.943 | 0.901 | 0.869 | 0.943 | 9119 |
| 8 | `fixed-vector-gemma-gte` | fixed | vector | gemma | gte | 0.914 | 0.898 | 0.839 | 0.914 | 9152 |
| 9 | `recursive-bm25-gte` | recursive | bm25 | - | gte | 0.914 | 0.887 | 0.815 | 0.943 | 8125 |
| 10 | `recursive-vector-granite-gte` | recursive | vector | granite | gte | 0.914 | 0.887 | 0.776 | 0.943 | 8105 |
| 11 | `fixed-hybrid-gemma-gte` | fixed | hybrid | gemma | gte | 0.886 | 0.870 | 0.852 | 0.943 | 9110 |
| 12 | `recursive-hybrid-granite-gte` | recursive | hybrid | granite | gte | 0.886 | 0.858 | 0.790 | 0.943 | 8104 |
| 13 | `recursive-hybrid-granite-norerank` | recursive | hybrid | granite | - | 0.886 | 0.858 | 0.696 | 0.943 | 46 |
| 14 | `fixed-hybrid-granite-gte` | fixed | hybrid | granite | gte | 0.857 | 0.841 | 0.824 | 0.914 | 9060 |
| 15 | `structure-vector-granite-minilm` | structure | vector | granite | minilm | 0.829 | 0.817 | 0.608 | 0.943 | 653 |
| 16 | `structure-hybrid-granite-norerank` | structure | hybrid | granite | - | 0.829 | 0.816 | 0.721 | 0.943 | 46 |
| 17 | `structure-vector-gemma-norerank` | structure | vector | gemma | - | 0.829 | 0.816 | 0.700 | 0.943 | 62 |
| 18 | `structure-vector-granite-norerank` | structure | vector | granite | - | 0.829 | 0.816 | 0.641 | 0.943 | 42 |
| 19 | `structure-bm25-gte` | structure | bm25 | - | gte | 0.829 | 0.812 | 0.774 | 0.857 | 9912 |
| 20 | `fixed-vector-gemma-norerank` | fixed | vector | gemma | - | 0.829 | 0.812 | 0.712 | 0.914 | 62 |
| 21 | `recursive-hybrid-gemma-norerank` | recursive | hybrid | gemma | - | 0.829 | 0.801 | 0.698 | 1.000 | 66 |
| 22 | `recursive-vector-gemma-norerank` | recursive | vector | gemma | - | 0.829 | 0.801 | 0.659 | 0.971 | 62 |
| 23 | `fixed-bm25-norerank` | fixed | bm25 | - | - | 0.829 | 0.801 | 0.704 | 0.829 | 3 |
| 24 | `recursive-bm25-norerank` | recursive | bm25 | - | - | 0.829 | 0.796 | 0.684 | 0.943 | 3 |
| 25 | `structure-vector-gemma-minilm` | structure | vector | gemma | minilm | 0.800 | 0.789 | 0.607 | 0.943 | 665 |
| 26 | `fixed-vector-granite-minilm` | fixed | vector | granite | minilm | 0.857 | 0.788 | 0.640 | 0.943 | 807 |
| 27 | `fixed-hybrid-gemma-norerank` | fixed | hybrid | gemma | - | 0.800 | 0.788 | 0.745 | 0.943 | 65 |
| 28 | `structure-bm25-norerank` | structure | bm25 | - | - | 0.800 | 0.788 | 0.698 | 0.857 | 4 |
| 29 | `structure-hybrid-gemma-norerank` | structure | hybrid | gemma | - | 0.800 | 0.788 | 0.697 | 0.971 | 66 |
| 30 | `fixed-hybrid-gemma-minilm` | fixed | hybrid | gemma | minilm | 0.829 | 0.785 | 0.610 | 0.943 | 826 |
| 31 | `fixed-bm25-gte` | fixed | bm25 | - | gte | 0.800 | 0.784 | 0.786 | 0.829 | 9184 |
| 32 | `recursive-bm25-minilm` | recursive | bm25 | - | minilm | 0.829 | 0.784 | 0.560 | 0.943 | 771 |
| 33 | `recursive-hybrid-gemma-minilm` | recursive | hybrid | gemma | minilm | 0.829 | 0.784 | 0.530 | 1.000 | 802 |
| 34 | `recursive-vector-gemma-minilm` | recursive | vector | gemma | minilm | 0.829 | 0.784 | 0.528 | 0.971 | 796 |
| 35 | `fixed-hybrid-granite-norerank` | fixed | hybrid | granite | - | 0.829 | 0.779 | 0.707 | 0.914 | 45 |
| 36 | `structure-bm25-minilm` | structure | bm25 | - | minilm | 0.800 | 0.772 | 0.615 | 0.857 | 649 |
| 37 | `fixed-hybrid-granite-minilm` | fixed | hybrid | granite | minilm | 0.800 | 0.757 | 0.619 | 0.914 | 800 |
| 38 | `fixed-vector-gemma-minilm` | fixed | vector | gemma | minilm | 0.800 | 0.757 | 0.594 | 0.914 | 818 |
| 39 | `recursive-hybrid-granite-minilm` | recursive | hybrid | granite | minilm | 0.800 | 0.755 | 0.534 | 0.943 | 780 |
| 40 | `fixed-vector-granite-norerank` | fixed | vector | granite | - | 0.800 | 0.751 | 0.641 | 0.943 | 42 |
| 41 | `fixed-bm25-minilm` | fixed | bm25 | - | minilm | 0.771 | 0.746 | 0.590 | 0.829 | 805 |
| 42 | `structure-hybrid-gemma-minilm` | structure | hybrid | gemma | minilm | 0.743 | 0.731 | 0.594 | 0.971 | 673 |
| 43 | `structure-hybrid-granite-minilm` | structure | hybrid | granite | minilm | 0.743 | 0.731 | 0.585 | 0.943 | 653 |
| 44 | `recursive-vector-granite-minilm` | recursive | vector | granite | minilm | 0.771 | 0.726 | 0.522 | 0.943 | 779 |
| 45 | `recursive-vector-granite-norerank` | recursive | vector | granite | - | 0.714 | 0.694 | 0.546 | 0.943 | 42 |
