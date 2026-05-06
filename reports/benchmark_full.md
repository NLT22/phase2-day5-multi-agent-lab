# Benchmark Report

_Generated: 2026-05-06 17:41:48_

## Metrics Summary

| Run | Latency (s) | Cost (USD) | Quality /10 | Words | Tokens in/out | Citations | Critic | Status | Rewrites | Issues |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|---:|
| baseline_q1 | 13.41 | $0.00071 | 7.6 | 559 | 1754/741 | 1.00 | not_run | running | 0 | 0 |
| multi_agent_q1 | 55.09 | $0.00204 | 8.7 | 575 | 6203/1858 | 1.00 | warn | warned | 0 | 6 |
| baseline_q2 | 21.14 | $0.00058 | 9.3 | 543 | 1174/668 | 1.00 | not_run | running | 0 | 0 |
| multi_agent_q2 | 76.10 | $0.00264 | 9.4 | 520 | 7700/2479 | 0.80 | warn | max_iterations | 0 | 8 |
| baseline_q3 | 15.82 | $0.00057 | 8.2 | 507 | 1269/633 | 0.60 | not_run | running | 0 | 0 |
| multi_agent_q3 | 77.10 | $0.00282 | 9.7 | 559 | 8450/2584 | 1.00 | warn | max_iterations | 1 | 0 |
| baseline_q4 | 16.07 | $0.00077 | 8.4 | 569 | 2047/765 | 1.00 | not_run | running | 0 | 0 |
| multi_agent_q4 | 85.07 | $0.00316 | 9.5 | 569 | 10644/2614 | 1.00 | warn | max_iterations | 0 | 5 |
| baseline_q5 | 15.05 | $0.00058 | 8.6 | 534 | 1019/705 | 0.80 | not_run | running | 0 | 0 |
| multi_agent_q5 | 55.04 | $0.00159 | 9.3 | 564 | 4522/1515 | 0.80 | warn | warned | 0 | 0 |
| baseline_q6 | 10.74 | $0.00060 | 8.4 | 535 | 1247/691 | 0.80 | not_run | running | 0 | 0 |
| multi_agent_q6 | 71.13 | $0.00266 | 9.1 | 463 | 7917/2451 | 0.80 | warn | max_iterations | 0 | 9 |
| baseline_q7 | 15.88 | $0.00083 | 9.0 | 541 | 2615/729 | 1.00 | not_run | running | 0 | 0 |
| multi_agent_q7 | 67.23 | $0.00212 | 9.4 | 536 | 7719/1599 | 1.00 | warn | warned | 0 | 2 |

## Quality Score Breakdown (max 2 each)

| Run | Length | Structure | Citations | Relevance | Completeness | **Total** |
|---|---:|---:|---:|---:|---:|---:|
| baseline_q1 | 2.0 | 2.0 | 2.0 | 0.3 | 1.3 | **7.6** |
| multi_agent_q1 | 2.0 | 2.0 | 2.0 | 0.7 | 2.0 | **8.7** |
| baseline_q2 | 2.0 | 2.0 | 2.0 | 2.0 | 1.3 | **9.3** |
| multi_agent_q2 | 2.0 | 2.0 | 1.6 | 1.8 | 2.0 | **9.4** |
| baseline_q3 | 2.0 | 2.0 | 1.2 | 1.7 | 1.3 | **8.2** |
| multi_agent_q3 | 2.0 | 2.0 | 2.0 | 1.7 | 2.0 | **9.7** |
| baseline_q4 | 2.0 | 1.3 | 2.0 | 1.8 | 1.3 | **8.4** |
| multi_agent_q4 | 2.0 | 2.0 | 2.0 | 1.5 | 2.0 | **9.5** |
| baseline_q5 | 2.0 | 2.0 | 1.6 | 1.7 | 1.3 | **8.6** |
| multi_agent_q5 | 2.0 | 2.0 | 1.6 | 1.7 | 2.0 | **9.3** |
| baseline_q6 | 2.0 | 2.0 | 1.6 | 1.5 | 1.3 | **8.4** |
| multi_agent_q6 | 2.0 | 2.0 | 1.6 | 1.5 | 2.0 | **9.1** |
| baseline_q7 | 2.0 | 2.0 | 2.0 | 1.7 | 1.3 | **9.0** |
| multi_agent_q7 | 2.0 | 2.0 | 2.0 | 1.4 | 2.0 | **9.4** |

> Length: 0-2 | Structure: 0-2 | Citations: 0-2 | Relevance: 0-2 | Completeness: 0-2

## Baseline vs Multi-Agent (averages)

| Metric | Baseline | Multi-Agent | Winner |
|---|---:|---:|:---:|
| Latency (s) | 15.45 | 69.54 | baseline |
| Cost (USD) | 0.000661 | 0.002433 | baseline |
| Quality /10 | 8.50 | 9.30 | multi-agent |

## Agent Routes

- **baseline_q1**: (single agent)
- **multi_agent_q1**: researcher → analyst → writer → critic → done
- **baseline_q2**: (single agent)
- **multi_agent_q2**: researcher → analyst → writer → critic → writer → critic → done
- **baseline_q3**: (single agent)
- **multi_agent_q3**: researcher → analyst → writer → critic → writer → critic → done
- **baseline_q4**: (single agent)
- **multi_agent_q4**: researcher → analyst → writer → critic → writer → critic → done
- **baseline_q5**: (single agent)
- **multi_agent_q5**: researcher → analyst → writer → critic → done
- **baseline_q6**: (single agent)
- **multi_agent_q6**: researcher → analyst → writer → critic → writer → critic → done
- **baseline_q7**: (single agent)
- **multi_agent_q7**: researcher → analyst → writer → critic → done

## Failure Analysis

### multi_agent_q3
- CriticAgent: verdict=fail — The article contains unsupported claims, citation inaccuracies, and risks of hallucination, indicating a need for significant revisions.
