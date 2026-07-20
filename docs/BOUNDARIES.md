# Project Boundaries

## In Scope

- Binary next-response correctness prediction.
- Continuous-time student-question interactions.
- Real question text and concept semantics from MOOCRadar.
- Dynamic histories on both graph sides: a student's previous questions and a
  question's previous responses.
- Temporal generalization and strict unseen-question evaluation.
- Reproducible preprocessing, ablations, training, evaluation and tests.

## Out of Scope

- Reproducing every baseline from the original DyGKT repository.
- Generating or recovering missing question text.
- Using answer keys, future labels or full-dataset difficulty as input.
- Student profiling beyond observed interaction history.
- Serving, recommendation policies, causal claims or production deployment.
- Treating course IDs or short concept labels as question-text substitutes.

## Research Claims This Project Can Support

The project can test whether real text semantics improve response prediction
and unseen-question transfer under a fixed temporal-history architecture. It
cannot by itself establish causal learning gains or pedagogical effectiveness.

The `hash` encoder exists for offline tests only. Semantic claims must use a
documented pretrained text encoder and compare `id`, `semantic`, and `hybrid`
modes under identical splits.

