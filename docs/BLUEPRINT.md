# Architecture Blueprint

## Goal

Build a clean successor experiment to DyGKT in which a question is represented
by both its content and its temporal graph behavior.

## Data Flow

```text
problem.json (JSONL)                  student-problem-fine.json
  content/options/concepts              user/problem/label/time
            |                                      |
            +---------- validate and join ----------+
                               |
                    chronological event table
                               |
              +----------------+----------------+
              |                                 |
       question text encoder             leak-safe split masks
              |                                 |
       static semantic matrix       temporal + unseen-question
              +----------------+----------------+
                               |
                     dual temporal histories
                student side        question side
                     |                   |
             Transformer encoder  Transformer encoder
                     +---------+---------+
                               |
                    semantic gated fusion
                               |
                         P(correct)
```

## DySemKT

For event `(student, question, time)`:

1. The student encoder consumes previous `(question semantic, response,
   elapsed time)` interactions together with explicit same-question,
   same-exercise and concept-overlap relations to the current question.
2. The question encoder consumes previous `(response, elapsed time)` events,
   representing population-level temporal behavior without student identity.
3. The current question combines frozen text features and an optional learned
   item ID embedding.
4. A learned gate fuses student state, dynamic question state and current
   semantics before binary prediction.

The student state also receives a repeat summary containing repeat count, the
last same-question result and time since that attempt. All repeat features are
computed from strictly prior, allowed history.

No current response or future event is visible to either encoder.

## Experiment Matrix

| Axis | Values |
|---|---|
| Feature mode | `id`, `semantic`, `hybrid` |
| Split | global temporal, strict unseen-question |
| Question history | enabled, ablated |
| Encoder | multilingual pretrained model, frozen |

Primary metrics are ROC-AUC, average precision, log loss and accuracy. Because
MOOCRadar is label-imbalanced, accuracy is never reported alone.

## Milestones

1. Data integrity and deterministic preprocessing.
2. Offline smoke model and end-to-end tests.
3. Pretrained multilingual embeddings and baseline runs.
4. Temporal and unseen-question ablations across seeds.
5. Optional extensions: concept graph encoder, learned text fine-tuning and
   richer question-side student-state aggregation.
