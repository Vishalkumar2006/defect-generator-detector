# G2.1 execution notes

The run started from fresh deterministic identity initialization. At joint step
566, Windows denied one atomic `metrics.jsonl.tmp` replacement while the live log
was being inspected. This was an artifact-write failure, not a numerical,
locality, semantic-retention, or gradient-safety gate failure.

The last durable checkpoint was joint step 500. A bounded retry was added for
transient Windows sharing violations and covered by a focused unit test. Resume
restored models, optimizers, RNG state, data epoch/position, monitor counters, and
the 128-pair identity. Metric rows beyond the checkpoint were discarded and
updates 501 onward were replayed deterministically. The same configuration then
completed 2,000/2,000 updates.

The step-200 generator/discriminator parameter hashes and both optimizer hashes
matched the independently completed selected smoke checkpoint exactly. No
hyperparameter sweep, validation/test access, or `best` checkpoint selection was
performed.

