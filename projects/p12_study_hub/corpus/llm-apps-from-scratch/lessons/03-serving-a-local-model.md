# Serving a Local Model: Loading Weights, Latency, and Batching

Once a prompt is well designed, Pebble still has to actually run somewhere and return an
answer in a reasonable amount of time. Serving covers everything between 'a set of model
weights on disk' and 'a running process that accepts a prompt and streams back tokens':
loading the weights into memory, choosing a numeric precision, and deciding how requests
are queued and processed.

Loading a model's weights from disk into memory is itself a cost worth measuring
separately from the cost of generating an answer: for Pebble, loading takes a few
seconds the first time a serving process starts, and every request after that reuses the
already-loaded weights. This is why a serving process is kept running as a long-lived
service rather than restarted for every single question - restarting per request would
make the one-time load cost dominate the actual response time.

Precision is a lever on the trade-off between speed, memory, and answer quality: running
Pebble in a reduced numeric precision (such as 8-bit integers instead of 32-bit floating
point for its weights) roughly halves or quarters the memory it needs and often speeds
up generation, at the cost of a small, usually acceptable, drop in answer quality.
Choosing the lowest precision that does not visibly hurt Pebble's answers on a fixed set
of test questions is a reasonable way to make this trade-off concretely rather than
guessing.

Generation itself proceeds one token at a time: the model produces a probability
distribution over the next possible token given everything so far, a token is chosen,
appended to the running text, and the process repeats. This is why response time scales
with how long an answer is, not just with the length of the question, and why Pebble's
prompts are written to ask for concise answers rather than open-ended essays whenever a
short answer will do.

Batching - processing several independent requests' next-token steps together - can
improve overall throughput when multiple support engineers query Pebble at the same
time, since the same hardware pass can compute the next token for several prompts at
once instead of one after another. The trade-off is that a request arriving just after a
batch has already started must wait for the next batch, slightly increasing that
individual request's latency in exchange for better throughput across all requests
together.

For Fenwick Labs' factory floor kiosk, where usually only one person is asking a
question at a time, this batching trade-off favors low latency for the single active
request over throughput for many simultaneous ones - a reminder that serving decisions
should follow from how the application is actually used, not from a generic 'best
practice' applied without considering the deployment.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is Pebble kept running as a long-lived serving process instead of being reloaded from disk for every single question?

- A) Loading the weights from disk is a measurable one-time cost, and restarting per request would make that cost dominate every response's latency. - **Correct answer.** This is exactly the reasoning the lesson gives for keeping the process long-lived.
- B) A model can only be loaded from disk successfully one time ever. - **Incorrect.** Weights can be loaded repeatedly; the concern is the wasted time from doing so unnecessarily.
- C) Restarting the process would delete the model weights permanently. - **Incorrect.** Restarting a serving process does not delete the weight files on disk.
- D) Long-lived processes are required by every operating system. - **Incorrect.** This is an application design choice, not an operating system requirement.

- **Answer:** A

### Q2. (True/False)
Response time for a language model scales with how long the generated answer is, since tokens are produced one at a time.

- **Answer:** True
- **Hint:** Recall how the lesson describes the token-by-token generation process.

### Q3. (Open-Ended)
Explain why batching multiple requests together might not be the right choice for Fenwick Labs' factory floor kiosk, even though it can improve overall throughput.

- **Answer:** The kiosk usually serves one person at a time, so there is little throughput benefit to gain from batching, while a request arriving just after a batch starts would still have to wait for the next batch, adding latency for the one user who is actually waiting.
