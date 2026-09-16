# Cost and Latency Trade-offs: Local vs Hosted Models

Fenwick Labs could, in principle, replace Pebble with a call to a larger, remotely
hosted model instead of running anything locally. This lesson works through that
comparison concretely, in terms of cost and latency, rather than treating 'local' or
'hosted' as an obviously correct default.

A hosted model's cost typically scales directly with usage, usually billed per token
processed and generated; a locally run model like Pebble has a fixed hardware and
electricity cost that does not grow with how many questions the kiosk answers. For a
factory floor kiosk fielding a modest, fairly predictable number of questions per day,
this means the hosted option's cost is more variable and, past a certain volume, larger,
while the local option's cost is flat and predictable regardless of how many times it is
actually used that day.

Latency has a similar shape: a hosted model call must cross the network, which adds a
delay that varies with connection quality and can spike or fail entirely if the factory
floor's connection is briefly down - a real concern given the kiosk's location. A local
call to Pebble has no such external dependency, so its latency is more consistent and,
for a well-tuned setup, typically lower for short factual answers, though a much larger
hosted model can sometimes make up for network delay with faster per-token generation on
more powerful hardware.

None of this makes local strictly better: a hosted model's larger size generally means
noticeably better answer quality on harder, more open-ended questions, and it requires
no local hardware to purchase, maintain, or eventually replace as it ages. For a task
that genuinely needs broad general knowledge or complex multi-step reasoning beyond what
a small model reliably provides, a hosted model may be worth its cost and latency trade-
offs.

Fenwick Labs' actual decision followed directly from the requirements established back
in lesson one: the privacy of unreleased-product content and the need for the kiosk to
work without a reliable connection ruled out a hosted-only approach regardless of the
cost and quality trade-offs discussed here, which is a reminder that this comparison
should follow the application's hard constraints, not the other way around.

A hybrid approach is also worth naming explicitly: some teams run a small local model
for most questions and fall back to a hosted model only for cases the local model's own
confidence, or the retrieval relevance score from earlier lessons, flags as likely to
need more capability than the small model can reliably provide - a middle ground between
the two extremes this lesson has mostly treated as a binary choice.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why did Fenwick Labs rule out a hosted-only approach for Pebble's use case, according to the lesson?

- A) Hosted models are always more expensive than local models regardless of usage. — **Incorrect.** The lesson describes cost as usage-dependent, not universally higher for hosted models.
- B) Hosted models cannot answer factual questions about product manuals. — **Incorrect.** Nothing in the lesson claims hosted models are incapable of this kind of task.
- C) Local models are always faster than hosted models for every kind of question. — **Incorrect.** The lesson notes a much larger hosted model can sometimes offset network delay with faster generation.
- D) The privacy of unreleased-product content and the need for the kiosk to keep working without a reliable connection are hard constraints that a hosted-only approach could not satisfy. — **Correct answer.** These hard constraints from lesson one are exactly what the lesson cites as ruling out a hosted-only approach.

- **Answer:** D

### Q2. (True/False)
A hosted model's per-usage cost structure means its total cost is more variable than a locally run model's roughly fixed hardware and electricity cost.

- **Answer:** True
- **Hint:** Consider how each cost structure responds to a change in daily usage volume.

### Q3. (Open-Ended)
Describe the hybrid approach the lesson mentions as a middle ground between purely local and purely hosted, and explain what might trigger falling back to the hosted model.

- **Answer:** A team can run a small local model for most questions and only call a hosted model when a signal such as low model confidence or a low retrieval relevance score suggests the question needs more capability than the small model can reliably provide.
