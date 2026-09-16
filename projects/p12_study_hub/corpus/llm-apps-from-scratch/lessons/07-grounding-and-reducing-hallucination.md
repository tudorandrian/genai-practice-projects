# Grounding Answers and Reducing Hallucination

A language model will produce a fluent, confident-sounding answer even when it has no
real basis for it - a failure mode usually called hallucination. For Fenwick Labs, an
invented answer about which cable a discontinued model used, delivered in exactly the
same confident tone as a correct answer, is worse than no answer at all, since a support
engineer has no obvious way to tell the two apart just by reading the response.

Grounding is the practice of constraining a model's answer to only the retrieved
context, and instructing it explicitly to say so when that context does not contain an
answer. Pebble's system prompt states the rule directly: answer using only the manual
excerpt provided, and if the excerpt does not contain the answer, reply with an exact,
fixed refusal sentence rather than guessing from general knowledge.

A fixed, exact refusal string is deliberately chosen over letting the model phrase its
own 'I don't know' in whatever words it likes: a consistent refusal is trivial for the
downstream application to detect programmatically - for instance, to route the question
to a human support engineer automatically - whereas a model's own free-form uncertainty
('I'm not entirely sure, but...') can vary in phrasing and is much harder to detect
reliably.

Even with a grounding instruction in place, a small model can still occasionally ignore
it and answer from general knowledge instead of admitting the context was insufficient -
this is exactly why the relevance of retrieval itself, covered in the previous lesson,
matters as a second line of defense: if the top retrieved chunk scores far below what a
genuinely relevant chunk usually scores, the pipeline can refuse before even calling
Pebble, catching a class of off-topic questions the model's own grounding instruction
might not catch reliably on its own.

Citing the source chunk alongside every answer is a second, complementary safeguard:
even when Pebble's answer is grounded, showing 'Section 4.2, Model 30 manual' next to it
lets a support engineer quickly verify the claim against the actual manual before
relaying it to a customer, rather than trusting the assistant's output blindly.

None of these techniques make hallucination impossible - a grounded, well-cited answer
can still occasionally misread or slightly misstate what the retrieved excerpt actually
says. The goal is to make hallucination rare and, more importantly, easy to catch: an
answer with a visible, checkable citation is something a human can verify quickly, which
is a much stronger property than an answer that merely sounds trustworthy.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why does Pebble's grounding prompt specify an exact, fixed refusal sentence rather than letting the model phrase its own uncertainty freely?

- A) A fixed sentence uses fewer tokens than any other possible reply. - **Incorrect.** Token count is not the reason given; some free-form replies could be just as short.
- B) Pebble is incapable of generating any text other than a fixed sentence. - **Incorrect.** Pebble can generate free-form text; the fixed refusal is a deliberate instruction, not a technical limitation.
- C) A consistent, exact refusal is trivial for the application to detect programmatically, unlike a model's own varying free-form expressions of uncertainty. - **Correct answer.** This reliable programmatic detection is exactly the reason given for using a fixed refusal string.
- D) Fixed refusal strings eliminate the possibility of retrieval ever failing. - **Incorrect.** The refusal string addresses generation behaviour, not retrieval failures, which are handled separately by the relevance check.

- **Answer:** C

### Q2. (True/False)
Citing the specific source chunk alongside an answer lets a human quickly verify the claim rather than having to trust the assistant's output on faith.

- **Answer:** True
- **Hint:** Consider what a support engineer can do with a citation that they cannot do with an uncited claim.

### Q3. (Open-Ended)
Explain why checking the relevance score of the top retrieved chunk, and refusing before ever calling Pebble if it is too low, acts as a second line of defense against hallucination.

- **Answer:** It catches off-topic questions at the retrieval stage itself, before the model has a chance to ignore its grounding instruction and answer confidently from general knowledge, so the pipeline does not rely solely on the model reliably following its own prompt.
