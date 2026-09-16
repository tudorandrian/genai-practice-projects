# Evaluating LLM App Quality with Offline Test Sets

Trying a handful of questions by hand and reading the answers is a reasonable way to get
a first impression of Pebble's pipeline, but it does not scale, is not repeatable across
changes, and tends to unconsciously favor whatever questions the person testing happens
to think of. A proper offline evaluation set fixes all three problems at once.

An evaluation set for Fenwick Labs' assistant is a fixed list of realistic questions,
each paired with the manual section that should answer it and, where practical, a short
reference answer. Questions should deliberately span easy cases (a fact stated plainly
in one place), harder cases (an answer that requires combining two nearby sentences),
and at least one trap case - a question with no answer anywhere in the manuals -
specifically to check that the refusal behaviour from the previous lesson actually
triggers when it should.

Running the full pipeline over this fixed set after every meaningful change - a new
chunk size, a reworded prompt, a different embedding model - and comparing results
against the previous run turns 'does this change help or hurt' from a subjective
impression into a measured before-and-after comparison.

Retrieval and generation deserve separate metrics, exactly because they can fail
independently, as the earlier retrieval lesson noted. Retrieval accuracy asks: was the
correct source chunk actually retrieved, ideally ranked first? Answer quality asks a
different question: given that the correct chunk was retrieved, did Pebble produce a
correct, appropriately cited answer from it? Reporting only a single combined 'did it
work' number hides which of the two stages is actually responsible when something goes
wrong.

Grading whether a generated answer matches a reference answer is not always as simple as
exact string comparison, since a correct answer can be phrased several different valid
ways. For Fenwick Labs' narrow, factual questions, checking whether a small set of key
phrases from the reference answer appear in the generated one is a practical middle
ground between brittle exact matching and the added complexity of using a separate model
to judge answer quality.

An evaluation set is not something written once and left alone; every real failure a
user reports is a candidate new test case, and adding it keeps the evaluation set
growing to reflect the actual ways the pipeline gets used and occasionally fails in
practice, rather than staying frozen at whatever the team happened to think of on day
one.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why does the evaluation approach measure retrieval accuracy and answer quality as two separate metrics rather than one combined score?

- A) Retrieval and generation can fail independently, so a single combined score would hide which stage is actually responsible when something goes wrong. — **Correct answer.** This exact independence-of-failure reasoning is why the two are measured separately.
- B) Scikit-learn cannot compute a single combined metric across two stages. — **Incorrect.** This is not a tooling limitation; the reasoning is about diagnosing which stage failed.
- C) Retrieval accuracy and answer quality always produce identical scores in practice. — **Incorrect.** If they were always identical there would be no reason to measure them separately.
- D) A combined score is technically impossible to define. — **Incorrect.** A combined score could be defined; the lesson argues it would simply be less diagnostically useful.

- **Answer:** A

### Q2. (True/False)
A well-designed evaluation set should include at least one trap question with no answer in the manuals, to check that the refusal behaviour actually triggers.

- **Answer:** True
- **Hint:** Recall the earlier lesson's grounding and refusal discussion.

### Q3. (Open-Ended)
Explain why exact string comparison between a generated answer and a reference answer is often too brittle a grading method, and what practical alternative the lesson suggests.

- **Answer:** A correct answer can be phrased in several different valid ways, so requiring an exact string match would mark many genuinely correct answers as wrong; checking whether a small set of key phrases from the reference answer appear in the generated answer is a more practical middle ground.
