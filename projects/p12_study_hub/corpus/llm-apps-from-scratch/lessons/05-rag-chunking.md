# Retrieval-Augmented Generation: Chunking Documents

Retrieval-augmented generation, or RAG, is the pattern that solves the context-window
problem from the previous lesson: instead of asking Pebble to answer from memory, or
trying to stuff an entire manual into one prompt, the pipeline first retrieves the small
number of passages most relevant to the question and only those passages go into the
prompt.

Before anything can be retrieved, the manuals have to be split into chunks - smaller
pieces of text, each ideally covering one coherent idea, short enough to fit comfortably
in a prompt alongside a question and an instruction. Splitting purely by a fixed
character count is simple but can cut a chunk off mid-sentence or mid-procedure, right
at the point that mattered most; Fenwick Labs' pipeline instead splits primarily on
manual section boundaries, falling back to a fixed size only for unusually long
sections, so a chunk boundary rarely lands in the middle of a single instruction step.

Chunk size is a genuine trade-off, not a setting with one obviously correct value.
Chunks that are too small lose surrounding context - a chunk that only contains a
warning light's color without the paragraph explaining what causes it is barely useful
on its own - while chunks that are too large dilute relevance, since a long chunk
covering five unrelated procedures will still be retrieved whenever any one of those
five is asked about, wasting context-window budget on four irrelevant procedures for
every one that is actually needed.

A common refinement is overlap between adjacent chunks: ending one chunk a little
further into the next section's opening sentence, so that content near a chunk boundary
is not orphaned from the sentence that explains it. Fenwick Labs' pipeline uses a modest
overlap of roughly fifty tokens between adjacent chunks, small enough not to bloat the
index but large enough to keep boundary sentences legible on both sides.

Each chunk should also carry metadata alongside its text: which manual it came from,
which section, and the product model the manual describes. This metadata is what later
lets Pebble's answer cite a specific source ('Section 4.2 of the Model 30 manual')
instead of an unattributed block of text, and lets the pipeline filter retrieval to only
the relevant product's manual when a question specifies a model number.

Getting chunking right is worth real iteration, since every later stage of the RAG
pipeline - embedding, retrieval, and the final answer - depends on the chunks having
been split sensibly in the first place; a retrieval system with excellent embeddings
still cannot recover context that was destroyed by a bad chunk boundary.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why does Fenwick Labs' pipeline split manuals primarily on section boundaries rather than by a fixed character count alone?

- A) Splitting on section boundaries avoids cutting a chunk off in the middle of a single coherent instruction or idea. - **Correct answer.** This is exactly the reasoning the lesson gives for preferring section boundaries.
- B) Fixed character counts cannot be computed for manual text. - **Incorrect.** Character counts can be computed for any text; the issue is where the cut falls, not whether it can be measured.
- C) Section boundaries always produce chunks of identical length. - **Incorrect.** Sections vary in length, so boundary-based chunks are not guaranteed to be uniform in size.
- D) Pebble cannot process chunks that were split by character count. - **Incorrect.** Pebble can process any text chunk regardless of how it was split; the concern is chunk quality, not compatibility.

- **Answer:** A

### Q2. (True/False)
A chunk that is too large can dilute relevance by covering several unrelated procedures, so it gets retrieved even when only one of those procedures is actually relevant to the question.

- **Answer:** True
- **Hint:** Consider what happens when a long chunk is retrieved for a question about only one of its several topics.

### Q3. (Open-Ended)
Explain why attaching metadata such as manual section and product model to each chunk is useful beyond simply enabling citations in the final answer.

- **Answer:** It also allows the retrieval step itself to be filtered to the relevant product's manual when a question specifies a model number, narrowing the search space and reducing the chance of retrieving a similarly worded passage from a different, irrelevant manual.
