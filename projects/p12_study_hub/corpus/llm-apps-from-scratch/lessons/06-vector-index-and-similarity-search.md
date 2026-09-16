# Building a Vector Index and Similarity Search

Having split the manuals into chunks, the pipeline still needs a way to decide, for any
given question, which chunks are actually relevant. Comparing the question against every
chunk's raw text with keyword matching alone misses a lot: a question asking 'how do I
replace the blade' should also match a chunk that talks about 'swapping the cutting
attachment' even though it shares almost no exact words with the question.

Embeddings solve this by converting a piece of text into a fixed-length vector of
numbers such that texts with similar meaning end up as vectors that are close together
in that numeric space, regardless of whether they share the same exact words. A small,
efficient embedding model - much smaller than Pebble itself, since its only job is
producing these vectors, not generating text - converts every chunk into an embedding
once, ahead of time, and converts each incoming question into an embedding at query
time.

A vector index stores every chunk's embedding in a structure built for fast nearest-
neighbor search: given a question's embedding, it returns the chunks whose embeddings
are closest to it, typically measured with cosine similarity, without having to compare
the question against every single chunk one by one through brute force - important once
Fenwick Labs' manual collection grows into the thousands of chunks.

Retrieval is normally asked for the top-k closest chunks, where k is a small number such
as three or four rather than every remotely related chunk in the collection; retrieving
too many chunks wastes context-window budget on marginally relevant material and can
actually make Pebble's answer worse by burying the truly relevant passage among several
less relevant ones.

It is worth periodically sanity-checking retrieval on its own, separately from the
generated answer: for a fixed list of test questions with a known correct source chunk,
check whether that correct chunk actually appears, ideally ranked first, among the top-k
results. A retrieval failure - the right chunk never even being retrieved - cannot be
fixed by any amount of prompt engineering downstream, since Pebble can only answer from
what it is actually given.

This index needs to be rebuilt, or at least incrementally updated, whenever the
underlying manuals change; an index that silently goes stale after a manual is revised
will keep retrieving the old wording indefinitely, with nothing about the pipeline
itself signalling that anything is out of date.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why can a question and a relevant chunk match well through embeddings even when they share almost no exact words?

- A) Embeddings ignore the question entirely and return chunks at random. — **Incorrect.** Embeddings are specifically computed from the question's own content, not chosen randomly.
- B) Embeddings represent meaning as vectors, placing texts with similar meaning close together in that numeric space regardless of exact wording overlap. — **Correct answer.** This meaning-based closeness, independent of exact word overlap, is exactly what embeddings capture.
- C) Keyword matching is always used instead of embeddings for the final ranking. — **Incorrect.** The lesson contrasts embeddings with keyword matching precisely because embeddings replace it here.
- D) Every chunk is guaranteed to be equally relevant to every question. — **Incorrect.** Retrieval specifically aims to distinguish more relevant chunks from less relevant ones, not treat them as equal.

- **Answer:** B

### Q2. (True/False)
Retrieving a very large number of chunks for every question generally improves Pebble's final answer quality.

- **Answer:** False
- **Hint:** Consider what happens to the truly relevant passage when it is buried among many less relevant ones.

### Q3. (Open-Ended)
Explain why checking retrieval quality on its own, separately from the generated answer, matters for debugging a RAG pipeline.

- **Answer:** If the correct source chunk is never retrieved in the first place, no amount of prompt engineering can produce a correct answer, since the model can only work with what it is given, so isolating and testing the retrieval step directly reveals failures that checking only the final answer would misattribute to the model itself.
