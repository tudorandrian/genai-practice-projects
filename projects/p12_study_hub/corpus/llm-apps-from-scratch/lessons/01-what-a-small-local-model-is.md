# What a Small Local Language Model Is, and Why Run One Locally

Fenwick Labs makes a line of workshop tools and ships a few hundred pages of product
manuals with them. Support engineers kept fielding the same handful of questions - how
to replace a part, what a warning light means, which cable an older model uses - and the
team decided to build a small assistant, nicknamed Pebble, that could answer those
questions by reading the manuals directly rather than by search alone.

'Small' here means a language model with roughly one to three billion parameters,
compared with the much larger models that only run economically on remote, specialized
hardware. A model this size can run entirely on a single ordinary machine - no network
call to an outside service required - which matters for Fenwick Labs for three reasons:
the manuals sometimes include unreleased-product drafts that should never leave the
building, a factory floor kiosk needs to keep answering questions even if its internet
connection drops, and a locally hosted model has no per-request charge that scales with
usage.

The trade-off for running locally and small is capability: Pebble knows far less general
world knowledge than a much larger, remotely hosted model, and it reasons less reliably
on tasks that need many steps of multi-step logic. This course's central argument is
that for a narrow, well-scoped job - answering questions from a fixed set of documents -
a small local model wired up carefully can perform surprisingly well, because it is not
being asked to know everything, only to read and summarize what is put in front of it.

That 'wiring up carefully' is most of what separates a genuinely useful small-model
application from a disappointing one. Across this course we will build Pebble's
supporting pipeline piece by piece: how to prompt it effectively, how to serve it with
acceptable latency, how tokenization and context limits constrain what it can read at
once, how to retrieve the right passage from the manuals before asking it anything, how
to evaluate whether it is actually working, what it costs compared to a larger hosted
alternative, and how to keep it from answering questions it has no business answering.

A recurring theme will be that a small model's weaknesses are often best addressed
outside the model itself - through better retrieval, a tighter prompt, or an explicit
refusal rule - rather than by hoping a bigger model would simply not have the problem.
Those outside-the-model techniques tend to transfer directly to larger models too, so
the habits built here are not just a workaround for Pebble's size.

By the end of this lesson you should be able to state, in your own words, the concrete
trade-off between running a small model locally and calling a larger hosted model over
the network, in terms of privacy, reliability, cost, and raw capability - a trade-off
every one of the following nine lessons will touch in some way.

## 1. Practice Questions

### Q1. (Multiple Choice)
What is the central argument this course makes about small local language models like Pebble?

- A) Small models are always superior to larger hosted models for every task. — **Incorrect.** The lesson explicitly states small models know less and reason less reliably on complex tasks.
- B) For a narrow, well-scoped job such as answering questions from a fixed set of documents, a carefully wired-up small local model can perform surprisingly well. — **Correct answer.** This is exactly the course's stated central argument about scoped tasks.
- C) Local models are only useful for tasks that require no internet connection at all. — **Incorrect.** Offline operation is one benefit mentioned, not the sole justification given.
- D) A small model's weaknesses can only be fixed by making the model itself larger. — **Incorrect.** The lesson argues the opposite: many weaknesses are better addressed outside the model.

- **Answer:** B

### Q2. (True/False)
One reason Fenwick Labs chose to run Pebble locally is that some manual content should never leave the building.

- **Answer:** True
- **Hint:** The lesson lists privacy of unreleased-product drafts as one of three reasons.

### Q3. (Open-Ended)
Name the three reasons the lesson gives for running Pebble locally instead of calling a larger hosted model over the network.

- **Answer:** Privacy of sensitive unreleased-product content, continued operation on a factory floor kiosk even without a reliable internet connection, and avoiding a per-request cost that scales with usage.
