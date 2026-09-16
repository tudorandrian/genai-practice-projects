# Prompting Basics: Instructions, Roles, and Few-Shot Examples

A language model like Pebble does not execute a specification; it continues a piece of
text in a way that statistically resembles its training data, shaped by whatever text -
the prompt - it is given first. Prompting is the practice of writing that text so the
continuation the model produces is actually the answer you wanted, in the form you
wanted.

The most basic structure is an instruction: a direct, explicit statement of the task,
such as 'Answer the following question using only the manual excerpt below.' Vague
instructions produce vague or inconsistent answers; a small model in particular benefits
from being told exactly what to do, in what format, and what to do when it does not know
the answer, rather than being left to infer those expectations on its own.

Many small local models are 'chat' models trained to expect messages tagged with roles -
system, user, assistant - rather than one undifferentiated block of text. The system
role is where a stable, task-wide instruction belongs (Pebble's system message fixes its
job as 'answer only from the provided manual excerpt'), while the user role carries the
specific question being asked this time; mixing these up, or putting the core
instruction in the user message where it competes with the question for the model's
attention, tends to produce less reliable behaviour.

Few-shot prompting adds one or more worked examples of the task - a sample question, a
sample excerpt, and the exact answer format expected - directly in the prompt before the
real question. For Pebble, a single example showing a properly cited, concise answer
noticeably improves how consistently it cites its source and avoids padding its answer
with unnecessary caveats, compared to giving the instruction alone with zero examples.

Prompts should be treated as code: versioned, tested against a fixed set of example
inputs, and changed deliberately rather than by trial and error in a single ad hoc
session. A prompt that happens to work on the three questions you tried by hand is not
evidence it will hold up across the hundreds of different phrasings real users will type
- lesson eight in this course covers building an actual evaluation set for exactly this
reason.

A small model is also more sensitive to prompt phrasing than a much larger one tends to
be, so small wording changes - reordering the instruction and the context, or changing
'answer briefly' to 'answer in one sentence' - can shift Pebble's output more than you
might expect from experience with a larger, more capable model.

## 1. Practice Questions

### Q1. (Multiple Choice)
In a chat-style prompt with system, user, and assistant roles, where does a stable, task-wide instruction like 'answer only from the provided excerpt' generally belong?

- A) In the assistant role, written as if the model already said it. — **Incorrect.** The assistant role is for the model's own generated turns, not for instructing it in advance.
- B) It should be repeated in every user message instead of stated once. — **Incorrect.** Repeating it in every user message is unnecessary once it is fixed in the system role.
- C) In the system role, since it is a stable, task-wide instruction rather than a specific question. — **Correct answer.** The system role is exactly where a stable, task-wide instruction belongs, as the lesson states.
- D) Prompts for chat models cannot use role tags at all. — **Incorrect.** Chat models are specifically trained to expect role-tagged messages.

- **Answer:** C

### Q2. (True/False)
A prompt that works well on a handful of examples tried by hand is sufficient evidence that it will work reliably across many different real user questions.

- **Answer:** False
- **Hint:** The lesson argues prompts should be tested against a proper evaluation set, not just a few manual tries.

### Q3. (Open-Ended)
Explain why a small model like Pebble tends to benefit more from few-shot examples than a much larger, more capable model might.

- **Answer:** A smaller model has less general capability to infer an unstated expected format or style on its own, so a worked example showing exactly how an answer should be cited and phrased gives it a concrete pattern to follow, reducing the inconsistency that an instruction alone tends to leave.
