# Tokenization and Context Windows

Pebble does not read text one character or one whole word at a time; it reads tokens,
sub-word pieces produced by a tokenizer trained alongside the model. A common English
word might be a single token, while an uncommon word, a part number like 'FW-2201B', or
a typo can be split into several smaller pieces - understanding this is the first step
to understanding two very practical limits: how much text the model can consider at
once, and roughly what a request will cost if billed per token.

The context window is the maximum number of tokens the model can process in a single
request, counting the prompt and the generated answer together. Pebble's context window
is 4,096 tokens - roughly three thousand words of English text - which sounds generous
until you remember that a Fenwick Labs manual chapter can easily run to several thousand
words on its own; the whole manual cannot simply be pasted into every prompt.

This limit is the direct motivation for retrieval, covered starting in the next lesson:
instead of handing Pebble the entire manual and hoping it finds the relevant part, only
the most relevant excerpt - a few hundred to a couple of thousand tokens - is retrieved
and placed in the prompt, leaving room for the instruction, the question, and the
generated answer, all within the 4,096-token budget.

Token counts are not simply word counts, and code, part numbers, and unusual punctuation
tend to tokenize less efficiently than plain prose, sometimes using two or three tokens
for what looks like a single short word. Measuring token counts directly with the actual
tokenizer, rather than estimating from word or character counts, is the only reliable
way to know whether a given prompt will fit.

Running out of context window shows up in one of two ways depending on how the serving
code handles it: an explicit error telling you the request was too long, which is
preferable because it is impossible to miss, or a silent truncation that quietly drops
text from the end of the prompt - often the part of the manual excerpt that actually
contained the answer - producing a confidently wrong response with no visible sign
anything went missing.

Because of this, Fenwick Labs' pipeline checks the token count of every assembled prompt
before sending it to Pebble and fails loudly, with a clear message, rather than risking
a silent truncation - a small, cheap check that avoids a failure mode which would
otherwise be very difficult to notice from the outside.

## 1. Practice Questions

### Q1. (Multiple Choice)
Why is measuring token counts with the actual tokenizer more reliable than estimating from word counts?

- A) Word counts and token counts are always identical for English text. - **Incorrect.** The lesson explicitly notes uncommon words, part numbers, and punctuation can split into several tokens.
- B) Tokenizers cannot process text containing numbers or punctuation. - **Incorrect.** Tokenizers routinely process numbers and punctuation, simply sometimes with more tokens than expected.
- C) The context window is measured in words, not tokens. - **Incorrect.** The lesson defines the context window explicitly in tokens, not words.
- D) Unusual text such as part numbers or code often uses more tokens than a plain word count would suggest, so only the real tokenizer gives an accurate count. - **Correct answer.** This exact discrepancy between word counts and true token counts is why direct measurement matters.

- **Answer:** D

### Q2. (True/False)
A silent truncation of a prompt that exceeds the context window is generally worse than an explicit error, because it can drop the part of the context that contained the answer with no visible warning.

- **Answer:** True
- **Hint:** Consider which failure mode is easier to notice and debug.

### Q3. (Open-Ended)
Explain why Pebble's 4,096-token context window makes retrieving only the most relevant excerpt necessary, rather than including an entire manual chapter in every prompt.

- **Answer:** A full manual chapter can already run to several thousand words, which would consume most or all of the context window on its own, leaving little or no room for the instruction, the question, and the generated answer, so only a smaller, relevant excerpt can realistically fit alongside everything else the prompt needs.
