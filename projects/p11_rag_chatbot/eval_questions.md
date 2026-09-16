# RAG evaluation questions

Twelve grounded questions (answers exist in `data/`) plus one **trap** question whose
answer is deliberately absent — it must trigger the refusal "I cannot find that
information in the documents." Run: `uv run p11-rag-chatbot -q "<question>"`.
`rag_chatbot.load_eval_questions()` parses the twelve grounded rows (skipping the
header, the separator row and the trap row) into `(question, expected_source)` pairs
for `test_evaluation_questions_retrieve_the_expected_source`; the trap row has its
own test, `test_trap_question_triggers_refusal`.

| # | Question | Expected answer | Source |
|---|----------|-----------------|--------|
| 1 | How many paid vacation days do employees get? | 25 days | acme_handbook.pdf |
| 2 | Who is the CEO of ACME Robotics? | Ioana Popescu | acme_handbook.pdf |
| 3 | What is the guest WiFi password? | acme-guest-2024 | acme_handbook.pdf |
| 4 | When and where was ACME Robotics founded? | 2014, in Cluj-Napoca | acme_handbook.pdf |
| 5 | How many days per week can employees work remotely? | Up to 3 days | acme_handbook.pdf |
| 6 | Who is the health insurance provider? | Regina Maria | acme_handbook.pdf |
| 7 | By when must expense reports be submitted? | By the 5th of each month | acme_handbook.pdf |
| 8 | What framework and database does the backend use? | FastAPI + PostgreSQL 16 | engineering_notes.md |
| 9 | How many approvals are required to merge a pull request? | At least two | engineering_notes.md |
| 10 | What is the public API rate limit? | 100 requests per minute per user | engineering_notes.md |
| 11 | Where is parking available? | Underground garage, level -2 | support_faq.txt |
| 12 | When is the all-hands meeting? | Last Friday of each month | support_faq.txt |
| **T** | **What is ACME Robotics' annual revenue?** (trap) | **I cannot find that information in the documents.** | (absent) |
