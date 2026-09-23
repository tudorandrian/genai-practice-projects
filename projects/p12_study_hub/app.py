"""app.py - Study Hub Assistant (capstone) entry point.

Serves the three modules - ``quiz_engine`` (quiz), ``progress`` (dashboard),
``tutor`` (RAG) - through one Gradio interface with a tab each, started with a
single command. Every module also runs headless in CLI mode. The modules are
coupled only through files (``history.json``, ``output/``, ``index/``), never
by calling into each other's UI.

Run
    uv run p12-study-hub                              # Gradio UI (3 tabs), http://127.0.0.1:7860
    uv run p12-study-hub --quiz --n 5 --seed 42        # quiz in the console
    uv run p12-study-hub --dashboard                   # progress scan + chart
    uv run p12-study-hub --ask "what is RAG?"           # RAG tutor
    uv run p12-study-hub --reindex                      # rebuild the lesson index
    uv run p12-study-hub --demo                         # offline demo (stub provider)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from projects.p12_study_hub import progress, quiz_engine, tutor
from shared.demo import DemoResult

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "corpus"
OUT_DIR = HERE / "output"

DEMO_QUESTION = "What does the missing-value lesson recommend for a skewed numeric column?"
# demo()'s synthetic 5/5 session goes here, never into output/history.json: that
# file is the user's own quiz record and feeds progress.success_rate().
DEMO_HISTORY_PATH = OUT_DIR / "demo-history.json"


# =============================================================================
# UI (Gradio, three tabs) - thin layer over the tested module functions
# =============================================================================


def _module_choices() -> list[str]:
    try:
        if not quiz_engine.BANK_PATH.exists():
            quiz_engine.build_bank_from_corpus(CORPUS_DIR, quiz_engine.BANK_PATH)
        return ["(all)"] + sorted(
            {q["module"] for q in quiz_engine.load_bank(quiz_engine.BANK_PATH)}
        )
    except Exception:  # the UI must still render if the bank isn't ready yet
        return ["(all)"]


def _tab_quiz(gr: Any) -> None:
    """One-question-at-a-time quiz flow using gr.State."""
    state = gr.State({})
    gr.Markdown("### Quiz - closed questions from the corpus question bank")
    with gr.Row():
        n_in = gr.Slider(1, 15, value=5, step=1, label="Number of questions")
        seed_in = gr.Number(value=42, label="Seed (reproducible)", precision=0)
        module_in = gr.Dropdown(_module_choices(), value="(all)", label="Course")
        type_in = gr.Dropdown(["(all)", *quiz_engine.CLOSED_TYPES], value="(all)", label="Type")
    start_btn = gr.Button("Start quiz", variant="primary")
    question_md = gr.Markdown()
    options_radio = gr.Radio(choices=[], label="Choose an answer", visible=False)
    submit_btn = gr.Button("Submit", visible=False)
    feedback_md = gr.Markdown()

    def _render(session: list[Any], idx: int) -> tuple[str, list[str]]:
        q = session[idx]
        text = f"**[{idx + 1}/{len(session)}]** _({q['module']} - {q['type']})_\n\n{q['prompt']}"
        choices = [f"{o['id']}) {o['text']}" for o in q["shuffled_options"]]
        return text, choices

    def start(n: float, seed: float, module: str, qtype: str) -> tuple[Any, ...]:
        if not quiz_engine.BANK_PATH.exists():
            quiz_engine.build_bank_from_corpus(CORPUS_DIR, quiz_engine.BANK_PATH)
        questions = quiz_engine.load_bank(quiz_engine.BANK_PATH)
        session = quiz_engine.build_session(
            questions,
            n=int(n),
            seed=int(seed),
            module=None if module == "(all)" else module,
            qtype=None if qtype == "(all)" else qtype,
        )
        text, choices = _render(session, 0)
        st = {"session": session, "idx": 0, "answers": []}
        return (
            st,
            text,
            gr.update(choices=choices, value=None, visible=True),
            gr.update(visible=True),
            "",
        )

    def answer(st: dict[str, Any], chosen: str) -> tuple[Any, ...]:
        session, idx = st["session"], st["idx"]
        q = session[idx]
        chosen_id = (chosen or "").split(")")[0].strip()
        is_correct = quiz_engine.grade(q, chosen_id)
        st["answers"].append(chosen_id)
        feedback = (
            "Correct!"
            if is_correct
            else f"Incorrect. Correct: **{', '.join(q['correct'])}**. {q.get('explanation', '')}"
        )
        if idx + 1 < len(session):
            st["idx"] = idx + 1
            text, choices = _render(session, idx + 1)
            return (
                st,
                text,
                gr.update(choices=choices, value=None, visible=True),
                gr.update(visible=True),
                feedback,
            )
        summary = quiz_engine._session_summary(session, st["answers"])  # same-package helper
        quiz_engine.save_session(summary)
        final = (
            f"{feedback}\n\n---\n"
            f"### Final score: **{summary['score']}/{summary['total']}** (saved to history)"
        )
        return st, "**Quiz complete.**", gr.update(visible=False), gr.update(visible=False), final

    start_btn.click(
        start,
        [n_in, seed_in, module_in, type_in],
        [state, question_md, options_radio, submit_btn, feedback_md],
    )
    submit_btn.click(
        answer, [state, options_radio], [state, question_md, options_radio, submit_btn, feedback_md]
    )


def _tab_progress(gr: Any) -> None:
    gr.Markdown("### Progress - status tables from corpus/ + quiz success rate")
    refresh = gr.Button("Scan & refresh", variant="primary")
    table = gr.Dataframe(label="Status by course")
    chart = gr.Image(label="progress.png", type="filepath")
    rate_md = gr.Markdown()

    def scan() -> tuple[Any, str, str]:
        df = progress.scan_statuses(CORPUS_DIR)
        pivot = progress.aggregate(df)
        chart_path = progress.plot_progress(pivot)
        rate = progress.success_rate()
        table_df = pivot.reset_index() if not pivot.empty else pivot
        if rate["by_module"]:
            md = (
                f"**Quiz success rate** ({rate['sessions']} sessions): "
                f"by course {rate['by_module']}; by type {rate['by_type']}"
            )
        else:
            md = "_No quiz sessions yet - play a quiz in the first tab._"
        return table_df, str(chart_path), md

    refresh.click(scan, None, [table, chart, rate_md])


def _ensure_lesson_index() -> None:
    """Build the lesson index if it has never been built, and bring it up to date
    when a lesson was added, edited or removed since (incremental: only the changed
    files are re-embedded). Without this, a fresh clone's first question runs
    against an empty store, and an edited corpus keeps answering from stale text
    until an explicit --reindex."""
    if tutor.index_is_stale():
        tutor.index_lessons()


def _tab_tutor(gr: Any) -> None:
    gr.Markdown("### Tutor - ask the course lessons (answers ONLY from them, with sources)")

    def respond(message: str, _history: Any) -> str:
        _ensure_lesson_index()
        result = tutor.ask(message)
        sources = f"\n\n_Sources: {', '.join(result['sources'])}_" if result["sources"] else ""
        return str(result["answer"]) + sources

    gr.ChatInterface(fn=respond, analytics_enabled=False)


def build_ui() -> Any:
    """Assemble the three-tab Gradio app."""
    import gradio as gr  # lazy import, see module docstring

    with gr.Blocks(title="Study Hub Assistant", analytics_enabled=False) as demo_app:
        gr.Markdown("# Study Hub Assistant - capstone (Quiz - Progress - Tutor)")
        with gr.Tab("Quiz"):
            _tab_quiz(gr)
        with gr.Tab("Progress"):
            _tab_progress(gr)
        with gr.Tab("Tutor"):
            _tab_tutor(gr)
    return demo_app


# =============================================================================
# demo()
# =============================================================================


def demo() -> DemoResult:
    """Build the merged bank from the corpus, run a seeded 5-question session with
    the correct answers (saved to ``output/demo-history.json``, not the user's
    ``history.json``), scan progress and save ``output/progress.png``, index the
    lessons and ask one question through the ``stub`` provider, and write
    ``output/tutor_session.txt`` and a deterministic ``output/metrics.txt``.

    Tier ``rag``: needs LangChain + Chroma + the MiniLM embedding model, so it
    never runs in CI and is skipped by ``uv run demo``/``uv run demo --models`` -
    only ``uv run demo --all`` runs it.

    The LLM provider is force-pinned to ``stub`` (not "whichever provider is
    configured") so the committed proof does not depend on whether Ollama
    happens to be running on this machine - see the README for running
    against a real provider. Every quiz answer is the question's own correct
    id, so the session score is deterministic (5/5) and independent of any
    random guessing.
    """
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bank_path = quiz_engine.build_bank_from_corpus(CORPUS_DIR, quiz_engine.BANK_PATH)
    questions = quiz_engine.load_bank(bank_path)
    session = quiz_engine.build_session(questions, n=5, seed=42)
    answers = [q["correct"] for q in session]
    summary = quiz_engine._session_summary(session, answers)  # same-package helper
    DEMO_HISTORY_PATH.unlink(missing_ok=True)  # one session per run, never accumulating
    quiz_engine.save_session(summary, path=DEMO_HISTORY_PATH)

    status_df = progress.scan_statuses(CORPUS_DIR)
    pivot = progress.aggregate(status_df)
    progress.plot_progress(pivot)

    index_report = tutor.index_lessons(reindex=True, corpus_dir=CORPUS_DIR)
    session_path = OUT_DIR / "tutor_session.txt"
    session_path.unlink(missing_ok=True)
    result = tutor.ask(DEMO_QUESTION, provider="stub")
    with session_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"Q: {DEMO_QUESTION}\nA: {result['answer']}\nSources: {', '.join(result['sources'])}\n"
        )

    figures = {
        "questions": str(len(session)),
        "session_score": f"{summary['score']}/{summary['total']}",
        "lessons_indexed": str(index_report["total_files"]),
        "provider": "stub",
    }
    (OUT_DIR / "metrics.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in figures.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    seconds = time.perf_counter() - start
    log.info("demo: %s", figures)
    return DemoResult("p12-study-hub", "ok", figures, seconds=round(seconds, 2))


# =============================================================================
# CLI dispatch
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    p.add_argument("--quiz", action="store_true", help="run the quiz in the console")
    p.add_argument("--dashboard", action="store_true", help="run the progress dashboard")
    p.add_argument("--ask", help="ask the RAG tutor a question")
    p.add_argument("--reindex", action="store_true", help="rebuild the lesson index")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--module", default=None)
    p.add_argument("--type", dest="qtype", default=None)
    p.add_argument("--ui", action="store_true", help="explicitly launch the Gradio UI")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--host", default="127.0.0.1", help="Bind address for the Gradio UI (default: 127.0.0.1)."
    )
    p.add_argument("--demo", action="store_true", help="run the offline demo (needs the rag group)")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch a CLI subcommand (--quiz/--dashboard/--ask/--reindex) or launch the UI."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )

    if args.demo:
        result = demo()
        print(f"p12-study-hub: {result.status}")
        for key, value in result.figures.items():
            print(f"  {key}: {value}")
        print("  wrote: output/")
        print(f"  seconds: {result.seconds:.2f}")
        return 0 if result.status == "ok" else 1

    if args.reindex:
        print("[tutor]", tutor.index_lessons(reindex=True))
    if args.quiz:
        quiz_engine.run_quiz_cli(n=args.n, seed=args.seed, module=args.module, qtype=args.qtype)
        return 0
    if args.dashboard:
        print(progress.run_dashboard_cli())
        print(f"\nChart saved to {OUT_DIR / 'progress.png'}")
        return 0
    if args.ask:
        _ensure_lesson_index()
        tutor_result = tutor.ask(args.ask)
        sources = ", ".join(tutor_result["sources"])
        print(f"\nQ: {args.ask}\nA: {tutor_result['answer']}\nSources: {sources}")
        return 0
    if args.reindex:  # a bare --reindex already ran above; don't also launch the UI
        return 0

    build_ui().launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
