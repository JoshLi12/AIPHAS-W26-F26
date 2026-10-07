"""Command line for the standardized pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from standardized_pipeline.live import open_live_view, start_live_view
from standardized_pipeline.pipeline import apply_reviews, run_pipeline

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_INPUT = (
    _PACKAGE_ROOT.parent / "Winter26" / "sample_data" / "AIPHAS_raw_data_2026-2.csv"
)
_DEFAULT_SCHEMA = _PACKAGE_ROOT / "examples" / "target_schema.json"
_DEFAULT_OUTPUT = _PACKAGE_ROOT / "output"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Turn a raw table into a dashboard file and a model file. "
            "Model decisions at or above the threshold are kept. "
            "Decisions below the threshold wait for a person."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Standardize a raw CSV.")
    run.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="Raw CSV. Default: Winter26/sample_data/AIPHAS_raw_data_2026-2.csv.",
    )
    run.add_argument(
        "--schema",
        type=Path,
        default=_DEFAULT_SCHEMA,
        help="Target schema JSON. Default: examples/target_schema.json.",
    )
    run.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    run.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Minimum confidence the model may complete on its own. Default: 0.80.",
    )
    run.add_argument(
        "--llm",
        choices=("auto", "offline", "ollama"),
        default="auto",
        help="auto uses Ollama when it is reachable, otherwise the offline model.",
    )
    run.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    run.add_argument("--ollama-model", default="llama3.2")
    run.add_argument(
        "--live",
        action="store_true",
        help="Serve a local page that updates as each step is written.",
    )
    run.add_argument("--live-port", type=int, default=8765)

    review = sub.add_parser("apply-reviews", help="Apply human answers and rebuild exports.")
    review.add_argument("--output-dir", type=Path, required=True)
    review.add_argument(
        "--responses",
        type=Path,
        required=True,
        help="CSV with decision_id and accepted_value.",
    )
    review.add_argument("--llm", choices=("auto", "offline", "ollama"), default="auto")
    review.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    review.add_argument("--ollama-model", default="llama3.2")
    review.add_argument(
        "--live",
        action="store_true",
        help="Serve a local page that updates as each review step is written.",
    )
    review.add_argument("--live-port", type=int, default=8765)

    watch = sub.add_parser("watch", help="Open the live page for an existing output folder.")
    watch.add_argument("--output-dir", type=Path, required=True)
    watch.add_argument("--port", type=int, default=8765)

    args = parser.parse_args()
    view = None
    if args.command == "watch" or getattr(args, "live", False):
        output_dir = args.output_dir
        port = args.port if args.command == "watch" else args.live_port
        view = start_live_view(output_dir, port)
        open_live_view(view)
    if args.command == "watch":
        print("Watching for new steps. Press Ctrl+C to stop.", flush=True)
        if view is not None:
            view.wait_until_stopped()
        return
    if args.command == "run":
        summary = run_pipeline(
            input_csv=args.input,
            schema_path=args.schema,
            output_dir=args.output_dir,
            threshold=args.threshold,
            llm=args.llm,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            echo=True,
        )
    else:
        summary = apply_reviews(
            output_dir=args.output_dir,
            responses_csv=args.responses,
            llm=args.llm,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
            echo=True,
        )
    print(json.dumps(summary, indent=2))
    if view is not None:
        print("Live view is still running. Press Ctrl+C to stop.", flush=True)
        view.wait_until_stopped()


if __name__ == "__main__":
    main()
