#!/usr/bin/env python3
"""Validate model/evaluation result artifacts for the text-game-agent project.

This script checks saved JSON outputs for structural correctness and internal
consistency. It does not modify any files.

Supported artifact types (auto-detected):
  - Game evaluation results produced by ``evaluate_games.py`` /
    ``evaluate_level2.py`` (summary + ``results`` with per-game ``steps``).
  - Training histories produced by ``train_sft.py`` (list of epoch dicts) or
    ``train_level2_corrections.py`` (dict with ``epochs``/best metadata).

Examples:
  python src/validate_results.py
  python src/validate_results.py evaluation/validation_base.json
  python src/validate_results.py results/evaluation evaluation/level2
  python src/validate_results.py --glob "evaluation/**/*.json" --json
  python src/validate_results.py models --kind training_history --strict
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Issue:
    level: str  # "error" | "warning"
    path: str
    message: str

    def format(self) -> str:
        return f"[{self.level.upper()}] {self.path}: {self.message}"


@dataclass
class FileReport:
    file: Path
    kind: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def error(self, path: str, message: str) -> None:
        self.issues.append(Issue("error", path, message))

    def warn(self, path: str, message: str) -> None:
        self.issues.append(Issue("warning", path, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": str(self.file),
            "kind": self.kind,
            "ok": self.ok,
            "errors": sum(1 for i in self.issues if i.level == "error"),
            "warnings": sum(1 for i in self.issues if i.level == "warning"),
            "issues": [i.__dict__ for i in self.issues],
        }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_finite_number(value: Any) -> bool:
    return is_number(value) and math.isfinite(float(value))


def approx_equal(a: Any, b: Any, *, abs_tol: float = 1e-9, rel_tol: float = 1e-9) -> bool:
    if not (is_number(a) and is_number(b)):
        return False
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def detect_kind(data: Any, file: Path) -> str:
    if file.name == "training_history.json":
        return "training_history"
    if isinstance(data, dict):
        if "results" in data and isinstance(data.get("results"), list):
            return "game_eval"
        if "epochs" in data and isinstance(data.get("epochs"), list):
            return "training_history"
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        if any("epoch" in x or "train_loss" in x or "validation_loss" in x for x in data):
            return "training_history"
    return "unknown"


def require(report: FileReport, path: str, obj: Any, key: str, types: tuple[type, ...] | None = None) -> Any:
    if not isinstance(obj, dict):
        report.error(path, f"expected object to contain '{key}'")
        return None
    if key not in obj:
        report.error(path, f"missing required field '{key}'")
        return None
    value = obj[key]
    if types is not None and not isinstance(value, types):
        names = ", ".join(t.__name__ for t in types)
        report.error(f"{path}.{key}", f"expected {names}, got {type(value).__name__}")
    return value


def check_optional_type(report: FileReport, path: str, obj: dict[str, Any], key: str, types: tuple[type, ...]) -> None:
    if key not in obj:
        return
    value = obj[key]
    if value is None:
        return
    if not isinstance(value, types):
        names = " or ".join(t.__name__ for t in types)
        report.error(f"{path}.{key}", f"expected {names} or null, got {type(value).__name__}")


def validate_game_step(report: FileReport, path: str, step: Any, strict: bool) -> dict[str, Any]:
    if not isinstance(step, dict):
        report.error(path, f"expected object, got {type(step).__name__}")
        return {}

    for key in ("step", "observation", "action", "is_admissible"):
        require(report, path, step, key)

    if "step" in step and not isinstance(step["step"], int):
        report.error(f"{path}.step", f"expected int, got {type(step['step']).__name__}")
    elif isinstance(step.get("step"), int) and step["step"] < 0:
        report.error(f"{path}.step", "step index must be >= 0")

    if "action" in step and not isinstance(step["action"], str):
        report.error(f"{path}.action", f"expected str, got {type(step['action']).__name__}")
    if "is_admissible" in step and not isinstance(step["is_admissible"], bool):
        report.error(
            f"{path}.is_admissible",
            f"expected bool, got {type(step['is_admissible']).__name__}",
        )
    for key in ("reward",):
        check_optional_type(report, path, step, key, (int, float))
    for key in ("done", "won", "lost"):
        check_optional_type(report, path, step, key, (bool,))

    if "admissible_commands" in step:
        cmds = step["admissible_commands"]
        if not isinstance(cmds, list) or not all(isinstance(c, str) for c in cmds):
            report.error(f"{path}.admissible_commands", "expected list of strings")
        elif strict and isinstance(step.get("action"), str) and isinstance(step.get("is_admissible"), bool):
            in_cmds = step["action"] in cmds
            if in_cmds != step["is_admissible"]:
                report.warn(
                    path,
                    "is_admissible differs from (action in admissible_commands); "
                    "this can happen if commands changed, but is worth checking",
                )

    if isinstance(step.get("done"), bool) and step["done"]:
        if isinstance(step.get("won"), bool) and isinstance(step.get("lost"), bool):
            if step["won"] and step["lost"]:
                report.error(path, "final/done step cannot have both won and lost true")
    return step


def validate_game_result(report: FileReport, path: str, game: Any, strict: bool, *, max_steps_hint: Any = None) -> None:
    if not isinstance(game, dict):
        report.error(path, f"expected object, got {type(game).__name__}")
        return

    for key in ("game_id", "success", "num_steps", "steps"):
        require(report, path, game, key)

    if "game_id" in game and not isinstance(game["game_id"], str):
        report.error(f"{path}.game_id", f"expected str, got {type(game['game_id']).__name__}")
    for key in ("success", "won", "lost"):
        check_optional_type(report, path, game, key, (bool,))
    for key in ("num_steps", "invalid_actions", "max_score"):
        check_optional_type(report, path, game, key, (int, float))
    for key in ("invalid_action_rate", "total_reward", "expert_steps", "steps_above_expert"):
        check_optional_type(report, path, game, key, (int, float))

    steps = game.get("steps")
    if not isinstance(steps, list):
        if "steps" in game:
            report.error(f"{path}.steps", f"expected list, got {type(steps).__name__}")
        return

    parsed_steps = [validate_game_step(report, f"{path}.steps[{i}]", s, strict) for i, s in enumerate(steps)]

    if isinstance(game.get("num_steps"), int):
        if game["num_steps"] != len(steps):
            report.error(
                path,
                f"num_steps={game['num_steps']} but steps contains {len(steps)} entries",
            )
        if game["num_steps"] < 0:
            report.error(f"{path}.num_steps", "num_steps must be >= 0")

    if steps:
        indices = [s.get("step") for s in parsed_steps if isinstance(s.get("step"), int)]
        if indices and indices != list(range(indices[0], indices[0] + len(indices))) and strict:
            report.warn(path, f"step indices are not contiguous from {indices[0]}: {indices[:10]}")
        if indices and indices[0] != 0:
            report.warn(path, f"first step index is {indices[0]}, expected 0")

    invalid_from_steps = sum(1 for s in parsed_steps if s.get("is_admissible") is False)
    admissible_flags = [s.get("is_admissible") for s in parsed_steps if isinstance(s.get("is_admissible"), bool)]
    if isinstance(game.get("invalid_actions"), int):
        if admissible_flags and game["invalid_actions"] != invalid_from_steps:
            report.error(
                path,
                f"invalid_actions={game['invalid_actions']} but step flags indicate {invalid_from_steps}",
            )
        if game["invalid_actions"] < 0:
            report.error(f"{path}.invalid_actions", "invalid_actions must be >= 0")

    if isinstance(game.get("invalid_action_rate"), (int, float)) and is_finite_number(game["invalid_action_rate"]):
        rate = float(game["invalid_action_rate"])
        if not (0.0 <= rate <= 1.0):
            report.error(f"{path}.invalid_action_rate", f"rate out of range [0,1]: {rate}")
        if len(steps) > 0 and isinstance(game.get("invalid_actions"), int):
            expected = game["invalid_actions"] / len(steps)
            if not approx_equal(rate, expected, abs_tol=1e-12, rel_tol=1e-12):
                report.error(
                    f"{path}.invalid_action_rate",
                    f"{rate} does not match invalid_actions/num_steps ({expected})",
                )
        elif len(steps) == 0 and not approx_equal(rate, 0.0):
            report.warn(f"{path}.invalid_action_rate", "no steps but invalid_action_rate is not 0")

    if isinstance(game.get("success"), bool) and isinstance(game.get("won"), bool):
        if game["success"] != game["won"] and strict:
            report.warn(path, f"success={game['success']} differs from won={game['won']}")
        if game["success"] and not game["won"]:
            report.error(path, "success is true but won is false")

    if parsed_steps:
        last = parsed_steps[-1]
        if isinstance(game.get("won"), bool) and isinstance(last.get("won"), bool) and strict:
            if game["won"] != last["won"]:
                report.warn(path, f"game.won={game['won']} differs from final step.won={last['won']}")
        if isinstance(game.get("lost"), bool) and isinstance(last.get("lost"), bool) and strict:
            if game["lost"] != last["lost"]:
                report.warn(path, f"game.lost={game['lost']} differs from final step.lost={last['lost']}")
        if isinstance(game.get("success"), bool) and game["success"] and isinstance(last.get("done"), bool):
            if not last["done"] and strict:
                report.warn(path, "successful game but final step.done is not true")

    if isinstance(max_steps_hint, int) and isinstance(game.get("num_steps"), int):
        if game["num_steps"] > max_steps_hint:
            report.error(
                path,
                f"num_steps={game['num_steps']} exceeds max_steps_per_game hint {max_steps_hint}",
            )


def validate_game_eval(report: FileReport, data: Any, strict: bool) -> None:
    if not isinstance(data, dict):
        report.error("$", f"expected top-level object, got {type(data).__name__}")
        return

    for key in ("num_games", "results"):
        require(report, "$", data, key)

    results = data.get("results")
    if not isinstance(results, list):
        return

    for key in (
        "num_games",
        "successful_games",
        "total_steps",
        "total_invalid_actions",
        "max_steps_per_game",
    ):
        check_optional_type(report, "$", data, key, (int,))
    for key in ("completion_rate", "average_steps", "invalid_action_rate", "average_expert_steps",
                "average_steps_above_expert", "average_steps_above_expert_on_success"):
        check_optional_type(report, "$", data, key, (int, float))
    for key in ("model", "adapter", "split", "prompt_version", "timestamp", "experiment", "manifest"):
        check_optional_type(report, "$", data, key, (str,))

    seen_game_ids: set[str] = set()
    max_steps_hint = data.get("max_steps_per_game")
    for i, game in enumerate(results):
        path = f"$.results[{i}]"
        validate_game_result(report, path, game, strict, max_steps_hint=max_steps_hint)
        if isinstance(game, dict) and isinstance(game.get("game_id"), str):
            gid = game["game_id"]
            if gid in seen_game_ids:
                report.error(path, f"duplicate game_id '{gid}'")
            seen_game_ids.add(gid)

    if isinstance(data.get("num_games"), int) and data["num_games"] != len(results):
        report.error("$", f"num_games={data['num_games']} but results contains {len(results)} entries")

    computed_games = len(results)
    computed_successes = sum(1 for g in results if isinstance(g, dict) and g.get("success") is True)
    computed_steps = sum(int(g["num_steps"]) for g in results if isinstance(g, dict) and isinstance(g.get("num_steps"), int))
    computed_invalid = sum(
        int(g["invalid_actions"])
        for g in results
        if isinstance(g, dict) and isinstance(g.get("invalid_actions"), int)
    )

    if isinstance(data.get("successful_games"), int) and data["successful_games"] != computed_successes:
        report.error(
            "$.successful_games",
            f"{data['successful_games']} but per-game successes sum to {computed_successes}",
        )

    if isinstance(data.get("completion_rate"), (int, float)) and is_finite_number(data["completion_rate"]):
        expected = (computed_successes / computed_games) if computed_games else 0.0
        if not approx_equal(data["completion_rate"], expected, abs_tol=1e-12, rel_tol=1e-12):
            report.error(
                "$.completion_rate",
                f"{data['completion_rate']} does not match successful_games/num_games ({expected})",
            )
        if not (0.0 <= float(data["completion_rate"]) <= 1.0):
            report.error("$.completion_rate", f"out of range [0,1]: {data['completion_rate']}")

    if isinstance(data.get("total_steps"), int) and data["total_steps"] != computed_steps:
        report.error("$.total_steps", f"{data['total_steps']} but per-game num_steps sum to {computed_steps}")

    if isinstance(data.get("average_steps"), (int, float)) and is_finite_number(data["average_steps"]):
        expected = (computed_steps / computed_games) if computed_games else 0.0
        if not approx_equal(data["average_steps"], expected, abs_tol=1e-12, rel_tol=1e-12):
            report.error(
                "$.average_steps",
                f"{data['average_steps']} does not match total_steps/num_games ({expected})",
            )

    if isinstance(data.get("total_invalid_actions"), int) and data["total_invalid_actions"] != computed_invalid:
        report.error(
            "$.total_invalid_actions",
            f"{data['total_invalid_actions']} but per-game invalid_actions sum to {computed_invalid}",
        )

    if isinstance(data.get("invalid_action_rate"), (int, float)) and is_finite_number(data["invalid_action_rate"]):
        expected = (computed_invalid / computed_steps) if computed_steps else 0.0
        if not approx_equal(data["invalid_action_rate"], expected, abs_tol=1e-12, rel_tol=1e-12):
            report.error(
                "$.invalid_action_rate",
                f"{data['invalid_action_rate']} does not match total_invalid_actions/total_steps ({expected})",
            )
        if not (0.0 <= float(data["invalid_action_rate"]) <= 1.0):
            report.error("$.invalid_action_rate", f"out of range [0,1]: {data['invalid_action_rate']}")

    if isinstance(max_steps_hint, int) and max_steps_hint < 0:
        report.error("$.max_steps_per_game", "must be >= 0")


def _validate_epoch_dict(report: FileReport, path: str, epoch: Any, strict: bool) -> None:
    if not isinstance(epoch, dict):
        report.error(path, f"expected object, got {type(epoch).__name__}")
        return
    if "epoch" in epoch and not isinstance(epoch["epoch"], int):
        report.error(f"{path}.epoch", f"expected int, got {type(epoch['epoch']).__name__}")
    for key in ("train_loss", "validation_loss", "learning_rate", "optimizer_steps"):
        if key in epoch and epoch[key] is not None:
            if not is_finite_number(epoch[key]):
                report.error(f"{path}.{key}", f"expected finite number, got {epoch[key]!r}")
            elif strict and key.endswith("loss") and float(epoch[key]) < 0:
                report.warn(f"{path}.{key}", f"negative loss: {epoch[key]}")


def validate_training_history(report: FileReport, data: Any, strict: bool) -> None:
    if isinstance(data, list):
        if not data:
            report.warn("$", "training history is empty")
        for i, epoch in enumerate(data):
            _validate_epoch_dict(report, f"$[{i}]", epoch, strict)
        epochs = [e.get("epoch") for e in data if isinstance(e, dict) and isinstance(e.get("epoch"), int)]
        if strict and epochs and epochs != sorted(epochs):
            report.warn("$", "epoch numbers are not sorted")
        return

    if not isinstance(data, dict):
        report.error("$", f"expected object or list, got {type(data).__name__}")
        return

    if "epochs" not in data:
        report.error("$", "training history dict missing 'epochs'")
        return
    epochs_val = data["epochs"]
    if not isinstance(epochs_val, list):
        report.error("$.epochs", f"expected list, got {type(epochs_val).__name__}")
        return
    for i, epoch in enumerate(epochs_val):
        _validate_epoch_dict(report, f"$.epochs[{i}]", epoch, strict)

    if "best_epoch" in data and not isinstance(data["best_epoch"], int):
        report.error("$.best_epoch", f"expected int, got {type(data['best_epoch']).__name__}")
    for key in ("initial_validation_loss", "best_validation_loss"):
        if key in data and data[key] is not None and not is_finite_number(data[key]):
            report.error(f"$.{key}", f"expected finite number, got {data[key]!r}")

    best_epoch = data.get("best_epoch")
    best_val = data.get("best_validation_loss")
    if isinstance(best_epoch, int) and epochs_val:
        matching = [e for e in epochs_val if isinstance(e, dict) and e.get("epoch") == best_epoch]
        if not matching:
            report.warn("$", f"best_epoch={best_epoch} not found in epochs")
        elif is_finite_number(best_val):
            epoch_val = matching[0].get("validation_loss")
            if is_finite_number(epoch_val) and not approx_equal(best_val, epoch_val, abs_tol=1e-12, rel_tol=1e-12):
                report.error(
                    "$.best_validation_loss",
                    f"{best_val} does not match epoch {best_epoch} validation_loss ({epoch_val})",
                )
        if strict and is_finite_number(best_val):
            vals = [e.get("validation_loss") for e in epochs_val if isinstance(e, dict) and is_finite_number(e.get("validation_loss"))]
            if vals and float(best_val) - min(float(v) for v in vals) > 1e-12:
                report.warn("$.best_validation_loss", f"{best_val} is not the minimum validation_loss in epochs ({min(vals)})")


def load_json(file: Path) -> Any:
    with file.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_file(file: Path, *, kind: str = "auto", strict: bool = False) -> FileReport:
    report = FileReport(file=file, kind="pending")
    try:
        data = load_json(file)
    except Exception as exc:  # noqa: BLE001 - report parse/load failures clearly
        report.kind = kind if kind != "auto" else "unknown"
        report.error("$", f"failed to load JSON: {exc}")
        return report

    detected = detect_kind(data, file) if kind == "auto" else kind
    report.kind = detected

    if detected == "game_eval":
        validate_game_eval(report, data, strict=strict)
    elif detected == "training_history":
        validate_training_history(report, data, strict=strict)
    else:
        report.warn("$", "unknown result format; no semantic checks applied")
    return report


def discover_inputs(paths: list[str], globs: list[str], *, include_models: bool) -> list[Path]:
    files: list[Path] = []
    root = PROJECT_ROOT

    if not paths and not globs:
        for base in (root / "results", root / "evaluation"):
            if base.exists():
                files.extend(sorted(p for p in base.rglob("*.json") if p.is_file()))
        if include_models:
            models = root / "models"
            if models.exists():
                files.extend(sorted(models.rglob("training_history.json")))

    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (root / p).resolve() if not (root / raw).exists() else (root / raw).resolve()
        if p.is_dir():
            files.extend(sorted(x for x in p.rglob("*.json") if x.is_file()))
        elif p.is_file():
            files.append(p)
        else:
            # Try glob relative to cwd/root.
            matched = list(Path().glob(raw)) or list(root.glob(raw))
            files.extend(sorted(m for m in matched if m.is_file()))

    for pattern in globs:
        files.extend(sorted(m for m in Path().glob(pattern) if m.is_file()))
        files.extend(sorted(m for m in root.glob(pattern) if m.is_file() and m not in files))

    # De-duplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        try:
            key = f.resolve()
        except OSError:
            key = f
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def print_report(reports: Iterable[FileReport], *, verbose: bool) -> int:
    reports = list(reports)
    errors = sum(1 for r in reports for i in r.issues if i.level == "error")
    warnings = sum(1 for r in reports for i in r.issues if i.level == "warning")
    failed = sum(1 for r in reports if not r.ok)

    for report in reports:
        if verbose or report.issues:
            status = "OK" if report.ok else "FAIL"
            print(f"{status} [{report.kind}] {report.file}")
        for issue in report.issues:
            print(f"  {issue.format()}")

    kinds: dict[str, int] = {}
    for r in reports:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    kind_summary = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"
    print(
        f"\nSummary: {len(reports)} file(s) checked ({kind_summary}), "
        f"{failed} failed, {errors} error(s), {warnings} warning(s)."
    )
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate saved model/evaluation result JSON files.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files, directories, or glob patterns to validate. Defaults to results/, evaluation/, and model training_history.json files.",
    )
    parser.add_argument("--glob", action="append", default=[], dest="globs", help="Additional glob pattern(s) to include.")
    parser.add_argument(
        "--kind",
        choices=["auto", "game_eval", "training_history"],
        default="auto",
        help="Force artifact type instead of auto-detection.",
    )
    parser.add_argument("--strict", action="store_true", help="Enable additional consistency warnings/checks.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    parser.add_argument("--verbose", action="store_true", help="Print every checked file, including clean files.")
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="When no paths are given, skip scanning models/**/training_history.json.",
    )
    args = parser.parse_args(argv)

    files = discover_inputs(args.paths, args.globs, include_models=not args.no_models)
    if not files:
        print("No result files found.", file=sys.stderr)
        return 2

    reports = [validate_file(f, kind=args.kind, strict=args.strict) for f in files]

    if args.json:
        payload = {
            "ok": all(r.ok for r in reports),
            "files": [r.to_dict() for r in reports],
            "summary": {
                "checked": len(reports),
                "failed": sum(1 for r in reports if not r.ok),
                "errors": sum(1 for r in reports for i in r.issues if i.level == "error"),
                "warnings": sum(1 for r in reports for i in r.issues if i.level == "warning"),
            },
        }
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1

    return print_report(reports, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
