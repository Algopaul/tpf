"""Delete pipeline artifacts from a given step onwards, enabling a clean restart.

Steps (matching CLAUDE.md numbering):
    01  process_trajectories  →  cfm_train_data/  +  cfm_train_data_wds/
    02  train_cfm             →  CFM checkpoints in multirun/
    03  gen_cond_trajectories →  cfm_trajectories/
    04  process_regression    →  reg_train_data/
    05  train_regression      →  regression checkpoints in multirun/

Usage:
    python tpflow/tools/pipeline_restart.py kolflow --from 3          # dry run
    python tpflow/tools/pipeline_restart.py kolflow --from 3 --yes    # delete
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

DATA_ROOT = Path("data/datasets")

STEP_LABELS = {
    1: "process_trajectories",
    2: "train_cfm",
    3: "gen_cond_trajectories",
    4: "process_regression",
    5: "train_regression",
}


def _find_checkpoint_dirs(outputs_dir: Path, dataset: str, regression: bool) -> list[Path]:
    """Return epoch-level checkpoint directories for *dataset* in *outputs_dir*."""
    results = []
    for info_file in outputs_dir.rglob("checkpoint_info.json"):
        try:
            info = json.loads(info_file.read_text())
        except Exception:
            continue
        if info.get("data_name") != dataset:
            continue
        is_reg = "time_conditioned" in info
        if is_reg != regression:
            continue
        results.append(info_file.parent)
    return results


def _artifacts(
    ds: str, from_step: int, outputs_dir: Path
) -> list[tuple[str, Path, bool]]:
    """Return [(label, path, is_dir)] for all artifacts at step >= from_step."""
    base = DATA_ROOT / ds
    items: list[tuple[int, str, Path]] = [
        (1, "cfm_train_data", base / "cfm_train_data"),
        (1, "cfm_train_data_wds", base / "cfm_train_data_wds"),
        (3, "cfm_trajectories", base / "cfm_trajectories"),
        (4, "reg_train_data", base / "reg_train_data"),
    ]
    results: list[tuple[str, Path, bool]] = []
    for step, label, path in items:
        if step >= from_step and path.exists():
            results.append((f"step {step:02d}  {label}", path, True))

    if from_step <= 2:
        for p in _find_checkpoint_dirs(outputs_dir, ds, regression=False):
            results.append(("step 02  cfm_checkpoint", p, True))
    if from_step <= 5:
        for p in _find_checkpoint_dirs(outputs_dir, ds, regression=True):
            results.append(("step 05  reg_checkpoint", p, True))

    return results


def _fmt_size(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if total >= 1 << 30:
        return f"{total / (1 << 30):.1f} GB"
    if total >= 1 << 20:
        return f"{total / (1 << 20):.1f} MB"
    return f"{total / (1 << 10):.1f} KB"


@app.command()
def main(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. kolflow, hw2d)"),
    from_step: int = typer.Option(..., "--from", help="Restart from this step (1-5)"),
    yes: bool = typer.Option(False, "--yes", help="Actually delete (default: dry run)"),
    outputs_dir: Path = typer.Option(Path("multirun"), help="Root multirun directory"),
):
    if from_step < 1 or from_step > 5:
        console.print("[red]--from must be between 1 and 5[/red]")
        raise typer.Exit(1)

    artifacts = _artifacts(dataset, from_step, outputs_dir)

    if not artifacts:
        console.print(f"[green]Nothing to delete for {dataset} from step {from_step}.[/green]")
        return

    label = STEP_LABELS[from_step]
    title = f"{'Deleting' if yes else 'Would delete'}: {dataset} from step {from_step} ({label})"
    table = Table(show_header=True, header_style="bold", title=title)
    table.add_column("Stage")
    table.add_column("Path")
    table.add_column("Size", justify="right")

    for stage_label, path, _ in artifacts:
        size = _fmt_size(path) if path.exists() else "—"
        table.add_row(stage_label, str(path), size)

    console.print(table)

    if not yes:
        console.print("\n[yellow]Dry run — pass --yes to actually delete.[/yellow]")
        return

    for _, path, _ in artifacts:
        if path.exists():
            shutil.rmtree(path)
            console.print(f"[red]deleted[/red]  {path}")

    console.print(f"\n[green]Done. Re-run `just status {dataset}` to see the next step.[/green]")


if __name__ == "__main__":
    app()
