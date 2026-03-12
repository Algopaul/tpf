"""Show pipeline progress for a dataset and suggest the next step.

Usage:
    python tpflow/tools/pipeline_status.py kolflow
    python tpflow/tools/pipeline_status.py kolflow --extras "+env=slurm"
    python tpflow/tools/pipeline_status.py kolflow --outputs-dir multirun
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

DATA_ROOT = Path("data/datasets")

# Dataset-specific just recipe names; None falls back to the generic field-* recipe.
_RECIPES: dict[str, dict[str, Optional[str]]] = {
    "kolflow": {
        "process": "kolflow-processed",
        "train_cfm": "kolflow-cfm",
        "cfm_trajectories": "kolflow-cfm-trajectories",  # <checkpoint> <modelname> [extras]
        "cfm_trajectories_processed": "kolflow-cfm-trajectories-processed",
        "regression": "kolflow-regression",
    },
    "hw2d": {
        "process": "hw2d-data",
        "train_cfm": "hw2d-cfm",
        "cfm_trajectories": None,
        "cfm_trajectories_processed": None,
        "regression": "hw2d-regression",
    },
    "imgrot": {
        "process": "imgrot-processed",
        "train_cfm": None,
        "cfm_trajectories": None,
        "cfm_trajectories_processed": None,
        "regression": None,
    },
}


def _recipe(ds: str, key: str) -> Optional[str]:
    return _RECIPES.get(ds, {}).get(key)


def _fmt_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def _zarr_exists(path: Path) -> bool:
    return (path / "zarr.json").exists()


def _zarr_mtime(path: Path) -> Optional[float]:
    p = path / "zarr.json"
    return p.stat().st_mtime if p.exists() else None


def _find_checkpoints(outputs_dir: Path, dataset: str, regression: bool) -> list[dict]:
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
        info["path"] = str(info_file.parent)
        info["_mtime"] = info_file.stat().st_mtime
        results.append(info)
    return results


def _latest(checkpoints: list[dict]) -> Optional[dict]:
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda c: (c.get("epoch", 0), c["_mtime"]))


def _next_step(ds: str, outputs_dir: Path, extras: str) -> tuple[str, str]:
    """Return (stage_label, command) for the next incomplete pipeline step."""
    base = DATA_ROOT / ds
    sfx = f" {extras}" if extras else ""

    if not _zarr_exists(base / "raw_trajectories" / "train.zarr"):
        return "raw_trajectories", f"# No raw trajectories found — generate them first"

    if not _zarr_exists(base / "cfm_train_data" / "train.zarr"):
        recipe = _recipe(ds, "process") or f"field-cfm-data {ds}"
        return "cfm_train_data", f"just {recipe}{sfx}"

    cfm_ckpt = _latest(_find_checkpoints(outputs_dir, ds, regression=False))
    if cfm_ckpt is None:
        recipe = _recipe(ds, "train_cfm") or f"field-cfm {ds}"
        return "cfm_checkpoint", f"just {recipe}{sfx}"

    if not _zarr_exists(base / "cfm_trajectories" / "model1.zarr"):
        ckpt = cfm_ckpt["path"]
        recipe = _recipe(ds, "cfm_trajectories")
        if recipe:
            cmd = f"just {recipe} {ckpt} model1{sfx}"
        else:
            cmd = f"just field-cfm-trajectories {ds} {ckpt} model1{sfx}"
        return "cfm_trajectories", cmd

    if not _zarr_exists(base / "reg_train_data" / "model1.zarr"):
        recipe = _recipe(ds, "cfm_trajectories_processed")
        if recipe:
            cmd = f"just {recipe}{sfx}"
        else:
            cmd = f"just field-cfm-trajectories-processed {ds}{sfx}"
        return "reg_train_data", cmd

    reg_ckpt = _latest(_find_checkpoints(outputs_dir, ds, regression=True))
    if reg_ckpt is None:
        recipe = _recipe(ds, "regression")
        if recipe:
            cmd = f"just {recipe}{sfx}"
        else:
            cmd = f"just field-regression {ds}{sfx}"
        return "reg_checkpoint", cmd

    return "done", "# Pipeline complete!"


@app.command()
def main(
    dataset: str = typer.Argument(..., help="Dataset name (e.g. kolflow, hw2d)"),
    outputs_dir: Path = typer.Option(Path("multirun"), help="Root multirun directory"),
    extras: str = typer.Option("", help="Extra just args appended to the next command"),
):
    base = DATA_ROOT / dataset
    cfm_ckpt = _latest(_find_checkpoints(outputs_dir, dataset, regression=False))
    reg_ckpt = _latest(_find_checkpoints(outputs_dir, dataset, regression=True))

    data_stages = [
        ("1", "raw_trajectories", base / "raw_trajectories" / "train.zarr"),
        ("2", "cfm_train_data", base / "cfm_train_data" / "train.zarr"),
        ("4", "cfm_trajectories", base / "cfm_trajectories" / "model1.zarr"),
        ("5", "reg_train_data", base / "reg_train_data" / "model1.zarr"),
    ]

    table = Table(show_header=True, header_style="bold", title=f"Pipeline: {dataset}")
    table.add_column("Step", justify="right")
    table.add_column("Stage", justify="center")
    table.add_column("Status")
    table.add_column("Modified")
    table.add_column("Details")

    for step, label, path in data_stages:
        ok = _zarr_exists(path)
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        mtime = _zarr_mtime(path)
        modified = _fmt_mtime(mtime) if mtime else ""
        table.add_row(step, label, status, modified, str(path) if ok else "")

        if label == "cfm_train_data":
            ok2 = cfm_ckpt is not None
            status2 = "[green]✓[/green]" if ok2 else "[red]✗[/red]"
            modified2 = _fmt_mtime(cfm_ckpt["_mtime"]) if ok2 else ""
            details2 = f"epoch {cfm_ckpt['epoch']}  {cfm_ckpt['path']}" if ok2 else ""
            table.add_row("3", "cfm_checkpoint", status2, modified2, details2)

        if label == "reg_train_data":
            ok3 = reg_ckpt is not None
            status3 = "[green]✓[/green]" if ok3 else "[red]✗[/red]"
            modified3 = _fmt_mtime(reg_ckpt["_mtime"]) if ok3 else ""
            details3 = f"epoch {reg_ckpt['epoch']}  {reg_ckpt['path']}" if ok3 else ""
            table.add_row("6", "reg_checkpoint", status3, modified3, details3)

    console.print(table)

    stage, cmd = _next_step(dataset, outputs_dir, extras)
    if stage == "done":
        console.print("\n[green]Pipeline complete![/green]")
    else:
        console.print(f"\n[bold]Next step[/bold] ([cyan]{stage}[/cyan]):")
        console.print(f"  [yellow]{cmd}[/yellow]")


if __name__ == "__main__":
    app()
