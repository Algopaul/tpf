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
    "langevin_potential": {
        "process": "langevin-processed",
        "train_cfm": "langevin-cfm",
        "cfm_trajectories": "langevin-cfm-trajectories",
        "cfm_trajectories_processed": "langevin-cfm-trajectories-processed",
        "regression": "langevin-regression",
    },
    "holder": {
        "process": "holder-processed",
        "train_cfm": "holder-cfm",
        "cfm_trajectories": "holder-cfm-trajectories",
        "cfm_trajectories_processed": "holder-cfm-trajectories-processed",
        "regression": "holder-regression",
    },
    "vlasov": {
        "process": "vlasov-processed",
        "train_cfm": "vlasov-cfm",
        "cfm_trajectories": "vlasov-cfm-trajectories",
        "cfm_trajectories_processed": "vlasov-cfm-trajectories-processed",
        "regression": "vlasov-regression",
    },
    "imgrot": {
        "process": "imgrot-processed",
        "train_cfm": None,
        "cfm_trajectories": None,
        "cfm_trajectories_processed": None,
        "regression": None,
    },
}

# Per-variant regression recipes: variant stem → just recipe name.
# The variant stem is the zarr filename without extension (e.g. "model1", "ot", "physics").
_REGRESSION_RECIPES: dict[str, dict[str, str]] = {
    "langevin_potential": {
        "model1":  "langevin-regression",
        "ot":      "langevin-ot-regression",
    },
    "kolflow": {
        "model1":  "kolflow-regression",
        "ot":      "kolflow-ot-regression",
        "physics": "kolflow-physics-regression",
    },
    "hw2d": {
        "model1":  "hw2d-regression",
    },
    "holder": {
        "model1":  "holder-regression",
        "ot":      "holder-ot-regression",
        "physics": "holder-physics-regression",
    },
    "vlasov": {
        "model1":  "vlasov-regression",
        "ot":      "vlasov-ot-regression",
        "physics": "vlasov-physics-regression",
    },
}


def _recipe(ds: str, key: str) -> Optional[str]:
    return _RECIPES.get(ds, {}).get(key)


def _fmt_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")


def _zarr_marker(path: Path) -> Optional[Path]:
    """Return the format marker file for a zarr store (v3: zarr.json, v2: .zgroup)."""
    for name in ("zarr.json", ".zgroup"):
        p = path / name
        if p.exists():
            return p
    return None


def _zarr_exists(path: Path) -> bool:
    return _zarr_marker(path) is not None


def _zarr_mtime(path: Path) -> Optional[float]:
    p = _zarr_marker(path)
    return p.stat().st_mtime if p is not None else None


def _data_variant(train_data: str) -> str:
    """Extract a short label from a train_data path, e.g. 'ot', 'model1', 'physics'."""
    return Path(train_data).stem if train_data else "unknown"


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


def _latest_per_variant(checkpoints: list[dict]) -> dict[str, dict]:
    """Return the highest-epoch checkpoint for each train_data variant."""
    by_variant: dict[str, list[dict]] = {}
    for ckpt in checkpoints:
        variant = _data_variant(ckpt.get("train_data", ""))
        by_variant.setdefault(variant, []).append(ckpt)
    return {v: _latest(ckpts) for v, ckpts in sorted(by_variant.items())}


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

    reg_ckpts = _find_checkpoints(outputs_dir, ds, regression=True)
    if not reg_ckpts:
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
    reg_ckpts_by_variant = _latest_per_variant(
        _find_checkpoints(outputs_dir, dataset, regression=True)
    )

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
            if reg_ckpts_by_variant:
                for i, (variant, ckpt) in enumerate(reg_ckpts_by_variant.items()):
                    step_label = "6" if i == 0 else ""
                    stage_label = f"reg ({variant})"
                    modified3 = _fmt_mtime(ckpt["_mtime"])
                    details3 = f"epoch {ckpt['epoch']}  {ckpt['path']}"
                    table.add_row(step_label, stage_label, "[green]✓[/green]", modified3, details3)
            else:
                table.add_row("6", "reg_checkpoint", "[red]✗[/red]", "", "")

    console.print(table)

    stage, cmd = _next_step(dataset, outputs_dir, extras)
    if stage not in ("reg_checkpoint", "done"):
        console.print(f"\n[bold]Next step[/bold] ([cyan]{stage}[/cyan]):")
        console.print(f"  [yellow]{cmd}[/yellow]")
        return

    # Steps 1-5 complete: show all available regression training options.
    reg_dir = base / "reg_train_data"
    available_variants = sorted(
        p.stem for p in reg_dir.glob("*.zarr") if _zarr_exists(p)
    ) if reg_dir.exists() else []

    if not available_variants:
        console.print(f"\n[bold]Next step[/bold] ([cyan]{stage}[/cyan]):")
        console.print(f"  [yellow]{cmd}[/yellow]")
        return

    variant_recipes = _REGRESSION_RECIPES.get(dataset, {})
    sfx = f" {extras}" if extras else ""

    console.print("\n[bold]Regression training options:[/bold]")
    rtable = Table(show_header=True, header_style="bold")
    rtable.add_column("Variant")
    rtable.add_column("Status")
    rtable.add_column("Epoch")
    rtable.add_column("Command")

    for variant in available_variants:
        ckpt = reg_ckpts_by_variant.get(variant)
        recipe = variant_recipes.get(variant)
        if recipe:
            command = f"just {recipe}{sfx}"
        else:
            command = f"[dim]no recipe defined[/dim]"
        if ckpt:
            status_str = "[green]trained[/green]"
            epoch_str = str(ckpt["epoch"])
        else:
            status_str = "[yellow]untrained[/yellow]"
            epoch_str = ""
        rtable.add_row(variant, status_str, epoch_str, command)

    console.print(rtable)


if __name__ == "__main__":
    app()
