"""List available model checkpoints.

Usage:
    python tpflow/tools/list_checkpoints.py
    python tpflow/tools/list_checkpoints.py --dataset gaurot
    python tpflow/tools/list_checkpoints.py --outputs-dir /path/to/outputs
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def find_checkpoints(outputs_dir: Path, dataset: Optional[str]) -> list[dict]:
  results = []
  for info_file in outputs_dir.rglob('checkpoint_info.json'):
    info = json.loads(info_file.read_text())
    if dataset and info.get('data_name') != dataset:
      continue
    info['path'] = str(info_file.parent)
    info['_mtime'] = info_file.stat().st_mtime
    results.append(info)
  results.sort(key=lambda c: (c.get('data_name', ''), c['_mtime']))
  return results


@app.command()
def main(
    dataset: Optional[str] = typer.Option(None, help='Filter by dataset name'),
    outputs_dir: Path = typer.Option(Path('outputs'), help='Root outputs dir'),
):
  checkpoints = find_checkpoints(outputs_dir, dataset)

  if not checkpoints:
    console.print('[yellow]No checkpoints found.[/yellow]')
    raise typer.Exit()

  table = Table(show_header=True, header_style='bold')
  table.add_column('epoch', justify='right')
  for c in ['dataset', 'model', 'sample_shape', 'path']:
    table.add_column(c)

  for c in checkpoints:
    table.add_row(
        str(c.get('epoch', '?')),
        c.get('data_name', '?'),
        c.get('model_type', '?'),
        str(c.get('sample_shape', '?')),
        c['path'],
    )

  console.print(table)


if __name__ == '__main__':
  app()
