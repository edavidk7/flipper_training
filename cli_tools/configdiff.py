import torch
from flipper_training import OmegaConf
from pathlib import Path
from rich.console import Console
from rich.table import Table


def add_diff_rows(table: Table, diff_dict: dict, current_path: str = ""):
    """Recursively add rows to the Rich table for differences."""
    for key, value in diff_dict.items():
        key_path = f"{current_path}.{key}" if current_path else key
        if isinstance(value, dict):
            # Recurse for nested differences
            add_diff_rows(table, value, key_path)
        elif isinstance(value, tuple) and len(value) == 2:
            # Actual difference found
            v1, v2 = value
            # Convert None to a displayable string
            str_v1 = str(v1) if v1 is not None else "[dim]N/A[/dim]"
            str_v2 = str(v2) if v2 is not None else "[dim]N/A[/dim]"
            # Add row with markup for clarity
            table.add_row(f"[cyan]{key_path}[/cyan]", str_v1, str_v2)
        else:
            # Should not happen with recursive_diff_dict output, but handle defensively
            table.add_row(f"[cyan]{key_path}[/cyan]", "[red]Error[/red]", f"[red]Unexpected value type: {type(value)}[/red]")


def recursive_diff_dict(d1, d2):
    """
    Recursively compare two dictionaries and return the differences.
    """
    diff = {}
    for key in d1.keys():
        if key not in d2:
            diff[key] = (d1[key], None)
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            nested_diff = recursive_diff_dict(d1[key], d2[key])
            if nested_diff:
                diff[key] = nested_diff
        else:
            v1 = d1[key]
            v2 = d2[key]
            v1 = v1 if not isinstance(v1, torch.Tensor) else v1.tolist()
            v2 = v2 if not isinstance(v2, torch.Tensor) else v2.tolist()
            if v1 != v2:
                diff[key] = (v1, v2)
    for key in d2.keys():
        if key not in d1:
            diff[key] = (None, d2[key])
    return diff


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two config files")
    parser.add_argument("config1", type=Path, help="Path to the first config file")
    parser.add_argument("config2", type=Path, help="Path to the second config file")
    args = parser.parse_args()
    # Get paths
    path1 = args.config1
    path2 = args.config2

    # Extract names: parent_dir/filename
    name1 = f"{path1.parent.name}/{path1.name}"
    name2 = f"{path2.parent.name}/{path2.name}"

    # Load configs
    config1 = OmegaConf.load(path1)
    config2 = OmegaConf.load(path2)

    # Convert to dictionaries for comparison
    config1_dict = OmegaConf.to_container(config1, resolve=True)
    config2_dict = OmegaConf.to_container(config2, resolve=True)

    # Calculate differences
    diff = recursive_diff_dict(config1_dict, config2_dict)

    # Display differences
    console = Console()
    if diff:
        table = Table(show_lines=True)
        table.add_column("Key", style="cyan", no_wrap=True)
        # Use extracted names as column headers
        table.add_column(name1, justify="right")
        table.add_column(name2, justify="right")
        add_diff_rows(table, diff)
        console.print(table)
    else:
        # Handle case with no differences
        console.print(f"No differences found between [green]{name1}[/green] and [green]{name2}[/green].")
