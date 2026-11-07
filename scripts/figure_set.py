"""Inspect, add, export and verify a figure set (``figures/<Set>/manifest.json``).

    python scripts/figure_set.py {Manuscript,Presentation} {list,add,export,refresh,verify} [--only SUBSTR]

    add  --source PATH --dest PATH_IN_SET --command CMD [--section S] [--workflow W] [--note N]
         registers one figure, copies it into the set and records its sha256. ``--source`` is a
         path under figures/ (absolute, or relative to figures/); ``--dest`` is relative to the set
         and its first folder is the section unless --section says otherwise.
    verify  checks every entry (dest in the set tree, section = first folder, dest is a byte copy
            of the source, recorded sha256 current, size budget) and, without --only, flags files
            in the set folder that no entry lists. Exit 1 on any problem.

Ported from DiffusionParameterCalibration ``Scripts/Tools/figure_set.py``; logic in
``scripts/figure_sets.py``. Run it on a compute node, like everything else here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figure_sets  # noqa: E402

figure_set = figure_sets.SETS["Manuscript"]


def _selected(only):
    return [e for e in figure_set.load_manifest() if only is None or only in e.dest]


def cmd_list(args) -> int:
    entries = _selected(args.only)
    if not entries:
        print(f"no entries in {figure_set.manifest_path()}")
        return 0
    for section in sorted({e.section for e in entries}):
        print(f"\n=== {section or '(no section)'} ===")
        for e in [x for x in entries if x.section == section]:
            digest = e.sha256[:12] + "…" if e.sha256 else "(no hash)"
            print(f"\n  {e.dest}\n    <- {e.source}\n    sha256 {digest}\n    $ {e.command}")
            if e.note:
                print(f"    note: {e.note}")
    print(f"\n{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}")
    return 0


def cmd_add(args) -> int:
    entry = figure_set.add(source=args.source, dest=args.dest, command=args.command,
                           section=args.section, workflow=args.workflow, note=args.note)
    print(f"added {entry.dest} <- {entry.source}")
    return 0


def cmd_export(args) -> int:
    written = figure_set.export_all(only=args.only)
    print(f"exported {len(written)} file(s); manifest hashes refreshed")
    return 0


def cmd_refresh(args) -> int:
    updated = figure_set.refresh(only=args.only)
    print(f"refreshed {len(figure_set.load_manifest())} entr(ies); {len(updated)} hash(es) changed")
    for dest in updated:
        print(f"  ~ {dest}")
    return 0


def cmd_verify(args) -> int:
    report = figure_set.verify(only=args.only, max_mb=args.max_mb)
    for problem in report.problems:
        print(f"  !! {problem}")
    status = "OK" if report.ok else f"FAILED ({len(report.problems)} problem(s))"
    print(f"{status}: {report.checked} entr{'y' if report.checked == 1 else 'ies'} "
          f"checked against {figure_set.manifest_path()}")
    return 0 if report.ok else 1


def main(argv=None) -> int:
    global figure_set
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", choices=sorted(figure_sets.SETS),
                        help="Which figure set (a folder under figures/).")
    sub = parser.add_subparsers(dest="command_name", required=True)

    def _add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--only", default=None, help="Substring match against the destination path.")
        p.set_defaults(func=func)
        return p

    _add("list", cmd_list, "Print each figure, its source and its command.")
    add_p = _add("add", cmd_add, "Register one figure, copy it and record its sha256.")
    add_p.add_argument("--source", required=True)
    add_p.add_argument("--dest", required=True)
    add_p.add_argument("--command", required=True, help="Shell command that regenerates the source.")
    add_p.add_argument("--section", default="")
    add_p.add_argument("--workflow", default="")
    add_p.add_argument("--note", default="")
    _add("export", cmd_export, "Copy every source to its destination, then refresh hashes.")
    _add("refresh", cmd_refresh, "Recompute the sha256 of each destination file.")
    verify_p = _add("verify", cmd_verify, "Check the set offline (no rendering); exit 1 on any problem.")
    verify_p.add_argument("--max-mb", type=float, default=50.0, help="Per-file size budget in MB (default: 50).")

    args = parser.parse_args(argv)
    figure_set = figure_sets.get(args.set)
    if args.func not in (cmd_verify, cmd_add) and args.only is not None and not _selected(args.only):
        print(f"--only {args.only!r} matched no entry in {figure_set.manifest_path()}")
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
