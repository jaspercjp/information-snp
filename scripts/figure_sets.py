"""Named figure sets: a curated tree of copied figures with its own manifest.

Ported from DiffusionParameterCalibration (``Scripts/Core/figure_sets.py``), itself a port of
AntarcticIceMeltFracture's ``Scripts/Core/plotting/figure_sets.py`` (their ADR 0018).

A *figure set* is a directory ``figures/<Set>/`` holding byte-copies of figures rendered elsewhere
under ``figures/`` (the canonical renders: ``main-text/``, ``supp-info/``, ``slides/``, written by
``style.save``), plus ``figures/<Set>/manifest.json``, which names for every copy its ``dest`` (in
the set), its ``source`` (the canonical render), the command that regenerates the source, and the
copy's sha256. Copies are made with ``copy2``, never re-saved, so ``verify`` can prove the set is
exactly what the producers wrote.

Two sets exist (:data:`SETS`): ``Manuscript`` (the publication set) and ``Presentation`` (figures
for talks). Each set's manifest is independent, so a figure may ship in both and a change to one
set's entries never rewrites the other's.

Producers call :func:`export_everywhere` after saving (``style.save`` does, for every file it
writes): it copies the file into every set that lists it as a ``source`` and is a no-op for a file
no set lists. A re-render that changes the bytes leaves the recorded sha256 stale until
``figure_set.py <Set> refresh``; ``verify`` reports that rather than hiding it.

All paths in a manifest are relative to the figures root, ``<repo>/figures``. CLI:
``scripts/figure_set.py``. Differences from the two ports it follows: ``verify`` also flags files in
the set folder that no entry lists (orphans) and a ``section`` that is not the dest's first folder
inside the set.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

__all__ = ["SCHEMA_VERSION", "FigureEntry", "VerifyReport", "FigureSet", "SETS", "get",
           "sha256_file", "export_everywhere", "figures_root"]

SCHEMA_VERSION = 1
GENERATED_BY = "scripts/figure_sets.py"
_CHUNK = 1 << 20
_NOT_FIGURES = {"manifest.json", "manifest.json.lock", "manifest.json.tmp"}


@dataclass(frozen=True)
class FigureEntry:
    """One figure in a set: where it lands, where it came from, how to remake it.

    ``section`` is the dest's first folder inside the set (``MainText``/``SuppInformation`` for the
    manuscript; a talk section such as ``ExchangeabilityMaps`` for the presentation); ``workflow``
    names the producing script.
    """

    dest: str
    source: str
    command: str
    sha256: str | None = None
    workflow: str = ""
    section: str = ""
    note: str = ""


@dataclass(frozen=True)
class VerifyReport:
    """Outcome of :meth:`FigureSet.verify`: ``ok``, how many entries were checked, why not."""

    ok: bool
    checked: int
    problems: list[str]


def figures_root() -> Path:
    """``<repo>/figures``; ``$FIGURES_ROOT`` overrides it (for tests)."""
    env = os.environ.get("FIGURES_ROOT")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "figures"


def _rel(p) -> str:
    p = Path(p)
    root = figures_root().resolve()
    if p.is_absolute():
        try:
            return Path(os.path.relpath(p.resolve(), root)).as_posix()
        except ValueError:
            return p.as_posix()
    parts = p.parts
    if parts and parts[0] == root.name:
        return Path(*parts[1:]).as_posix()
    return p.as_posix()


def _abs(rel: str) -> Path:
    return figures_root() / rel


def sha256_file(path) -> str:
    """Streaming sha256 of ``path`` (1 MiB chunks), as a lowercase hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _entries_from_doc(doc: dict) -> tuple:
    fields = set(FigureEntry.__dataclass_fields__)
    out = [FigureEntry(**{k: v for k, v in raw.items() if k in fields})
           for raw in doc.get("entries", [])]
    return tuple(sorted(out, key=lambda e: e.dest))


def _doc_from_entries(entries, generated_by: str) -> dict:
    return {"schema": SCHEMA_VERSION, "generated_by": generated_by,
            "entries": [asdict(e) for e in sorted(entries, key=lambda e: e.dest)]}


def _read(path: Path) -> tuple[FigureEntry, ...]:
    if not path.exists():
        return ()
    with open(path) as f:
        return _entries_from_doc(json.load(f))


def _write(path: Path, entries, generated_by: str = GENERATED_BY) -> None:
    text = json.dumps(_doc_from_entries(entries, generated_by), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(text)
    os.replace(tmp, path)


class _locked:
    """Exclusive ``flock`` on ``<manifest>.lock`` for a read-modify-write of the manifest."""

    def __init__(self, path: Path):
        self.lock_path = path.with_suffix(path.suffix + ".lock")

    def __enter__(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.lock_path, "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        return False


def _selected(entries, only):
    return [e for e in entries if only is None or only in e.dest]


class FigureSet:
    """One named set: ``figures/<name>/`` and its ``manifest.json``."""

    def __init__(self, name: str, *, generated_by: str = GENERATED_BY):
        self.name = name
        self.generated_by = generated_by

    def __repr__(self) -> str:
        return f"FigureSet({self.name!r})"

    def manifest_path(self) -> Path:
        """``<figures_root>/<name>/manifest.json`` (never creates the directory)."""
        return figures_root() / self.name / "manifest.json"

    def load_manifest(self, path=None) -> tuple[FigureEntry, ...]:
        """Read the manifest (default :meth:`manifest_path`); ``()`` if it is absent."""
        return _read(Path(path) if path else self.manifest_path())

    def save_manifest(self, entries, path=None) -> None:
        """Merge ``entries`` into the manifest on disk, keyed by ``dest``."""
        path = Path(path) if path else self.manifest_path()
        with _locked(path):
            merged = {e.dest: e for e in _read(path)}
            for e in entries:
                merged[e.dest] = e
            _write(path, merged.values(), self.generated_by)

    def add(self, *, source, dest, command, section="", workflow="", note="") -> FigureEntry:
        """Register one figure (``dest`` relative to this set's folder), copy it, hash it.

        ``section`` defaults to the dest's first folder."""
        rel_src = _rel(source)
        if not _abs(rel_src).exists():
            raise FileNotFoundError(f"source does not exist: {_abs(rel_src)}")
        rel_dest = Path(self.name, dest).as_posix()
        section = section or (Path(dest).parts[0] if len(Path(dest).parts) > 1 else "")
        entry = FigureEntry(dest=rel_dest, source=rel_src, command=command, workflow=workflow,
                            section=section, note=note)
        self.save_manifest([entry])
        self.export_source(_abs(rel_src), entries=(entry,))
        self.refresh(only=rel_dest)
        return entry

    def remove_entries(self, dests) -> list[str]:
        """Drop the entries whose ``dest`` is in ``dests`` (the copies are left on disk)."""
        path = self.manifest_path()
        dests = set(dests)
        with _locked(path):
            entries = _read(path)
            _write(path, [e for e in entries if e.dest not in dests], self.generated_by)
        return sorted(dests & {e.dest for e in entries})

    def entries_for_source(self, source, entries=None) -> tuple[FigureEntry, ...]:
        """Manifest entries whose ``source`` is ``source`` (absolute or relative)."""
        rel = _rel(source)
        pool = self.load_manifest() if entries is None else tuple(entries)
        return tuple(e for e in pool if e.source == rel)

    def export_source(self, source, *, entries=None, quiet=False) -> list[Path]:
        """Copy (never re-save) a canonical artifact to its dest(s) in this set; no-op if unlisted."""
        matches = self.entries_for_source(source, entries)
        if not matches:
            return []
        src = Path(source)
        if not src.is_absolute():
            src = _abs(_rel(source))
        written: list[Path] = []
        for entry in matches:
            dest = _abs(entry.dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            if not quiet:
                print(f"  -> {self.name.lower()}: {entry.dest}")
            written.append(dest)
        return written

    def export_all(self, *, only=None) -> list[Path]:
        """Copy every manifest source to its dest, then :meth:`refresh` the hashes."""
        entries = self.load_manifest()
        written: list[Path] = []
        for entry in _selected(entries, only):
            src = _abs(entry.source)
            if not src.exists():
                print(f"  !! missing source, skipped: {entry.source}")
                continue
            written.extend(self.export_source(src, entries=(entry,)))
        self.refresh(only=only)
        return written

    def refresh(self, *, only=None) -> list[str]:
        """Recompute each dest's sha256 and write it back."""
        path = self.manifest_path()
        updated: list[str] = []
        with _locked(path):
            out = []
            for entry in _read(path):
                dest = _abs(entry.dest)
                if (only is not None and only not in entry.dest) or not dest.exists():
                    out.append(entry)
                    continue
                digest = sha256_file(dest)
                if digest != entry.sha256:
                    updated.append(entry.dest)
                out.append(replace(entry, sha256=digest))
            _write(path, out, self.generated_by)
        return updated

    def orphans(self) -> list[str]:
        """Files in ``figures/<name>/`` that no manifest entry lists (figures-root relative)."""
        folder = figures_root() / self.name
        listed = {e.dest for e in self.load_manifest()}
        if not folder.exists():
            return []
        return sorted(_rel(p) for p in folder.rglob("*")
                      if p.is_file() and p.name not in _NOT_FIGURES and _rel(p) not in listed)

    def verify(self, *, only=None, max_mb: float = 50.0) -> VerifyReport:
        """Check the set offline -- **no rendering**, only stats and hashes."""
        entries = _selected(self.load_manifest(), only)
        problems: list[str] = []
        if only is not None and not entries:
            problems.append(f"--only {only!r} matched no manifest entry "
                            f"(nothing was checked; fix the selector)")
        prefix = f"{self.name}/"
        for entry in entries:
            if not entry.dest.startswith(prefix):
                problems.append(f"{entry.dest}: dest is outside the {self.name}/ tree")
            parts = Path(entry.dest).parts
            if entry.section and (len(parts) < 3 or parts[1] != entry.section):
                problems.append(f"{entry.dest}: section {entry.section!r} is not the dest's "
                                f"first folder in {self.name}/")
            dest, src = _abs(entry.dest), _abs(entry.source)
            if not dest.exists():
                problems.append(f"{entry.dest}: dest missing")
                continue
            dest_hash = sha256_file(dest)
            if not src.exists():
                problems.append(f"{entry.dest}: source missing ({entry.source})")
            elif sha256_file(src) != dest_hash:
                problems.append(f"{entry.dest}: differs from source {entry.source} "
                                f"(re-export; the copy must be a copy2, not a re-render)")
            if entry.sha256 is None:
                problems.append(f"{entry.dest}: no recorded sha256 (run `refresh`)")
            elif entry.sha256 != dest_hash:
                problems.append(f"{entry.dest}: sha256 {dest_hash[:12]}… != manifest "
                                f"{entry.sha256[:12]}… (file changed since the last refresh)")
            size = dest.stat().st_size / 1e6
            if size > max_mb:
                problems.append(f"{entry.dest}: {size:.1f} MB exceeds the {max_mb:.0f} MB budget")
        if only is None:
            problems += [f"{p}: in the {self.name}/ folder but in no manifest entry (orphan; "
                         f"add it or delete it)" for p in self.orphans()]
        return VerifyReport(ok=not problems, checked=len(entries), problems=problems)


SETS: dict[str, FigureSet] = {"Manuscript": FigureSet("Manuscript"),
                              "Presentation": FigureSet("Presentation")}
"""Every figure set, by directory name."""


def get(name: str) -> FigureSet:
    """The set named ``name`` (case-insensitive); ``KeyError`` lists the known ones."""
    for key, figure_set in SETS.items():
        if key.lower() == name.lower():
            return figure_set
    raise KeyError(f"unknown figure set {name!r}; known: {', '.join(SETS)}")


def export_everywhere(source, *, quiet=True) -> list[Path]:
    """Copy ``source`` into every set that lists it; ``[]`` if none does. The producer hook."""
    written: list[Path] = []
    for figure_set in SETS.values():
        written.extend(figure_set.export_source(source, quiet=quiet))
    return written
