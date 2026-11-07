"""Checks of figure_sets.py (named figure sets, one manifest per set) and of style.save's hook.

    python scripts/tests/test_figure_sets.py      (or pytest)

Adapted from AntarcticIceMeltFracture's Scripts/Core/Tests/test_figure_sets.py; every test runs in
a temporary $FIGURES_ROOT, never in the real figures/ tree.
"""
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import figure_sets as FS                                             # noqa: E402

SOURCE = "slides/png/A.png"
PRES = FS.FigureEntry(dest="Presentation/ExchangeabilityMaps/png/A.png", source=SOURCE,
                      command="python scripts/x.py", workflow="x", section="ExchangeabilityMaps")
MANU = FS.FigureEntry(dest="Manuscript/MainText/png/A.png", source=SOURCE,
                      command="python scripts/x.py", workflow="x", section="MainText")


@contextmanager
def figures_root():
    previous = os.environ.get("FIGURES_ROOT")
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCRATCH")) as tmp:
        os.environ["FIGURES_ROOT"] = tmp
        try:
            root = Path(tmp)
            (root / SOURCE).parent.mkdir(parents=True)
            (root / SOURCE).write_bytes(b"png-bytes" * 50)
            yield root
        finally:
            if previous is None:
                os.environ.pop("FIGURES_ROOT", None)
            else:
                os.environ["FIGURES_ROOT"] = previous


def test_sets_have_separate_manifests():
    with figures_root() as root:
        pres, manu = FS.get("Presentation"), FS.get("Manuscript")
        assert pres.manifest_path() == root / "Presentation" / "manifest.json"
        pres.save_manifest([PRES])
        assert pres.load_manifest() == (PRES,)
        assert manu.load_manifest() == ()
        assert not manu.manifest_path().exists()
        doc = json.loads(pres.manifest_path().read_text())
        assert doc["schema"] == 1 and doc["generated_by"] == "scripts/figure_sets.py"


def test_get_is_case_insensitive_and_rejects_unknown():
    assert FS.get("presentation") is FS.SETS["Presentation"]
    try:
        FS.get("Poster")
    except KeyError as exc:
        assert "Presentation" in str(exc)
    else:
        raise AssertionError("unknown set accepted")


def test_save_merges_by_dest():
    with figures_root():
        pres = FS.get("Presentation")
        other = FS.FigureEntry(dest="Presentation/ExchangeabilityMaps/png/B.png", source=SOURCE,
                               command="y")
        pres.save_manifest([PRES])
        pres.save_manifest([other])
        assert pres.load_manifest() == (PRES, other)


def test_export_everywhere_copies_into_every_listing_set():
    with figures_root() as root:
        FS.get("Presentation").save_manifest([PRES])
        FS.get("Manuscript").save_manifest([MANU])
        written = FS.export_everywhere(root / SOURCE)
        assert sorted(written) == sorted([root / PRES.dest, root / MANU.dest])
        assert (root / PRES.dest).read_bytes() == (root / SOURCE).read_bytes()


def test_export_everywhere_unlisted_is_noop():
    with figures_root() as root:
        assert FS.export_everywhere(root / "supp-info/png/other.png") == []


def test_add_copies_hashes_and_sets_section():
    with figures_root() as root:
        pres = FS.get("Presentation")
        e = pres.add(source=root / SOURCE, dest="ExchangeabilityMaps/png/A.png", command="x")
        assert e.dest == PRES.dest and e.section == "ExchangeabilityMaps"
        assert pres.load_manifest()[0].sha256 == FS.sha256_file(root / SOURCE)
        assert pres.verify().ok


def test_refresh_and_verify_one_set_leave_the_other_alone():
    with figures_root() as root:
        pres, manu = FS.get("Presentation"), FS.get("Manuscript")
        pres.save_manifest([PRES])
        manu.save_manifest([MANU])
        FS.export_everywhere(root / SOURCE)
        pres.refresh()
        assert pres.verify().ok
        assert not manu.verify().ok                                  # never hashed
        assert pres.verify().ok


def test_verify_flags_changed_and_missing_files():
    with figures_root() as root:
        pres = FS.get("Presentation")
        pres.add(source=root / SOURCE, dest="ExchangeabilityMaps/png/A.png", command="x")
        (root / SOURCE).write_bytes(b"re-rendered")                  # source changed, not exported
        report = pres.verify()
        assert any("differs from source" in p for p in report.problems)
        FS.export_everywhere(root / SOURCE)                          # copied, hash not refreshed
        assert any("file changed since the last refresh" in p for p in pres.verify().problems)
        pres.refresh()
        assert pres.verify().ok
        (root / PRES.dest).unlink()
        assert any("dest missing" in p for p in pres.verify().problems)


def test_verify_flags_dest_outside_the_set_tree_and_wrong_section():
    with figures_root() as root:
        pres = FS.get("Presentation")
        stray = FS.FigureEntry(dest="Elsewhere/A.png", source=SOURCE, command="x")
        wrong = FS.FigureEntry(dest="Presentation/Other/png/A.png", source=SOURCE, command="x",
                               section="ExchangeabilityMaps")
        pres.save_manifest([stray, wrong])
        pres.export_all()
        problems = pres.verify().problems
        assert any("outside the Presentation/ tree" in p for p in problems)
        assert any("is not the dest's first folder" in p for p in problems)
        assert (root / "Elsewhere/A.png").exists()


def test_verify_flags_orphans():
    with figures_root() as root:
        pres = FS.get("Presentation")
        pres.add(source=root / SOURCE, dest="ExchangeabilityMaps/png/A.png", command="x")
        (root / "Presentation/ExchangeabilityMaps/png/old.png").write_bytes(b"left behind")
        problems = pres.verify().problems
        assert any("old.png" in p and "orphan" in p for p in problems)
        assert pres.verify(only="A.png").ok                          # --only skips the sweep


def test_verify_only_matching_nothing_fails():
    with figures_root():
        pres = FS.get("Presentation")
        pres.save_manifest([PRES])
        assert not pres.verify(only="nothing-like-this").ok


def test_remove_entries_drops_only_named_dests():
    with figures_root():
        pres = FS.get("Presentation")
        other = FS.FigureEntry(dest="Presentation/ExchangeabilityMaps/png/B.png", source=SOURCE,
                               command="y")
        pres.save_manifest([PRES, other])
        assert pres.remove_entries([PRES.dest, "Presentation/missing.png"]) == [PRES.dest]
        assert pres.load_manifest() == (other,)


def test_style_save_is_deterministic_and_exports():
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("SCRATCH", "/tmp"),
                                                       ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import style
    with figures_root() as root:
        FS.get("Presentation").save_manifest([FS.FigureEntry(
            dest="Presentation/ExchangeabilityMaps/pdf/t.pdf", source="slides/pdf/t.pdf",
            command="x", section="ExchangeabilityMaps")])
        digests = []
        for _ in range(2):
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.plot([0, 1], [0, 1])
            style.save(fig, "t", dpi=50, folder="slides")
            plt.close(fig)
            digests.append(FS.sha256_file(root / "slides/pdf/t.pdf"))
        assert digests[0] == digests[1], "re-rendering the same figure changed the pdf bytes"
        assert (root / "Presentation/ExchangeabilityMaps/pdf/t.pdf").read_bytes() == \
            (root / "slides/pdf/t.pdf").read_bytes()
        assert (root / "slides/png/t.png").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(name, "ok")
