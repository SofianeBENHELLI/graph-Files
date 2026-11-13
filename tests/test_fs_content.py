"""Content coverage added for graphify fs: .pptx decks, HEIC photos, iCloud placeholders."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from graphify import detect as detect_mod
from graphify.detect import FileType, classify_file, convert_office_file, count_words, detect, pptx_to_markdown

pptx = pytest.importorskip("pptx")


def _make_deck(path: Path) -> Path:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Quarterly Roadmap"
    s1.placeholders[1].text = "Ship the knowledge graph explorer"
    s1.notes_slide.notes_text_frame.text = "Mention Finder tags"
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Budget"
    table = s2.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(4), Inches(1)).table
    table.cell(0, 0).text = "Item"
    table.cell(0, 1).text = "Cost"
    table.cell(1, 0).text = "GPU"
    table.cell(1, 1).text = "1|2"
    prs.save(path)
    return path


def test_pptx_to_markdown_keeps_slides_tables_and_notes(tmp_path):
    md = pptx_to_markdown(_make_deck(tmp_path / "deck.pptx"))
    assert "## Slide 1: Quarterly Roadmap" in md
    assert "Ship the knowledge graph explorer" in md
    assert "> Speaker notes: Mention Finder tags" in md
    assert "## Slide 2: Budget" in md
    assert "| GPU | 1\\|2 |" in md


def test_pptx_is_a_document_and_converts_to_sidecar(tmp_path):
    deck = _make_deck(tmp_path / "deck.pptx")
    assert classify_file(deck) == FileType.DOCUMENT
    assert count_words(deck) > 5
    sidecar = convert_office_file(deck, tmp_path / "graphify-out" / "converted", root=tmp_path)
    assert sidecar is not None and sidecar.suffix == ".md"
    assert "Quarterly Roadmap" in sidecar.read_text()


def test_detect_includes_pptx_sidecar(tmp_path):
    _make_deck(tmp_path / "deck.pptx")
    result = detect(tmp_path)
    docs = result["files"]["document"]
    assert any(Path(p).name.startswith("deck_") and p.endswith(".md") for p in docs)


def test_corrupt_pptx_returns_empty(tmp_path):
    bad = tmp_path / "bad.pptx"
    bad.write_bytes(b"not a zip")
    assert pptx_to_markdown(bad) == ""


def test_heic_is_an_image():
    assert classify_file(Path("photo.HEIC")) == FileType.IMAGE
    assert classify_file(Path("photo.heif")) == FileType.IMAGE


@pytest.mark.skipif(shutil.which("sips") is None, reason="needs macOS sips")
def test_heic_is_sent_to_vision_as_jpeg(tmp_path):
    from PIL import Image
    from graphify.llm import _build_image_refs

    png = tmp_path / "a.png"
    Image.new("RGB", (32, 32), (10, 120, 200)).save(png)
    heic = tmp_path / "a.heic"
    subprocess.run(["sips", "-s", "format", "heic", str(png), "--out", str(heic)], check=True, capture_output=True)
    (ref,) = _build_image_refs([heic], tmp_path)
    assert ref.media_type == "image/jpeg"
    assert ref.raw is not None and ref.raw[:3] == b"\xff\xd8\xff"


def _fake_dataless(monkeypatch, names: set[str]):
    real_stat = os.stat

    class _St:
        def __init__(self, st):
            self._st = st

        def __getattr__(self, item):
            return getattr(self._st, item)

        st_flags = detect_mod.SF_DATALESS

    def fake_stat(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if Path(path).name in names:
            return _St(st)
        return st

    monkeypatch.setattr(detect_mod.os, "stat", fake_stat)


def test_icloud_placeholder_is_skipped_by_default(tmp_path, monkeypatch):
    (tmp_path / "online.md").write_text("# Online\n" + "word " * 20)
    (tmp_path / "cloud.md").write_text("# Cloud\n" + "word " * 20)
    _fake_dataless(monkeypatch, {"cloud.md"})
    monkeypatch.delenv("GRAPHIFY_DOWNLOAD_ICLOUD", raising=False)
    result = detect(tmp_path)
    docs = [Path(p).name for p in result["files"]["document"]]
    assert "online.md" in docs and "cloud.md" not in docs
    assert any("iCloud placeholder" in s for s in result["skipped_sensitive"])


def test_icloud_placeholder_opt_in(tmp_path, monkeypatch):
    (tmp_path / "cloud.md").write_text("# Cloud\n" + "word " * 20)
    _fake_dataless(monkeypatch, {"cloud.md"})
    monkeypatch.setenv("GRAPHIFY_DOWNLOAD_ICLOUD", "1")
    result = detect(tmp_path)
    assert "cloud.md" in [Path(p).name for p in result["files"]["document"]]


def test_out_dir_sidecars_are_readable_by_semantic_pass(tmp_path):
    """`extract --out DIR` writes Office sidecars outside the corpus root; they must still be read."""
    from graphify import paths
    from graphify.llm import _read_files, _resolve_under_root

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _make_deck(corpus / "deck.pptx")
    out = tmp_path / "index"
    result = detect(corpus, cache_root=out)
    sidecar = next(Path(p) for p in result["files"]["document"] if p.endswith(".md"))
    assert not sidecar.resolve().is_relative_to(corpus.resolve())
    assert _resolve_under_root(sidecar, corpus) is not None
    assert "Quarterly Roadmap" in _read_files([sidecar], corpus)
    # Anything else outside the root stays refused.
    stray = tmp_path / "stray.md"
    stray.write_text("secret")
    assert _resolve_under_root(stray, corpus) is None
    assert paths.in_sidecar_root(sidecar.resolve())
