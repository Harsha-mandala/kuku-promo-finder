import io
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.ui.data import DEMO_ROOT, PROJECT_ROOT
from src.ui.results_upload import load_results_zip


def results_zip(files):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


def test_load_small_synthetic_results_zip(tmp_path):
    demo = DEMO_ROOT / "precomputed"
    files = {
        "raw_outputs/parsed/episode_001.json": (demo / "episodes/episode_001.json").read_bytes(),
        "raw_outputs/final_fixture/promos_ranked.json": "[]",
        "raw_outputs/final_fixture/report.md": "# Synthetic results",
        "raw_outputs/llm_raw/ignored.json": "{}",
    }
    bundle = load_results_zip(results_zip(files), tmp_path)
    assert bundle["tag"] == "fixture"
    assert [episode.number for episode in bundle["episodes"]] == [1]
    assert not (tmp_path / "llm_raw").exists()


@pytest.mark.parametrize("name", ["raw_outputs/../escape.json", "raw_outputs\\..\\escape.json"])
def test_reject_results_zip_traversal(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe path"):
        load_results_zip(results_zip({name: "{}"}), tmp_path)
    assert not list(tmp_path.iterdir())


def test_reject_results_zip_without_parsed_episodes(tmp_path):
    files = {"final_fixture/promos_ranked.json": "[]", "final_fixture/report.md": "# Synthetic"}
    with pytest.raises(ValueError, match="missing parsed"):
        load_results_zip(results_zip(files), tmp_path)


def test_reject_malformed_rankings(tmp_path):
    files = {"parsed/episode_001.json": "{}", "final_fixture/report.md": "# Synthetic",
             "final_fixture/promos_ranked.json": "[null]"}
    with pytest.raises(ValueError, match="malformed JSON in final_fixture/promos_ranked.json"):
        load_results_zip(results_zip(files), tmp_path)


def test_load_private_submission_when_present(tmp_path):
    path = Path(PROJECT_ROOT) / "submission/raw_outputs.zip"
    if not path.exists():
        pytest.skip("Private submission is not distributed in CI")
    bundle = load_results_zip(path.read_bytes(), tmp_path)
    assert len(bundle["episodes"]) == 50
    assert len(bundle["ranked"]) == 111
    assert len(bundle["trailers"]) == 3
    assert len(bundle["editor_order"]) == 18
    assert bundle["safety_queue"] and bundle["bible"]
    assert (tmp_path / "eval_gemini_full/metrics.json").is_file()
    assert (tmp_path / "llm_usage.csv").is_file()


def test_public_upload_reads_quality_and_clear_deletes_workspace(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.ui.results_upload import MODE

    demo = DEMO_ROOT / "precomputed"
    data = results_zip({
        "parsed/episode_001.json": (demo / "episodes/episode_001.json").read_bytes(),
        "final_fixture/promos_ranked.json": "[]",
        "final_fixture/report.md": "# Synthetic",
        "eval_fixture/metrics.json": (demo / "results/eval_demo/metrics.json").read_bytes(),
    })
    monkeypatch.setenv("PUBLIC_DEPLOY", "1")
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value(MODE).run()
    uploader = next(item for item in app.file_uploader if item.label == "Results ZIP")
    uploader.set_value(("results.zip", data, "application/zip")).run()
    assert not app.exception
    assert any("Judge versus human" in item.value for item in app.markdown)
    workspace = Path(app.session_state["results_workspace"].name)
    assert workspace.is_dir()
    next(button for button in app.button if button.label == "Clear session").click().run()
    assert not workspace.exists()
