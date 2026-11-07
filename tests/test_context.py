import json
from pathlib import Path

import pytest

from src.context import build_story_bible, episode_metadata, load_hints, speaker_evidence, render_bible_compact, render_bible_markdown, validate_bible
from src.llm import MockProvider
from src.llm import LLMBudgetError
from src.models import Character, StoryBible
from src.parser import parse_file

FIXTURE = Path(__file__).parent / "fixtures" / "sample_episode.txt"


def test_budget_stop_preserves_partial_bible_evidence(tmp_path):
    from src.context import StoryOverview
    episode = parse_file(FIXTURE)
    overview = StoryOverview(title_guess="Test", premise="Anna confronts Victor.", main_conflicts=[], arcs=[])
    client = MockProvider(responses=[overview.model_dump_json()], max_api_calls=1,
                          cache_dir=tmp_path / "cache", raw_log_dir=tmp_path / "raw")
    client.limiter.tokens_per_minute = 1500
    client.limiter.wait = lambda *args: None
    output = tmp_path / "output"
    with pytest.raises(LLMBudgetError):
        build_story_bible([episode], client, output)
    progress = json.loads((output / "story_bible_progress.json").read_text())
    assert progress["completed_evidence_batches"] == 0
    assert progress["overview"]["title_guess"] == "Test"
    assert progress["episode_numbers"] == [99]
    assert len(client.calls) == 1


def synthetic_bible():
    return StoryBible(
        title_guess="The Test Heiress", premise="Anna confronts Victor. He has always been there.",
        characters=[
            Character(name="Anna", aliases=["ANNA"], role="Test heiress", relationships="Confronts Victor",confidence="high",evidence="E099_L004"),
            Character(name="Victor", aliases=["VICTOR"], role="Test villain", relationships="Confronted by Anna",confidence="high",evidence="E099_L006"),
        ], main_conflicts=["Anna confronts Victor"], arcs=[],
    )


def test_story_bible_metadata_and_outputs(tmp_path):
    episode = parse_file(FIXTURE)
    bible = synthetic_bible()
    client = MockProvider(
        responses=[bible.model_dump_json()], cache_dir=tmp_path / "cache",
        raw_log_dir=tmp_path / "raw", requests_per_minute=60,
    )
    result = build_story_bible([episode], client, tmp_path / "outputs")
    assert result == bible
    assert len(client.calls) == 1
    metadata = json.loads(client.calls[0].user.split("\n\nReturn ONLY")[0])
    assert metadata["all_speaker_labels"] == ["ANNA", "VICTOR"]
    assert metadata["episodes"][0]["speakers"] == ["ANNA", "VICTOR"]
    assert metadata["episodes"][0]["logline"] == episode.logline
    assert "Wide shot of the hall." in episode_metadata([episode])
    assert metadata["speaker_evidence"][0]["samples"][0]["line_id"] == "E099_L004"
    assert (tmp_path / "outputs" / "story_bible.json").exists()
    assert "Relationships: Confronts Victor" in (tmp_path / "outputs" / "story_bible.md").read_text(encoding="utf-8")
    assert not validate_bible(bible, [episode])
    bible.characters[0].aliases = []
    assert "ANNA" in validate_bible(bible, [episode])[0]


def test_author_identity_hints_reject_incorrect_character_merges():
    episode = parse_file(FIXTURE)
    bible = synthetic_bible()
    assert validate_bible(bible, [episode], {"ANNA": "Anna"}) == []
    errors = validate_bible(bible, [episode], {"ANNA": "Victor"})
    assert "alias_identity" in errors[0]
    assert "Victor" in errors[0]
    assert json.loads(episode_metadata([episode], {"ANNA": "Anna"}))["identity_hints"] == {"ANNA": "Anna"}


def test_compact_bible_has_bounded_size_and_main_aliases():
    bible = synthetic_bible()
    assert "ANNA" in render_bible_compact(bible)
    bible.premise *= 100
    bible.characters *= 100
    bible.main_conflicts *= 100
    compact = render_bible_compact(bible)
    assert len(compact) <= 1800
    assert "Conflicts:" in compact
    assert "## Characters" in render_bible_markdown(bible)


def test_hints_are_empty_unless_explicit_file_is_supplied(tmp_path):
    assert load_hints() == {}
    path=tmp_path/'hints.yaml'
    path.write_text('alias_hints:\n  ANNA: Anna\n',encoding='utf-8')
    assert load_hints(path)=={'ANNA':'Anna'}


def test_samples_include_action_and_both_neighbour_cameras():
    episode=parse_file(FIXTURE)
    records={item['speaker']:item for item in speaker_evidence([episode])}
    anna=records['ANNA']['samples'][0]
    assert anna['line_id']=='E099_L004'
    assert [shot['shot_id'] for shot in anna['shots']]==['E099_S001','E099_S002','E099_S003']
    assert anna['shots'][0]['actions'][0]['text']=='Anna walks in slowly.'
    assert len(records['VICTOR']['samples'])==2
    assert len(records['ANNA']['samples'])==1  # Only one source utterance exists.


def test_minor_recurring_speaker_and_invented_name_are_rejected():
    episode=parse_file(FIXTURE)
    bible=synthetic_bible()
    bible.characters[0].name='Anna Inventedsurname'
    assert any('unsupported_name' in error for error in validate_bible(bible,[episode]))
    bible=synthetic_bible()
    bible.characters=[]
    bible.minor_characters=['ANNA','VICTOR']
    assert validate_bible(bible,[episode])==[]
    second=episode.model_copy(update={'number':100})
    assert any('main_character_required' in error for error in validate_bible(bible,[episode,second]))
    bible=synthetic_bible()
    bible.characters[0].name='Anna Victor'  # Both words exist, but the combined name does not.
    assert any('unsupported_name' in error for error in validate_bible(bible,[episode]))


def test_batched_bible_uses_no_hints_and_covers_all_samples(tmp_path):
    episode=parse_file(FIXTURE)
    bible=synthetic_bible()
    overview={'title_guess':bible.title_guess,'premise':bible.premise,'main_conflicts':bible.main_conflicts,'arcs':[]}
    identities={'speakers':[{'label':name,'name':name.title(),'confidence':'high','evidence':'E099_L004'} for name in ('ANNA','VICTOR')]}
    client=MockProvider(responses=[json.dumps(overview),json.dumps(identities),bible.model_dump_json()],raw_log_dir=tmp_path/'raw',cache_dir=tmp_path/'cache',tokens_per_minute=2000)
    client.limiter.wait=lambda *args:None
    result=build_story_bible([episode],client,tmp_path/'out')
    assert result.title_guess==bible.title_guess
    assert result.minor_characters==['ANNA','VICTOR']  # One episode and not named in its logline.
    assert len(client.calls)==2
    payload=json.loads(client.calls[1].user.split('\n\nReturn ONLY')[0])
    assert payload['identity_hints']=={}
    assert {item['speaker'] for item in payload['speakers']}=={'ANNA','VICTOR'}


def test_generic_role_in_logline_is_not_a_named_character():
    episode=parse_file(FIXTURE)
    for line in episode.lines:
        if line.speaker=='ANNA':line.speaker='LORD'
    episode.logline='A noble lord confronts a test villain.'
    bible=synthetic_bible()
    bible.characters=[]
    bible.minor_characters=['LORD','VICTOR']
    assert validate_bible(bible,[episode])==[]
