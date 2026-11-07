"""Deterministic, source-grounded test responses for arbitrary upload packs."""

import hashlib
import re

from .baselines import build_baseline
from .context import load_episodes
from .editor_pass import EditorChoice, EditorOrder
from .judge import JudgeBatch, JudgeScore
from .llm import LLMRequestError, MockProvider, Response
from .models import Candidate, CandidateBatch, CandidateList, Character, StoryBible, Trailer, TrailerBatch, TrailerSegment
from .trailers import TrailerJudgeBatch, TrailerJudgeScore, validate_trailer


def mock_score(identifier):
    digits = hashlib.sha256(identifier.encode()).digest()
    return JudgeScore(promo_id=identifier, hook_score=2+digits[0]%4, cliffhanger_score=2+digits[1]%4,
                      standalone_score=2+digits[2]%4, emotion_score=2+digits[3]%4, post_priority=4+digits[4]%7,
                      ad_safety='safe', ad_safety_reason='', weakest_point='Test score only.',
                      one_line_pitch='Mock preview.', why_it_works='Deterministic test response with fake scores.')


class UploadMockProvider(MockProvider):
    def __init__(self, parsed_dir, **kwargs):
        super().__init__(**kwargs)
        self.parsed_dir = parsed_dir
        self.limiter.wait = lambda *_: None

    def _send(self, request):
        self.calls.append(request)
        schema = request.schema
        if schema in (JudgeBatch, TrailerJudgeBatch, EditorOrder):
            result = self._review(schema, request.user)
        else:
            episodes = {episode.number: episode for episode in load_episodes(self.parsed_dir)}
            if schema is StoryBible:
                evidence = {line.speaker: line.id for episode in episodes.values() for line in episode.lines if line.speaker}
                characters = [Character(name=name, aliases=[name], role='Observed speaker', relationships='Mock test run',
                                        confidence='low', evidence=source) for name, source in sorted(evidence.items())]
                result = StoryBible(title_guess='Mock upload', premise='Upload pipeline test', characters=characters,
                                    main_conflicts=[], arcs=[])
            elif schema is CandidateBatch:
                requested = re.search(r'for each episode: ([\d, ]+)\.', request.user)
                if not requested:
                    raise LLMRequestError('Mock request did not identify the batch episodes')
                result = CandidateBatch([self._candidates(episodes[int(number)]) for number in requested[1].split(',')])
            elif schema is TrailerBatch:
                result = self._trailer(episodes, request.user)
            else:
                raise LLMRequestError(f'Mock uploads do not support schema {schema}')
        return Response(result.model_dump_json())

    @staticmethod
    def _candidates(episode):
        candidates = [Candidate(start_shot_id=promo.start_shot_id, end_shot_id=promo.end_shot_id,
                                hook_line_id=promo.hook_line_id, cliffhanger_line_id=promo.cliffhanger_line_id,
                                hook_type='other', cliffhanger_type='other', context_needed='',
                                why_it_works='Rules-only window for the mock upload test.', self_score=3)
                      for promo in build_baseline(episode, 'rules_only')]
        return CandidateList(episode=episode.number, candidates=candidates)

    @staticmethod
    def _review(schema, user):
        if schema is EditorOrder:
            identifiers = re.findall(r'^PROMO_ID: (\S+)', user, re.MULTILINE)
            return EditorOrder([EditorChoice(promo_id=key, reason='Deterministic mock posting order.') for key in sorted(identifiers)])
        if schema is TrailerJudgeBatch:
            identifiers = re.findall(r'^TRAILER (T\d+) \|', user, re.MULTILINE)
            return TrailerJudgeBatch([TrailerJudgeScore(**mock_score(key).model_dump(), coherence=3) for key in identifiers])
        identifiers = re.findall(r'^CLIP (E\d+_C\d+) \|', user, re.MULTILINE)
        return JudgeBatch([mock_score(key) for key in identifiers])

    @staticmethod
    def _trailer(episodes, user):
        allowed = set(re.findall(r'\[(E\d+_S\d+)\]', user))
        pieces = []
        for episode in episodes.values():
            for index, first in enumerate(episode.shots):
                if first.id not in allowed or not first.clean_start:
                    continue
                hooks = [line for line in first.lines if line.type in ('dialogue', 'action')
                         and 0 <= line.start-first.effective_start <= 3]
                if not hooks:
                    continue
                for last in episode.shots[index:]:
                    if last.id not in allowed or last.effective_end-first.effective_start > 25:
                        break
                    cliffs = [line for line in last.lines if line.type in ('dialogue', 'action')]
                    if last.clean_end and cliffs and last.effective_end-first.effective_start >= 10:
                        pieces.append((episode.number, TrailerSegment(start_shot_id=first.id, end_shot_id=last.id),
                                       hooks[0].id, cliffs[-1].id))
                        break
        for episode, first, hook, _ in pieces:
            for other, last, _, cliff in pieces:
                if episode >= other:
                    continue
                trailer = Trailer(segments=[first, last], hook_line_id=hook, cliffhanger_line_id=cliff,
                                  story='Two source excerpts for the upload test.', why_it_works='Mock trailer.',
                                  spoiler_risk='low', self_score=3)
                if validate_trailer(trailer, episodes, 1, allowed).status != 'rejected':
                    return TrailerBatch([trailer])
        return TrailerBatch([])
