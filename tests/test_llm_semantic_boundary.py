"""Regressions for Gemini-selected group-chat boundaries."""
import json
from datetime import timedelta
from types import SimpleNamespace

from tle.cogs import _llm_context as llm_context
from tle.cogs import _llm_history as llm_history
from tle.cogs import _llm_pipeline as llm_pipeline
from tle.util import gemini_api
from tle.util.llm_keypool import Lease
from tests.llm_test_utils import run
from tests.test_llm_history import (
    FakeGatherCtx, FakeHistoryChannel, HistMessage, _BASE,
)


def _message(message_id, content, offset, *, author_id=1, reply_to=None,
             resolved=None):
    message = HistMessage(
        content=content, offset=offset, author_id=author_id)
    message.id = message_id
    if reply_to is not None or resolved is not None:
        message.reference = SimpleNamespace(
            message_id=reply_to, resolved=resolved)
    return message


class FetchableHistoryChannel(FakeHistoryChannel):
    def __init__(self, messages):
        super().__init__(messages)
        self.id = 77
        self.guild = SimpleNamespace(id=9)
        self.fetches = []
        self.by_id = {message.id: message for message in messages}
        for message in messages:
            message.channel = self
            message.guild = self.guild

    async def fetch_message(self, message_id):
        self.fetches.append(message_id)
        return self.by_id[message_id]


class TestCandidateCollection:
    def test_no_time_gap_cuts_the_candidate_stream(self):
        older = _message(1, '7:10 message', 0)
        newer = _message(2, '8:00 message', 50 * 60)
        command = _message(3, '@grok summarize this', 51 * 60)
        channel = FetchableHistoryChannel([older, newer, command])

        got = run(llm_history.collect_candidates(
            channel, before=command, limit=200))

        assert [message.id for message in got] == [1, 2]

    def test_old_reply_ancestors_are_fetched_and_pinned(self):
        parent = _message(1, 'root topic', -20_000)
        target = _message(
            2, 'old reply target', -10_000, reply_to=1)
        recent = [
            _message(100 + index, f'later {index}', index)
            for index in range(205)
        ]
        command = _message(999, '@grok why?', 1_000)
        channel = FetchableHistoryChannel(
            [parent, target, *recent, command])
        ctx = FakeGatherCtx(channel, command)

        got = run(llm_pipeline.gather_candidates(
            ctx, target, message_limit=200))

        assert len(got) == 200
        assert [message.id for message in got[:2]] == [1, 2]
        assert got[-1].id == 304
        assert channel.fetches == [1]


class TestBoundarySelection:
    def test_structured_selector_uses_cheapest_gemini_model(
            self, monkeypatch):
        seen = {}

        async def complete(pool, prompt, **kwargs):
            seen['prompt'] = prompt
            seen.update(kwargs)
            return '{"start_index": 1}', Lease(
                1, 'key', 'label', 'cheap')

        monkeypatch.setattr(gemini_api, 'complete', complete)
        pool = SimpleNamespace(models=['cheap', 'expensive'])
        candidates = [
            _message(1, 'old topic', 0),
            _message(2, 'new topic starts', 10),
            _message(3, 'new topic continues', 20),
        ]

        start = run(llm_pipeline.select_boundary(
            pool, 'summarize this', candidates, force_context=True,
            author_id=42, sent_at=_BASE + timedelta(seconds=30)))

        assert start == 1
        assert seen['models'] == ['cheap']
        assert seen['response_mime_type'] == 'application/json'
        assert seen['response_schema']['required'] == ['start_index']
        assert 'force_context: yes' in seen['prompt']
        records = [
            json.loads(line) for line in
            seen['prompt'].split('--- BEGIN CANDIDATE MESSAGES ---\n', 1)[1]
            .split('\n--- END CANDIDATE MESSAGES ---', 1)[0]
            .splitlines()
        ]
        assert [record['index'] for record in records] == [0, 1, 2]

    def test_reply_boundary_cannot_drop_available_ancestors(self):
        parent = _message(1, 'root', 0)
        target = _message(
            2, 'focus', 10, reply_to=1, resolved=parent)
        later = _message(3, 'later noise', 20)
        candidates = [parent, target, later]

        assert llm_pipeline.apply_boundary(
            candidates, 2, referenced=target) == candidates

    def test_invalid_or_failed_selection_forwards_all_candidates(
            self, monkeypatch):
        candidates = [_message(1, 'one', 0), _message(2, 'two', 10)]
        assert llm_pipeline.parse_boundary('not json', len(candidates)) == 0

        async def fail(*args, **kwargs):
            raise gemini_api.NoCapacityError('quota')

        monkeypatch.setattr(gemini_api, 'complete', fail)
        pool = SimpleNamespace(models=['cheap'])
        assert run(llm_pipeline.select_boundary(
            pool, 'summarize this', candidates)) == 0


def test_answer_prompt_warns_about_interleaved_topics():
    prompt = llm_context.build_context_prompt(
        'summarize this', '{"index":0,"content":"hello"}')
    assert 'topics can still interleave' in prompt
    assert 'clearly unrelated' in prompt
