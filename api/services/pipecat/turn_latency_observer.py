"""Per-turn latency probe for the low-latency Silero + Smart Turn experiment.

Timestamps six events on every user turn, then logs the deltas once the
first TTS audio starts:

    t0  VAD reported user stopped speaking     (VADUserStoppedSpeakingFrame)
    t1  Smart Turn V3 published a decision      (MetricsFrame w/ TurnMetricsData)
    t2  User turn committed downstream          (UserStoppedSpeakingFrame)
    t3  LLM context frame handed to Gemini      (LLMContextFrame, first arrival
                                                  either direction)
    t4  Gemini started its response             (LLMFullResponseStartFrame)
    t5  Cartesia started TTS playback           (TTSStartedFrame)

The observer is passive — it never mutates frames, only stamps their arrival
time. Enabled per workflow via ``enable_turn_latency_logs`` in workflow
config.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TTSStartedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed


@dataclass
class _TurnStamps:
    turn_index: int
    vad_stop: float | None = None
    smart_turn: float | None = None
    smart_turn_prediction: bool | None = None
    smart_turn_probability: float | None = None
    smart_turn_inference_ms: float | None = None
    user_turn_committed: float | None = None
    llm_request: float | None = None
    llm_first_response: float | None = None
    tts_started: float | None = None
    logged: bool = False


class TurnLatencyObserver(BaseObserver):
    """Observer stamping the end-of-turn → first-audio latency breakdown."""

    def __init__(self, *, workflow_run_id: int, vad_stop_secs: float):
        super().__init__()
        self._workflow_run_id = workflow_run_id
        self._vad_stop_secs = vad_stop_secs
        self._turn_index = 0
        self._current: _TurnStamps | None = None
        # Frames are observed by every push; dedupe by object id so a fan-out
        # doesn't overwrite an earlier (truer) timestamp for the same frame.
        self._seen: set[int] = set()

    async def cleanup(self):
        pass

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        fid = id(frame)
        if fid in self._seen:
            return
        self._seen.add(fid)

        now = time.perf_counter()

        if isinstance(frame, UserStartedSpeakingFrame):
            if self._current and not self._current.logged:
                # Previous turn never made it to TTS (interruption, hang-up).
                self._flush(reason="turn_superseded")
            self._turn_index += 1
            self._current = _TurnStamps(turn_index=self._turn_index)
            return

        if self._current is None:
            return

        stamps = self._current

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # Multiple VAD stops can fire per turn when Smart Turn returns
            # INCOMPLETE. Keep the *last* one — that's the stop that led to
            # a successful classification.
            stamps.vad_stop = now
            return

        if isinstance(frame, MetricsFrame):
            for item in frame.data or []:
                if isinstance(item, TurnMetricsData):
                    stamps.smart_turn = now
                    stamps.smart_turn_prediction = item.is_complete
                    stamps.smart_turn_probability = item.probability
                    stamps.smart_turn_inference_ms = item.e2e_processing_time_ms
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            stamps.user_turn_committed = now
            return

        if isinstance(frame, LLMContextFrame):
            # First arrival wins, either direction. See design note in the
            # experiment brief — a loose stamp is fine, missing it is not.
            if stamps.llm_request is None:
                stamps.llm_request = now
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            if stamps.llm_first_response is None:
                stamps.llm_first_response = now
            return

        if isinstance(frame, TTSStartedFrame):
            if stamps.tts_started is None:
                stamps.tts_started = now
                self._flush(reason="tts_started")
            return

    def _flush(self, *, reason: str):
        s = self._current
        if s is None or s.logged:
            return
        s.logged = True

        def ms(a: float | None, b: float | None) -> str:
            if a is None or b is None:
                return "—"
            return f"{(b - a) * 1000:.1f}ms"

        logger.info(
            "[turn-latency run={run} turn={turn} reason={reason} "
            "vad_stop_cfg={cfg:.3f}s] "
            "vad→smart={a} smart→commit={b} commit→llm={c} "
            "llm→first_resp={d} first_resp→tts={e} "
            "total_stop→tts={total} | smart_turn(pred={pred}, p={prob}, "
            "inf={inf}ms)",
            run=self._workflow_run_id,
            turn=s.turn_index,
            reason=reason,
            cfg=self._vad_stop_secs,
            a=ms(s.vad_stop, s.smart_turn),
            b=ms(s.smart_turn, s.user_turn_committed),
            c=ms(s.user_turn_committed, s.llm_request),
            d=ms(s.llm_request, s.llm_first_response),
            e=ms(s.llm_first_response, s.tts_started),
            total=ms(s.vad_stop, s.tts_started),
            pred=s.smart_turn_prediction,
            prob=(
                f"{s.smart_turn_probability:.3f}"
                if s.smart_turn_probability is not None
                else "—"
            ),
            inf=(
                f"{s.smart_turn_inference_ms:.1f}"
                if s.smart_turn_inference_ms is not None
                else "—"
            ),
        )
