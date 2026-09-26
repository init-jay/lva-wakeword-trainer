"""What a TTS engine is, and the one batch algorithm the timestamp engines share.

An Engine is a way to turn (voice, text, speed) into 16 kHz mono int16 audio.
The server side of this repo is Engine implementations in tts-service/engines/,
one uv project per engine; the client side is TtsClient (client.py), which
also conforms to this interface so the same corpus code drives an in-process
engine and a remote one identically.

    engine.available()                          (bool, reason); never raises
    voices = engine.voices(max_speakers=...)    engine-specific shape: Kokoro
                                               voice ids, (voice, speaker)
                                               pairs for Piper
    audio, ts = engine.timed_render(v, text, s) 16 kHz int16 or None
    clips   = engine.batch(v, s, ["a", "b"])    see batch()

Two conventions, inherited from the corpus layer they were extracted from:

  * the kokoro engines RETURN (None, ...) on failure - a transient TTS error must
    not take down a corpus stage that has run for an hour;
  * the piper engine RAISES on transport failure, because its caller wants to
    report WHICH (voice, speaker) failed, and a bare None would say nothing.
    The piper corpus generator catches per clip exactly as it always did.
"""


class Engine:
    name = "engine"

    # Whether timed_render returns per-word start/end times. This is what makes
    # batch() a real batch rather than a loop - see batch().
    supports_timestamps = False

    # Whether voices() returns (voice, speaker) pairs (Piper) rather than plain
    # names (Kokoro). The protocol carries the flag (wire.py); a local engine
    # declares it here so both sides of that wire agree with the process that
    # never crosses one.
    speaker_voices = False


    # How the client generator drives this engine:
    #   "batch"   - join a group of texts into one render and split on word
    #               timestamps; needs supports_timestamps
    #   "serial"  - one render per clip, in the caller's order; piper needs this
    #               because its server holds exactly one voice loaded at a time,
    #               so order is a correctness property, not a preference
    #               (src/train/corpus/piper.py: "VOICE IS THE OUTER LOOP ON PURPOSE").
    batch_mode = "serial"

    def available(self):
        """(usable, reason). Never raises - the caller decides what to do about it."""
        raise NotImplementedError

    def voices(self, **kwargs):
        raise NotImplementedError

    def timed_render(self, voice, text: str, speed: float = 1.0):
        """(16 kHz int16 audio or None, word timestamps or None).

        Timestamps are dicts with word / start_time / end_time in seconds - the
        shape /dev/captioned_speech and the kokoro-mlx fork both return, the one
        phrase_end_sample and split_joined read.
        """
        raise NotImplementedError

    def render(self, voice, text: str, speed: float = 1.0):
        audio, _ = self.timed_render(voice, text, speed)
        return audio

    def batch(self, voice, speed: float, texts: list):
        """Render several utterances and return them as a list of (audio, timestamps).

        Moved from corpus/kokoro.py's kokoro_tts_batch, which measured the case:
        a short request to Kokoro-FastAPI costs ~119 ms of fixed overhead plus
        ~42 ms per second of audio, so for a phrase under a second THREE QUARTERS
        of the request is overhead. In isolation a batch of 16-32 reached ~37 ms/
        clip against 182 individually (5x); at the ~9-clip buckets the default
        corpus actually forms, 137 -> 42 ms/clip end to end, a 3.3x speedup.

        WHAT MADE IT MOBILE: the algorithm only needs word timestamps, and those
        come from the engine, not from the wire. Kokoro over HTTP gets them from
        /dev/captioned_speech; kokoro-mlx from the fork's return_timestamps (the
        fork exists to add exactly that). Piper has none, so its batch() degrades
        to a per-clip loop - the same work, without the joining.

        The split is exact, not energy-based: each utterance is cut at its own
        word boundaries, so coarticulation between the joined texts does not leak
        across a cut. The texts are joined with ". " and the punctuation tokens
        are filtered back out of the timestamp list.

        Every text in a batch shares one voice and one speed - that is what makes
        it a single forward pass - so callers must group by (voice, speed) before
        calling.

        Returns a list in the order given, with None for any utterance whose
        words could not be located. A caller seeing None should render that one
        individually (the corpus generator does).
        """
        if not self.supports_timestamps:
            return [(self.render(voice, t, speed), None) for t in texts]
        joined = ". ".join(t.rstrip(".") for t in texts) + "."
        data, timestamps = self.timed_render(voice, joined, speed)
        if data is None or not timestamps:
            return [(None, None)] * len(texts)
        return split_joined(data, timestamps, texts)


def split_joined(data, timestamps, texts, sr: int = 16000):
    """Split one joined rendering back into its per-utterance clips.

    The body of the old kokoro_tts_batch, verbatim: given the audio of
    `". ".join(texts) + "."` and its word timestamps, cut each text at its own
    word boundaries.

    Punctuation arrives as its own token; drop it so word indices line up.
    """
    words = [t for t in timestamps if str(t.get("word", "")).strip(".,!?;:")]

    out, cursor = [], 0
    # 30 ms of pad either side: enough to keep the phoneme onsets and codas the
    # trim step would otherwise need, without adding measurable silence.
    pad = int(sr * 30 / 1000)
    for text in texts:
        n_words = len(text.split())
        if cursor + n_words > len(words):
            out.append((None, None))
            continue
        span = words[cursor:cursor + n_words]
        cursor += n_words

        start = max(0, int(span[0]["start_time"] * sr) - pad)
        end = min(len(data), int(span[-1]["end_time"] * sr) + pad)
        if end <= start:
            out.append((None, None))
            continue

        # Re-base the timestamps so they read as if this clip were rendered alone -
        # phrase_end_sample and the run-on cut both index from the clip's own start.
        rebased = [{"word": t.get("word"),
                    "start_time": t["start_time"] - start / sr,
                    "end_time": t["end_time"] - start / sr} for t in span]
        out.append((data[start:end], rebased))
    return out
