from __future__ import division, print_function

from common.core import *
from common.clock import kTicksPerQuarter
from common.audio import *
from common.writer import *
from common.wavesrc import WaveFile
from common.synth import *

from collections import Counter

import demo_chords
from chords_gen import Chord, ChordTemplate

import random
from pitch_detector import PitchDetector
MYPY = False
if MYPY: from typing import List, Tuple

NUM_CHANNELS = 2

NOTE_CHANNEL = 0
CHORD_CHANNEL = 1

# snap = (0, 2, 4, 5, 7, 9, 11, 12)
# snap = (0, 2, 4, 7, 9, 12)
# snap = (0, 4, 7, 12)

def frame_of_tick(bpm, tick):
    return int(round(Audio.sample_rate * 60.0 / bpm * tick / kTicksPerQuarter))

def make_snap_template(chord):
    # type: (Chord) -> List[Tuple[int, int]]
    template = list(sorted((p % 12, w) for p, w in chord.pitches.iteritems()))
    lp, lw = template[-1]
    rp, rw = template[0]
    return [(lp - 12, lw)] + template + [(rp + 12, rw)]

def snap_to_template(pitch, template, aggro):
    # template = (0, 2, 4, 5, 7, 9, 11, 12)
    if pitch == 0: return 0
    octave = 12 * (pitch // 12)
    # print(template, pitch, template, aggro)
    return int(octave + min(template, key=lambda (p, w): abs(p - (pitch - octave)) / w ** aggro)[0])

def push_near(anchor, pitch, max_jump):
    if pitch == 0: return 0
    while pitch > anchor + max_jump: pitch -= 12
    while pitch < anchor - max_jump: pitch += 12
    return pitch

def pitch_segments(pitch_detector, mono_frame_array):
    mi = 0
    WINDOW = 1024
    ret = []
    while mi < mono_frame_array.size:
        cur_slice = mono_frame_array[mi:mi + WINDOW]
        pitch = pitch_detector.write(cur_slice)
        mi += WINDOW
        ret.append((pitch, cur_slice.size))
    return ret

def majority_pitch(pitch_segments, template, aggro, truncate):
    note_votes = Counter()
    for pitch, weight in pitch_segments:
        if pitch > truncate:
            cur_note = 0
        else:
            cur_note = snap_to_template(pitch, template, aggro)
        note_votes[cur_note] += weight
    return max(note_votes.items(), key=lambda x: x[1])[0]

WINDOW = 1024

class VoxxPartial(object):
    def __init__(self, bpm, chords, aggro, tick_unit, truncate):
        self.all_buffers = [] # unprocessed

        self.pitch_detector = PitchDetector()

        self.all_processed_segments = []
        self.all_segments = []

        # A chunk has many segments. Frames are flushed to segments, which are flushed to chunks.
        # Unflushed raw pitch segments
        self.unflushed_segments = []
        self.unflushed_segments_size = 0
        self.buffer_acc = np.array([], dtype=np.float32)

        self.bpm = bpm
        self.tick_unit = tick_unit
        self.chords = chords
        self.aggro = aggro
        self.truncate = truncate

        self.tick = 0
        self.cur_template = make_snap_template(self.chords[0])
        self.chord_idx = 0
        self.chord_tick = 0
        self.last_pitch = 0

    def current_chunk_frames(self):
        return frame_of_tick(self.bpm, self.tick + self.tick_unit) - frame_of_tick(self.bpm, self.tick)
    def current_segment_frames(self):
        return min(WINDOW, self.current_chunk_frames() - self.unflushed_segments_size)

    def flush_chunk(self):
        cur_pitch = majority_pitch(self.unflushed_segments, self.cur_template, self.aggro, self.truncate)
        if self.last_pitch and cur_pitch:
            cur_pitch = push_near(self.last_pitch, cur_pitch, 10)
        self.last_pitch = cur_pitch

        self.all_processed_segments.append((cur_pitch, self.unflushed_segments_size))

        self.tick += self.tick_unit
        self.chord_tick += self.tick_unit
        self.unflushed_segments = []
        self.unflushed_segments_size = 0

        if self.chord_tick >= self.chords[self.chord_idx].duration:
            self.chord_idx = (self.chord_idx + 1) % len(self.chords)
            self.cur_template = make_snap_template(self.chords[self.chord_idx])
            self.chord_tick = 0

    def append(self, frames, channels):
        mono_frames = frames[::channels]
        self.all_buffers.append(mono_frames)
        self.buffer_acc = np.append(self.buffer_acc, mono_frames)

        # can we flush a segment?
        while self.current_segment_frames() <= self.buffer_acc.size:
            s = self.current_segment_frames()
            segment_frames = self.buffer_acc[:s]
            self.buffer_acc = self.buffer_acc[s:]
            pitch = self.pitch_detector.write(segment_frames)
            self.all_segments.append((pitch, segment_frames.size))
            self.unflushed_segments.append((pitch, segment_frames.size))
            self.unflushed_segments_size += segment_frames.size

            # can we (should we) flush a chunk?
            if self.unflushed_segments_size >= self.current_chunk_frames():
                assert self.unflushed_segments_size == self.current_chunk_frames()
                self.flush_chunk()

class VoxxEngine(object):
    def __init__(self):
        if False:
            self.chords = demo_chords.which # type: List[Chord]
            self.lines = [demo_chords.baseline, demo_chords.guitar2, demo_chords.guitar3] # type: List[List[Tuple[int, int]]]
            self.duration_texts = demo_chords.texts # type: List[DurationText]
        else:
            self.set_chord_template(ChordTemplate([1, 3, 6, 4, 2, 7], ('e', 'minor'), 240))
        # self.note_instrument = 40 # violin
        self.chord_instrument = 24 # flute?
        self.bpm = 120

    def play_lines(self, synth, scheduler, gain_callback, text_callback):

        stopper = [False]

        def next_note_play(tick, (melody, i)):
            synth.noteoff(CHORD_CHANNEL, melody[(i - 1)%len(melody)][1])
            if stopper[0]: return
            synth.noteon(CHORD_CHANNEL, melody[i][1], gain_callback())
            scheduler.post_at_tick(tick + melody[i][0], next_note_play, (melody, (i + 1) % (len(melody))))

        for line in self.lines:
            next_note_play(scheduler.get_tick(), (line, 0))

        def next_text(tick, i):
            if stopper[0]: return
            dt = self.duration_texts[i]
            text_callback(i, dt.text)
            scheduler.post_at_tick(tick + dt.duration, next_text, (i + 1) % (len(self.duration_texts)))

        next_text(scheduler.get_tick(), 0)

        def stop_callback():
            stopper[0] = True
        return stop_callback

    def set_chord_template(self, ct):
        # type: (ChordTemplate) -> None
        self.lines = ct.lines
        self.chords = ct.chords
        self.duration_texts = ct.duration_texts

    def make_partial(self, pitch_snap, tick_unit, truncate):
        return VoxxPartial(self.bpm, self.chords, pitch_snap, tick_unit, truncate)

    def process(self, buf, pitch_snap, tick_unit, truncate):
        partial = self.make_partial(pitch_snap, tick_unit, truncate)
        partial.append(buf.data, buf.get_num_channels())
        return partial.all_segments, partial.all_processed_segments

    def render(self, pitch_segments, note_instrument, layer_gain, chords_gain = None):
        synth = Synth('data/FluidR3_GM.sf2')
        synth.program(NOTE_CHANNEL, 0, note_instrument)
        synth.program(CHORD_CHANNEL, 0, self.chord_instrument)

        line_progress = [(0, 0)] * len(self.lines)

        ret_data_list = []

        if chords_gain is not None:
            for line in self.lines:
                synth.noteon(CHORD_CHANNEL, line[0][0], chords_gain)

        last_pitch = 0
        i = 0
        for cur_pitch, size in pitch_segments:
            if last_pitch != cur_pitch:
                if last_pitch:
                    synth.noteoff(NOTE_CHANNEL, last_pitch)
                if cur_pitch:
                    synth.noteon(NOTE_CHANNEL, cur_pitch, layer_gain)
            last_pitch = cur_pitch

            synth_data, continue_flag = synth.generate(size, 2)
            ret_data_list.append(synth_data)

            if chords_gain is not None:
                for i, ((note_idx, note_tick), line) in enumerate(zip(line_progress, self.lines)):
                    note_tick += tick_unit
                    if note_tick >= line[note_idx][0]:
                        synth.noteoff(CHORD_CHANNEL, line[note_idx][1])
                        note_idx = (note_idx + 1) % len(line)
                        synth.noteon(CHORD_CHANNEL, line[note_idx][1], chords_gain)
                        note_tick = 0
                    line_progress[i] = (note_idx, note_tick)
        return combine_buffers(ret_data_list)

if __name__ == "__main__":
    infile = WaveFile('solo_test_files/solo_test_60bpm_la_connected.wav')
    VoxxEngine().process(infile)
