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
            text_callback(dt.text)
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

    def process(self, buf, note_instrument, layer_gain, aggro, tick_unit, truncate, chords_gain = None):

        pitch = PitchDetector()
        synth = Synth('data/FluidR3_GM.sf2')
        synth.program(NOTE_CHANNEL, 0, note_instrument)
        synth.program(CHORD_CHANNEL, 0, self.chord_instrument)

        tick = 0
        # note_frame_count = Audio.sample_rate / 2
        note_frame_count = int(round(Audio.sample_rate * 60.0 / self.bpm))
        cur_pitch = None
        # writer = AudioWriter('processed')
        # writer.start()
        ret_data_list = []

        cur_template = make_snap_template(self.chords[0])
        line_progress = [(0, 0)] * len(self.lines)
        chord_idx = 0
        chord_tick = 0
        if chords_gain is not None:
            for line in self.lines:
                synth.noteon(CHORD_CHANNEL, line[0][0], chords_gain)

        raw_pitch_segments = []
        processed_pitch_segments = []

        last_pitch = 0
        while True:
            frame = int(round(Audio.sample_rate * 60.0 / self.bpm * tick / kTicksPerQuarter))
            end_frame = int(round(Audio.sample_rate * 60.0 / self.bpm * (tick + tick_unit) / kTicksPerQuarter))
            # print(frame, end_frame)
            unknown_slice = buf.get_frames(frame, end_frame)
            if not unknown_slice.size: break
            mono_slice = unknown_slice[::buf.get_num_channels()]
            segments = pitch_segments(pitch, mono_slice)
            cur_pitch = majority_pitch(segments, cur_template, aggro, truncate)
            if last_pitch and cur_pitch:
                cur_pitch = push_near(last_pitch, cur_pitch, 10)
            # print(cur_pitch)

            raw_pitch_segments.extend(segments)
            processed_pitch_segments.append((cur_pitch, mono_slice.size))

            if last_pitch != cur_pitch:
                if last_pitch:
                    synth.noteoff(NOTE_CHANNEL, last_pitch)
                if cur_pitch:
                    synth.noteon(NOTE_CHANNEL, cur_pitch, layer_gain)
            last_pitch = cur_pitch

            synth_data, continue_flag = synth.generate(end_frame - frame, 2)
            ret_data_list.append(synth_data)

            tick += tick_unit

            chord_tick += tick_unit
            if chord_tick >= self.chords[chord_idx].duration:
                chord_idx = (chord_idx + 1) % len(self.chords)
                cur_template = make_snap_template(self.chords[chord_idx])
                chord_tick = 0

            if chords_gain is not None:
                for i, ((note_idx, note_tick), line) in enumerate(zip(line_progress, self.lines)):
                    note_tick += tick_unit
                    if note_tick >= line[note_idx][0]:
                        synth.noteoff(CHORD_CHANNEL, line[note_idx][1])
                        note_idx = (note_idx + 1) % len(line)
                        synth.noteon(CHORD_CHANNEL, line[note_idx][1], chords_gain)
                        note_tick = 0
                    line_progress[i] = (note_idx, note_tick)

        return combine_buffers(ret_data_list), raw_pitch_segments, processed_pitch_segments

if __name__ == "__main__":
    infile = WaveFile('solo_test_files/solo_test_60bpm_la_connected.wav')
    VoxxEngine().process(infile)
