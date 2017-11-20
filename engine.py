from common.core import *
from common.audio import *
from common.writer import *
from common.wavesrc import WaveFile
from common.synth import *

from collections import Counter

import demo_chords

import random
from pitch_detector import PitchDetector

NUM_CHANNELS = 2

CHORD_CHANNEL = 1

class VoxxEngine(object):
    def __init__(self):
        self.chords = demo_chords.which
        self.lines = [demo_chords.baseline, demo_chords.guitar2, demo_chords.guitar3]

    def process(self, buf):

        pitch = PitchDetector()
        synth = Synth('data/FluidR3_GM.sf2')
        synth.program(0, 0, 40) # violin
        synth.program(1, 0, 24)

        # self.next_template = make_snap_template(self.chord_seq[0][1:])
        # index into mono
        mi = 0
        # note_frame_count = Audio.sample_rate / 2
        note_frame_count = 1000
        cur_pitch = None
        writer = AudioWriter('processed')
        writer.start()
        while True:
            if cur_pitch is not None:
                synth.noteoff(CHORD_CHANNEL, cur_pitch)
            unknown_slice = buf.get_frames(mi, mi + note_frame_count)
            mono_slice = unknown_slice[::buf.get_num_channels()]
            cur_pitch = int(round(pitch.write(mono_slice)))
            print(cur_pitch)
            synth.noteon(CHORD_CHANNEL, cur_pitch, 100)
            synth_data, continue_flag = synth.generate(note_frame_count, 2)
            writer.add_audio(synth_data, 2)
            mi += note_frame_count

            if len(mono_slice) < note_frame_count: break
        writer.stop()

if __name__ == "__main__":
    infile = WaveFile('solo_test_files/solo_test_60bpm_la_connected.wav')
    VoxxEngine().process(infile)

