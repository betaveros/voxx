#####################################################################
#
# input_demo.py
#
# Copyright (c) 2017, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

# contains example code for some simple input (microphone) processing.
# Requires aubio (pip install aubio).


import sys
sys.path.append('..')

from common.core import *
from common.audio import *
from common.writer import *
from common.mixer import *
from common.gfxutil import *
from common.wavegen import *
from common.clock import *
from common.metro import *
from common.synth import *
from buffers import *
from engine import *

from collections import Counter

from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen

import chords_gen
import demo_chords

import random
from pitch_detector import PitchDetector

NUM_CHANNELS = 2

CHORD_CHANNEL = 1

# Same as WaveSource interface, but is given audio data explicity.
class WaveArray(object):
    def __init__(self, np_array, num_channels):
        super(WaveArray, self).__init__()

        self.data = np_array
        self.num_channels = num_channels

    # start and end args are in units of frames,
    # so take into account num_channels when accessing sample data
    def get_frames(self, start_frame, end_frame) :
        start_sample = start_frame * self.num_channels
        end_sample = end_frame * self.num_channels
        return self.data[start_sample : end_sample]

    def get_num_channels(self):
        return self.num_channels


# this class is a generator. It does no actual buffering across more than one call. 
# So underruns/overruns are likely, resulting in pops here and there. 
# But code is simpler to deal with and it reduces latency. 
# Otherwise, it would need a FIFO read-write buffer
class IOBuffer(object):
    def __init__(self):
        super(IOBuffer, self).__init__()
        self.buffer = None

    # add data
    def write(self, data):
        self.buffer = data

    # send that data to the audio sink
    def generate(self, num_frames, num_channels) :
        num_samples = num_channels * num_frames

        # if nothing was added, just send out zeros
        if self.buffer is None:
            return np.zeros(num_samples), True

        # if the data added recently is not of the proper size, just resize it.
        # this will cause some pops here and there. So, not great for a real solution,
        # but ok for now.
        if num_samples != len(self.buffer):
            tmp = self.buffer.copy()
            tmp.resize(num_samples)
            if num_samples < len(self.buffer):
                print 'IOBuffer:overrun'
            else:
                print 'IOBuffer:underrun'

        else:
            tmp = self.buffer

        # clear out buffer because we just used it
        self.buffer = None
        return tmp, True


# looks at incoming audio data, detects onsets, and then a little later, classifies the onset as 
# "kick" or "snare"
# calls callback function with message argument that is one of "onset", "kick", "snare"
class OnsetDectior(object):
    def __init__(self, callback):
        super(OnsetDectior, self).__init__()
        self.callback = callback

        self.last_rms = 0
        self.buffer = FIFOBuffer(4096)
        self.win_size = 512 # window length for analysis
        self.min_len = 0.1  # time (in seconds) between onset detection and classification of onset

        self.cur_onset_length = 0 # counts in seconds
        self.zc = 0               # zero-cross count

        self.active = False # is an onset happening now

    def write(self, signal):
        # use FIFO Buffer to create same-sized windows for processing
        self.buffer.write(signal)
        while self.buffer.get_read_available() >= self.win_size:
            data = self.buffer.read(self.win_size)
            self._process_window(data)

    # process a single window of audio, of length self.win_size
    def _process_window(self, signal):
        # only look at the difference between current RMS and last RMS
        rms = np.sqrt(np.mean(signal ** 2))
        delta = rms - self.last_rms
        self.last_rms = rms

        # if delta exceeds threshold and not active:
        if not self.active and delta > 0.003:
            self.callback('onset')
            self.active = True
            self.cur_onset_length = 0  # begin timing onset length
            self.zc = 0                # begin counting zero-crossings

        self.cur_onset_length += len(signal) / float(Audio.sample_rate)

        # count and accumulate zero crossings:
        zc = np.count_nonzero(signal[1:] * signal[:-1] < 0)
        self.zc += zc

        # it's classification time!
        # classify based on a threshold value of the accumulated zero-crossings.
        if self.active and self.cur_onset_length > self.min_len:
            self.active = False
            # print 'zero cross', self.zc
            self.callback(('kick', 'snare')[self.zc > 200])


# graphical display of a meter
class MeterDisplay(InstructionGroup):
    def __init__(self, pos, height, in_range, color):
        super(MeterDisplay, self).__init__()
        
        self.max_height = height
        self.range = in_range

        # dynamic rectangle for level display
        self.rect = Rectangle(pos=(1,1), size=(50,self.max_height))

        self.add(PushMatrix())
        self.add(Translate(*pos))

        # border
        w = 52
        h = self.max_height+2
        self.add(Color(0,0,0))
        self.add(Line(points=(0,0, 0,h, w,h, w,0, 0,0), width=2))

        # meter
        self.add(Color(*color))
        self.add(self.rect)

        self.add(PopMatrix())

    def set(self, level):
        h = np.interp(level, self.range, (0, self.max_height))
        self.rect.size = (50, h)


# graphical display of onsets as a growing (snare) or shrinking (kick) circle
class OnsetDisplay(InstructionGroup):
    def __init__(self, pos):
        super(OnsetDisplay, self).__init__()

        self.anim = None
        self.start_sz = 100
        self.time = 0

        self.color = Color(1,1,1,1)
        self.circle = CEllipse(cpos=(0,0), csize=(self.start_sz, self.start_sz))

        self.add(PushMatrix())
        self.add(Translate(*pos))
        self.add(self.color)        
        self.add(self.circle)
        self.add(PopMatrix())

    def set_type(self, t):
        print t
        if t == 'kick':
            self.anim = KFAnim((0, 1,1,1,1, self.start_sz), (0.5, 1,0,0,1, 0))
        else:
            self.anim = KFAnim((0, 1,1,1,1, self.start_sz), (0.5, 1,1,0,0, self.start_sz*2))

    def on_update(self, dt):
        if self.anim == None:
            return True

        self.time += dt
        r,g,b,a,sz = self.anim.eval(self.time)
        self.color.rgba = r,g,b,a
        self.circle.csize = sz, sz

        return self.anim.is_active(self.time)


# continuous plotting and scrolling line
class GraphDisplay(InstructionGroup):
    def __init__(self, pos, height, num_pts, in_range, color):
        super(GraphDisplay, self).__init__()

        self.num_pts = num_pts
        self.range = in_range
        self.height = height
        self.points = np.zeros(num_pts*2, dtype = np.int)
        self.points[::2] = np.arange(num_pts) * 4
        self.idx = 0
        self.mode = 'scroll'
        self.line = Line( width = 1.5 )
        self.add(PushMatrix())
        self.add(Translate(*pos))
        self.add(Color(*color))
        self.add(self.line)
        self.add(PopMatrix())

    def add_point(self, y):
        y = int( np.interp( y, self.range, (0, self.height) ))

        if self.mode == 'loop':
            self.points[self.idx + 1] = y
            self.idx = (self.idx + 2) % len(self.points)

        elif self.mode == 'scroll':
            self.points[3:self.num_pts*2:2] = self.points[1:self.num_pts*2-2:2]
            self.points[1] = y

        self.line.points = self.points.tolist()

class MainWidget1(BaseWidget) :
    def __init__(self):
        super(MainWidget1, self).__init__()

        self.writer = AudioWriter('data') # for debugging audio output
        self.audio = Audio(NUM_CHANNELS, self.writer.add_audio, input_func=self.receive_audio)
        self.mixer = Mixer()
        self.io_buffer = IOBuffer()
        self.mixer.add(self.io_buffer)

        self.synth = Synth('data/FluidR3_GM.sf2')
        self.synth.program(0, 0, 40) # violin
        self.synth.program(0, 0, 73) # flute
        self.synth.program(1, 0, 24)
        self.synth_note = 0
        self.last_note = 60 # always nonzero, user's note won't be too far away
        self.max_jump = 10
        self.pitch_offset_index = 2
        self.snap_chance_index = 2
        self.is_random = False
        self.quantization_unit_index = 3
        self.mixer.add(self.synth)

        self.note_votes = Counter()

        self.onset_detector = OnsetDectior(self.on_onset)
        self.pitch = PitchDetector()

        self.recording = False
        self.monitor = False
        self.channel_select = 0
        self.input_buffers = []
        self.live_wave = None

        self.info = topleft_label()
        self.info.color = (0, 0, 0, 1)
        self.add_widget(self.info)

        self.anim_group = AnimGroup()

        self.mic_meter = MeterDisplay((50, 25),  150, (-96, 0), (0.0,0.6,0.1))
        self.mic_graph = GraphDisplay((110, 25), 150, 300, (-96, 0), (0.0,0.6,0.1))

        self.pitch_meter = MeterDisplay((50, 200), 300, (30, 90), (.9,.1,.3))
        self.pitch_graph = GraphDisplay((110, 200), 300, 300, (30, 90), (.9,.1,.3))

        self.output_pitch_graph = GraphDisplay((110, 200), 300, 300, (30, 90), (0.0,0.0,1.0))

        self.canvas.add(self.mic_meter)
        self.canvas.add(self.mic_graph)
        self.canvas.add(self.pitch_meter)
        self.canvas.add(self.pitch_graph)
        self.canvas.add(self.output_pitch_graph)

        self.canvas.add(self.anim_group)

        self.onset_disp = None
        self.cur_pitch = 0
        # self.chord_seq = chords_gen.chord_generater([1, 3, 6, 4, 2, 7], ['e', 'minor'], 240)[3]
        self.chord_seq = demo_chords.which
        self.next_template = make_snap_template(self.chord_seq[0][1:])

        self.recording_idx = 0

        self.tempo_map  = SimpleTempoMap(120)
        self.sched = AudioScheduler(self.tempo_map)
        self.mixer.add(self.sched)

        self.audio.set_generator(self.mixer)
        self.next_note(self.sched.get_tick(), (0, 0)) # specifies what was just voted on and should start playing
        self.next_note_play(self.sched.get_tick(), (demo_chords.baseline, 0))
        self.next_note_play(self.sched.get_tick(), (demo_chords.guitar2, 0))
        self.next_note_play(self.sched.get_tick(), (demo_chords.guitar3, 0))

    def get_pitch_offset(self):
        return (-24, -12, 0, 12, 24)[self.pitch_offset_index]

    def get_snap_chance(self):
        return self.snap_chance_index / 2.0

    def get_quantization_unit(self):
        return (40, 60, 80, 120, 240)[self.quantization_unit_index]

    def next_note_play(self, tick, (melody, i)):

        self.synth.noteoff(CHORD_CHANNEL, melody[(i - 1)%len(melody)][1])
        self.synth.noteon(CHORD_CHANNEL, melody[i][1], 100)
        self.sched.post_at_tick(tick + melody[i][0], self.next_note_play, (melody, (i + 1) % (len(melody))))

    def next_note(self, tick, (ci, ct)):
        # print('next note')

        TICK_UNIT = self.get_quantization_unit()


        cur_duration = self.chord_seq[ci][0]
        # vote on this next:
        if ct + TICK_UNIT < cur_duration:
            nci, nct = ci, ct + TICK_UNIT
            ticks_to_next = TICK_UNIT
        else:
            nci, nct = (ci + 1)%len(self.chord_seq), 0
            ticks_to_next = cur_duration

        if self.note_votes:
            maj_note = max(self.note_votes.items(), key=lambda x: x[1])[0]
            # print('=' * 100)
            # print('MAJ NOTE', maj_note)
            # print('=' * 100)
            if maj_note != self.synth_note:
                CHANNEL = 0
                self.synth.noteoff(CHANNEL, int(self.synth_note))
                self.synth_note = maj_note
                if self.synth_note != 0:
                    self.last_note = self.synth_note
                    self.synth.noteon(CHANNEL, int(self.synth_note), 100)
            self.note_votes = Counter()

        self.next_template = make_snap_template(self.chord_seq[nci][1:])

        self.sched.post_at_tick(tick + TICK_UNIT, self.next_note, (nci, nct))

    def on_update(self) :
        self.audio.on_update()
        self.anim_group.on_update()

        self.info.text = 'fps:%d\n' % kivyClock.get_fps()
        self.info.text += 'load:%.2f\n' % self.audio.get_cpu_load()
        self.info.text += 'gain:%.2f\n' % self.mixer.get_gain()
        self.info.text += "pitch: %.1f (o: %+d)\n" % (self.cur_pitch, self.get_pitch_offset())
        self.info.text += "j/k: max jump: %d\n" % self.max_jump
        self.info.text += "q: quantization: %d" % self.get_quantization_unit()
        self.info.text += " / s: snap: %.2f\n" % self.get_snap_chance()
        self.info.text += "z: random input: %s\n" % ("OFF", "ON")[self.is_random]

        self.info.text += "c: analyzing channel:%d\n" % self.channel_select
        self.info.text += "r: toggle recording: %s\n" % ("OFF", "ON")[self.recording]
        self.info.text += "m: monitor: %s\n" % ("OFF", "ON")[self.monitor]
        self.info.text += "p: playback memory buffer"

    def receive_audio(self, frames, num_channels) :
        # print '#', frames.size
        # handle 1 or 2 channel input.
        # if input is stereo, mono will pick left or right channel. This is used
        # for input processing that must receive only one channel of audio (RMS, pitch, onset)
        if num_channels == 2:
            mono = frames[self.channel_select::2] # pick left or right channel
        else:
            mono = frames

        # Microphone volume level, take RMS, convert to dB.
        # display on meter and graph
        rms = np.sqrt(np.mean(mono ** 2))
        rms = np.clip(rms, 1e-10, 1) # don't want log(0)
        db = 20 * np.log10(rms)      # convert from amplitude to decibels 
        self.mic_meter.set(db)
        self.mic_graph.add_point(db)

        # pitch detection: get pitch and display on meter and graph
        self.cur_pitch = self.pitch.write(mono)
        if self.is_random:
            self.cur_pitch = random.randint(40, 80)
        self.pitch_meter.set(self.cur_pitch)
        self.pitch_graph.add_point(self.cur_pitch)

        pitch = self.cur_pitch
        if pitch:
            pitch += self.get_pitch_offset()
            pitch = push_near(self.last_note, pitch, self.max_jump)
            if random.random() < self.get_snap_chance():
                cur_note = snap_to_template(pitch, self.next_template)
            else:
                cur_note = int(round(pitch))
        else:
            cur_note = 0
        self.note_votes[cur_note] += len(frames)
        # print(cur_note)
        # cur_note = 60

        self.output_pitch_graph.add_point(self.synth_note)

        # onset detection and classification
        self.onset_detector.write(mono)

        # optionally send input out to speaker for live playback
        # note that this will have a ton of latency and is therefore not
        # practical
        if self.monitor:
            self.io_buffer.write(frames)

        # record to internal buffer for later playback as a WaveGenerator
        if self.recording:
            self.input_buffers.append(frames)


    def on_onset(self, msg):
        if msg == 'onset':
            self.onset_disp = OnsetDisplay((random.randint(650, 750), 100))
            self.anim_group.add(self.onset_disp)
        elif self.onset_disp:
            self.onset_disp.set_type(msg)
            self.onset_disp = None

    def on_key_down(self, keycode, modifiers):
        # toggle recording
        if keycode[1] == 'r':
            if self.recording:
                self._process_input(self.recording_idx)
                self.recording_idx = self.recording_idx + 1
            self.recording = not self.recording

        if keycode[1] == 'x':
            self.writer.toggle()

        # toggle monitoring
        if keycode[1] == 'm':
            self.monitor = not self.monitor

        # play back live buffer
        if keycode[1] == 'p':
            if self.live_wave:
                self.mixer.add(WaveGenerator(self.live_wave))

        if keycode[1] == 'c' and NUM_CHANNELS == 2:
            self.channel_select = 1 - self.channel_select

        # toggle snap
        if keycode[1] == 's':
            self.snap_chance_index = (self.snap_chance_index + 1) % 3

        # toggle octave offset
        if keycode[1] == 'o':
            self.pitch_offset_index = (self.pitch_offset_index + 1) % 5

        # quantization
        if keycode[1] == 'q':
            self.quantization_unit_index = (self.quantization_unit_index + 1) % 5

        # toggle random input
        if keycode[1] == 'z':
            self.is_random = not self.is_random

        # toggle jump
        if keycode[1] == 'j' and self.max_jump > 7:
            self.max_jump -= 1
        if keycode[1] == 'k':
            self.max_jump += 1

        # adjust mixer gain
        gf = lookup(keycode[1], ('up', 'down'), (1.1, 1/1.1))
        if gf:
            new_gain = self.mixer.get_gain() * gf
            self.mixer.set_gain( new_gain )

    def _process_input(self,idx) :
        data = combine_buffers(self.input_buffers)
        print 'live buffer size:', len(data) / NUM_CHANNELS, 'frames'
        write_wave_file(data, NUM_CHANNELS, 'recording' + str(REC_INDEX[idx]) + '.wav' )
        self.live_wave = WaveArray(data, NUM_CHANNELS)
        self.input_buffers = []

class ScreenWithBackground(Screen):
    def __init__(self, name, rgba = (0.694, 0.976, 0.988, 1)):
        super(Screen, self).__init__(name=name)

        with self.canvas.before:
            Color(*rgba)
            rect = Rectangle(size=self.size, pos=(0, 0))
        def update_rect(instance, value):
            rect.size = instance.size
        self.bind(size=update_rect)

class IntInput(TextInput):

    def __init__(self, **kwargs):
        super(IntInput, self).__init__(multiline=False, **kwargs)

    def insert_text(self, substring, from_undo=False):
        good = ''.join(c for c in substring if c.isdigit())
        return super(IntInput, self).insert_text(good, from_undo=from_undo)


green_button = (0.164, 0.517, 0.552, 1)
orange_button = (0.980, 0.521, 0.4, 1)
text_color = (0.992, 0.925, 0.960,1)
class MainMainWidget1(ScreenManager):

    def __init__(self):
        super(MainMainWidget1, self).__init__()
        self.make_start_screen()
        self.make_mood_screen_1()
        self.make_mood_screen_2()
        self.make_record_screen()
        main_screen = ScreenWithBackground('main')
        self.w1 = MainWidget1()
        main_screen.add_widget(self.w1)
        self.add_widget(main_screen)



    def make_start_screen(self):
        screen = ScreenWithBackground('start')
        label1 = Label(text='VoXX!',
                font_size = 300,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color= green_button)
        label2 = Label(text='Set Background Track',
                font_size = 100,
                size_hint=(.7, .2), pos_hint={'x':.15, 'y':.4},
                color=green_button)
        button = Button(text='Set by Mood',
                size_hint=(.5, .25), pos_hint={'x':.25, 'y':.25},
                background_color=(0, 0.5, 0.6, 1))
        button.bind(on_press=self.go_to_callback('mood1'))

        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(button)
        self.add_widget(screen)

    def make_mood_screen_1(self):
        screen = ScreenWithBackground('mood1')
        label = Label(text='Mood 1',
                font_size = 300,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color=(0, 0.5, 0.6, 1))
        button = Button(text='Record',
                size_hint=(.5, .25), pos_hint={'x':.25, 'y':.25},
                background_color=(0, 0.5, 0.6, 1))
        button.bind(on_press=self.go_to_callback('record'))

        screen.add_widget(label)
        screen.add_widget(button)
        self.add_widget(screen)

    def make_mood_screen_2(self):
        screen = ScreenWithBackground('mood2')
        label = Label(text='Mood 2',
                font_size = 300,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color=(0, 0.5, 0.6, 1))
        button = Button(text='Record',
                size_hint=(.5, .25), pos_hint={'x':.25, 'y':.25},
                background_color=(0, 0.5, 0.6, 1))
        button.bind(on_press=self.go_to_callback('record'))

        screen.add_widget(label)
        screen.add_widget(button)
        self.add_widget(screen)

    def make_record_screen(self):
        screen = ScreenWithBackground('record')
        label = Label(text='Record TODO',
                font_size = 300,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color=(0, 0.5, 0.6, 1))

        screen.add_widget(label)
        self.add_widget(screen)

    def go_to_callback(self, name):
        def callback(instance):
            self.current = name
        return callback

    def on_update(self):
        self.w1.on_update()
    def on_key_down(self, keycode, modifiers):
        self.w1.on_key_down(keycode, modifiers)

REC_INDEX = [1,2,3,4,5,6,7,8,9]
# pass in which MainWidget to run as a command-line arg
run(MainMainWidget1)
