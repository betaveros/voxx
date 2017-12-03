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
from kivy.uix.slider import Slider
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
# calls callback function with message argument that is one of "onset", "kick", ""
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
        self.translate = Translate(*pos)
        self.add(PushMatrix())
        self.add(self.translate)
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

    def set_pos(self, pos):
        self.translate.xy = pos

class GraphDisplayWidget(BaseWidget):
    def __init__(self, **kwargs):
        super(GraphDisplayWidget, self).__init__(**kwargs)

        self.graph = GraphDisplay(self.pos, 300, 300, (30, 90), (.9,.1,.3))
        self.canvas.add(self.graph)

        self.bind(pos=self.redraw)

    def redraw(self, x, y):
        self.graph.set_pos(self.pos)

class SegmentsDisplay(InstructionGroup):
    def __init__(self, pos, height, x_scale, segments, in_range, color):
        super(SegmentsDisplay, self).__init__()

        self.height = height
        self.x_scale = x_scale
        self.range = in_range

        self.translate = Translate(*pos)
        self.line = Line( width = 1.5 )
        self.add(PushMatrix())
        self.add(self.translate)
        self.add(Color(*color))
        self.add(self.line)
        self.add(PopMatrix())

        self.set_segments(segments)

    def set_segments(self, segments):
        points = [0] * (4*len(segments))

        x = 0
        for i, (val, length) in enumerate(segments):
            j = i * 4
            y = int(np.interp(val, self.range, (0, self.height)))
            points[j] = x
            points[j+1] = y
            x += self.x_scale * length
            points[j+2] = x
            points[j+3] = y

        self.line.points = points

    def set_pos(self, pos):
        self.translate.xy = pos

class SegmentsDisplayWidget(BaseWidget):
    def __init__(self, **kwargs):
        super(SegmentsDisplayWidget, self).__init__(**kwargs)

        self.display = SegmentsDisplay(self.pos, 300, 0.002, [(0, 0)], (30, 90), kwargs['color'])
        self.canvas.add(self.display)
        self.bind(pos=self.redraw)

    def redraw(self, x, y):
        self.display.set_pos(self.pos)

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
        self.next_template = make_snap_template(self.chord_seq[0].notes)

        self.recording_idx = 0

        self.tempo_map  = SimpleTempoMap(120)
        self.sched = AudioScheduler(self.tempo_map)
        self.mixer.add(self.sched)

        self.audio.set_generator(self.mixer)

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


        cur_duration = self.chord_seq[ci].duration
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

        self.next_template = make_snap_template(self.chord_seq[nci].notes)

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

        if keycode[1] == 'spacebar':
            self.next_note(self.sched.get_tick(), (0, 0)) # specifies what was just voted on and should start playing
            self.next_note_play(self.sched.get_tick(), (demo_chords.baseline, 0))
            self.next_note_play(self.sched.get_tick(), (demo_chords.guitar2, 0))
            self.next_note_play(self.sched.get_tick(), (demo_chords.guitar3, 0))

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

class LineTextInput(TextInput):

    def __init__(self, **kwargs):
        super(LineTextInput, self).__init__(
                multiline=False,
                font_size = 100,
                color = dark_teal,
                background_normal = '', background_color = light_pink,
                foreground_color = dark_teal,
                cursor_color = dark_teal,
                **kwargs)

class IntInput(LineTextInput):

    def __init__(self, **kwargs):
        super(IntInput, self).__init__(**kwargs)

    def insert_text(self, substring, from_undo=False):
        good = ''.join(c for c in substring if c.isdigit())
        return super(IntInput, self).insert_text(good, from_undo=from_undo)

    @property
    def int_value(self):
        try:
            return int(self.text)
        except ValueError:
            print("error converting to int:", repr(self.text))
            return 0

background = (0.694, 0.976, 0.988, 1)
dark_teal = (0.164, 0.517, 0.552, 1)
darker_teal = (0.035, 0.345, 0.364,1)
coral = (0.980, 0.521, 0.4, 1)
light_pink = (0.992, 0.925, 0.960,1)
black = (0.317, 0.321, 0.317,1)
bright_blue = (0.160, 0.850, 1,1)


def make_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size=50, color=light_pink, bg_color=dark_teal):
    return Button(text=text, font_size=font_size, color=color, size_hint=(size_hint_x, size_hint_y), pos_hint={'x': pos_hint_x, 'y': pos_hint_y}, background_normal='', background_color=bg_color)

class CoralButtonGroup(object):
    def __init__(self):
        super(CoralButtonGroup, self).__init__()
        self.pressed = None

    def press(self, button):
        if self.pressed is None:
            button.background_color = coral
            self.pressed = button
        if button != self.pressed:
            self.pressed.background_color = dark_teal
            button.background_color = coral
            self.pressed = button

    def clear_pressed(self):
        self.pressed.background_color = dark_teal
        self.pressed = None

    def make_button(self, text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size=50, callback=None):
        button = make_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size)
        def cur_press(instance):
            if callback is not None: callback(instance)
            self.press(instance)
        button.bind(on_press=cur_press)
        return button

def make_bg_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size=50):
    return make_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size, color=black, bg_color=background)

class Layer(object):
    def __init__(self, instrument, data, gain):
        self.instrument = instrument
        self.data = data
        self.gain = gain

class MainMainWidget1(ScreenManager):

    def __init__(self):
        super(MainMainWidget1, self).__init__()
        self.audio = Audio(NUM_CHANNELS, input_func=self.receive_audio)
        self.pitch = PitchDetector()

        self.input_mode = None
        self.mood = None

        self.playing = False
        self.recording = False
        self.input_buffers = []
        self.layers = []
        self.cur_layer = Layer(40, None, 100)

        self.channel_select = 0


        self.make_start_screen()

        self.make_mood_screen_1()
        self.make_progression_screen()

        self.make_input_screen()
        self.make_rhythm_screen()

        self.make_record_screen()
        self.make_tracks_screen()

        self.make_instrument_screen()
        main_screen = ScreenWithBackground('main')
        self.w1 = MainWidget1()
        main_screen.add_widget(self.w1)
        self.add_widget(main_screen)

        self.engine = VoxxEngine()

        self.mixer = Mixer()
        self.synth = Synth('data/FluidR3_GM.sf2')
        self.synth.program(NOTE_CHANNEL, 0, 73) # flute
        self.synth.program(CHORD_CHANNEL, 0, 24)
        self.tempo_map = SimpleTempoMap(120)
        self.sched = AudioScheduler(self.tempo_map)
        self.mixer.add(self.synth)
        self.mixer.add(self.sched)

        self.audio.set_generator(self.mixer)

        Clock.schedule_interval(self.on_update, 0)

    def make_start_screen(self):
        screen = ScreenWithBackground('start')
        label1 = Label(text=u'V\u00f6XX!',
                font_size = 300,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color= dark_teal)

        label2 = Label(text='Set Background Track',
                font_size = 70,
                size_hint=(.7, .3), pos_hint={'x':.15, 'y':.4},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=dark_teal)

        g = CoralButtonGroup()
        button1 = g.make_button('Templates',  .25, .15, .2 , .25, 50, self.go_to_callback('mood1'))
        button2 = g.make_button('Advanced', .25, .15, .55, .25, 50, self.go_to_callback('input'))

        button3 = make_bg_button('Skip Background Track', .25, .15, .7, .04, 40)
        button3.bind(on_press=self.go_to_callback('record'))

        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(button1)
        screen.add_widget(button2)
        screen.add_widget(button3)

        self.add_widget(screen)


    def make_mood_screen_1(self):
        screen = ScreenWithBackground('mood1')
        label = Label(text='What mood would you like today?',
                font_size = 100,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color=(0, 0.5, 0.6, 1))

        mood_group = CoralButtonGroup()
        def add_mood_button(name, sx, sy, px, py, chords, key, rhythm):
            button = mood_group.make_button(name, sx, sy, px, py,
                    callback=self.mood_callback(chords, key, rhythm))
            screen.add_widget(button)

        add_mood_button('Happy', .15, .15, .08, .4, [6, 4, 1, 5], ['c', 'major'], 240)
        add_mood_button('Sad'  , .15, .15, .31, .4, [1, 7, 5, 4], ['c', 'minor'], 1920)
        add_mood_button('Epic' , .15, .15, .54, .4, [4, 1, 6, 5], ['d', 'major'], 480)
        add_mood_button('Chill', .15, .15, .77, .4, [1, 7, 6, 5], ['f', 'minor'], 960)

        button_back = make_bg_button('Back', .1, .15, .01, .02)
        button_next = make_bg_button('Next', .1, .15, .89, .02)
        button_advanced = make_bg_button('Advanced', .2, .15, .4, .02)

        button_next.bind(on_press=self.go_to_callback('length'))
        button_back.bind(on_press=self.go_to_callback('start'))
        button_advanced.bind(on_press=self.go_to_callback('input'))

        screen.add_widget(label)
        screen.add_widget(button_back)
        screen.add_widget(button_next)
        screen.add_widget(button_advanced)
        self.add_widget(screen)

    def make_progression_screen(self):
        screen = ScreenWithBackground('length')
        label1 = Label(text='Almost done!',
                font_size = 150,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                color=dark_teal)

        label2 = Label(text='Set length of the background loop',
                font_size = 80,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.5},
                color=dark_teal)

        self.measure_group = CoralButtonGroup()
        self.button_short = self.measure_group.make_button('Short\n (4 measures)',  .2, .15, .1, .3)
        self.button_mid   = self.measure_group.make_button('Medium\n (6 measures)', .2, .15, .4, .3)
        self.button_long  = self.measure_group.make_button('Long\n (8 measures)',   .2, .15, .7, .3)

        button_back = make_bg_button('Back', .1, .15, .01, .02)
        button_next = make_bg_button('Next', .1, .15, .89, .02)
        button_advanced = make_bg_button('Advanced', .2, .15, .4, .02)


        button_next.bind(on_press=self.go_to_callback('instrument'))
        button_back.bind(on_press=self.go_to_callback('mood1'))
        button_advanced.bind(on_press=self.go_to_callback('input'))


        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(self.button_short)
        screen.add_widget(self.button_mid)
        screen.add_widget(self.button_long)
        screen.add_widget(button_back)
        screen.add_widget(button_next)
        screen.add_widget(button_advanced)
        self.add_widget(screen)

    def finish_set_instrument(self, instance):
        self.cur_layer.instrument = self.instrument_input.int_value
        self.current = 'record'

    def make_input_screen(self):
        screen = ScreenWithBackground('input')
        label1 = Label(text='Advanced Settings',
                font_size = 150,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                color= dark_teal)

        label2 = Label(text='for your masterpiece',
                font_size = 80,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.55},
                color=dark_teal)


        label3 = Label(text='Set Tempo:',
                font_size = 80,
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.5},
                color= dark_teal)

        label4 = Label(text='Set Key:',
                font_size = 80,
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.38},
                color= dark_teal)

        label5 = Label(text='Set Chords:',
                font_size = 80,
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.26},
                color= dark_teal)

        label6 = Label(text='BPM',
                font_size = 60,
                size_hint=(.3, .05), pos_hint={'x':.51, 'y':.5},
                color= dark_teal)

        self.bpm_input = IntInput(
                text = '120',
                size_hint=(.2, .1), pos_hint={'x':.4, 'y':.5})

        self.key_input = LineTextInput(
                text = 'C',
                size_hint=(.2, .1), pos_hint={'x':.4, 'y':.38})

        self.chords_input = LineTextInput(
                text = '1,4,5,1', 
                size_hint=(.5, .1), pos_hint={'x':.4, 'y':.26})

        self.rhythm = 240 # TODO

        button_back = make_bg_button('Back', .1, .15, .01, .02)
        button_next = make_bg_button('Next', .1, .15, .89, .02)

        button_next.bind(on_press=self.go_to_callback('rhythm'))
        button_back.bind(on_press=self.go_to_callback('start'))

        self.mode_group = CoralButtonGroup()
        self.button_major = self.mode_group.make_button('Major', .13, .1, .62, .38)
        self.button_minor = self.mode_group.make_button('Minor', .13, .1, .78, .38)

        # button_major.bind(on_press=self.measure_callback(self.button_major))
        # button_minor.bind(on_press=self.measure_callback(button_minor))

        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(label3)
        screen.add_widget(label4)
        screen.add_widget(label5)
        screen.add_widget(label6)
        screen.add_widget(self.bpm_input)
        screen.add_widget(self.key_input)
        screen.add_widget(self.chords_input)
        screen.add_widget(self.button_major)
        screen.add_widget(self.button_minor)
        screen.add_widget(button_back)
        screen.add_widget(button_next)

        self.add_widget(screen)    

    def make_rhythm_screen(self):
        screen = ScreenWithBackground('rhythm')
        label1 = Label(text='What rhythm would you like?',
                font_size = 100,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                color=dark_teal)

        label2 = Label(text='Select the fastest notes you want in your chord',
                font_size = 50,
                size_hint=(.6, .2), pos_hint={'x':.2, 'y':.56},
                color=dark_teal)


        self.speed_group = CoralButtonGroup()
        button_slow = self.speed_group.make_button('Slow\n (1/4 note)',   .2, .15, .1, .4)
        button_mid  = self.speed_group.make_button('Medium\n (1/8 note)', .2, .15, .4, .4)
        button_fast = self.speed_group.make_button('Fast\n (1/16 note)',  .2, .15, .7, .4)

        button_preview = make_button('Preview', .2, .15, 0.41, 0.03)

        button_back = make_bg_button('Back', .1, .15, .01, .02)

        button_next = make_bg_button('Next', .1, .15, .89, .02)

        button_next.bind(on_press=self.go_to_callback('instrument'))
        button_back.bind(on_press=self.go_to_callback('input'))


        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(button_slow)
        screen.add_widget(button_mid)
        screen.add_widget(button_fast)
        screen.add_widget(button_preview)
        screen.add_widget(button_back)
        screen.add_widget(button_next)
        self.add_widget(screen)


    def make_instrument_screen(self):
        screen = ScreenWithBackground('instrument')
        label1 = Label(text='Select Instrument',
                font_size = 100,
                size_hint=(.7, .2), pos_hint={'x':.15, 'y':.8},
                color=dark_teal)

        label2 = Label(text='Quick Selection',
                font_size = 60,
                size_hint=(.38, .2), pos_hint={'x':.5, 'y':.65},
                color=dark_teal)


        label3 = Label(text='MIDI Number',
                font_size = 70,
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.65},
                color=dark_teal)

        self.instrument_input = IntInput(
                text = '40', # violin?? idk
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.5})

        def add_instrument_button(num, name, sx, sy, px, py):
            def cb(instance):
                self.instrument_input.text = str(num)
            button = make_button(name, sx, sy, px, py)
            button.bind(on_press=cb)
            screen.add_widget(button)

        add_instrument_button( 0, 'Piano'    , .18, .15, .5,  .5)
        add_instrument_button(24, 'Guitar'   , .18, .15, .7,  .5)
        add_instrument_button(40, 'Violin'   , .18, .15, .5, .32)
        add_instrument_button(42, 'Cello'    , .18, .15, .7, .32)
        add_instrument_button(32, 'Bass'     , .18, .15, .5, .14)
        add_instrument_button(65, 'Saxophone', .18, .15, .7, .14)

        button_preview = make_button('Preview', .18, .15, .08, .14, bg_color = darker_teal)
        button_done    = make_button('Done', .18, .15, .28, .14, bg_color = darker_teal)
        button_done.bind(on_press=self.finish_set_instrument)


        button_cancel = make_bg_button('Cancel',.1, .1, .85, .02)
        button_cancel.bind(on_press=self.go_to_callback('start'))

        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(label3)
        screen.add_widget(self.instrument_input)
        screen.add_widget(button_preview) 
        screen.add_widget(button_done)               
        screen.add_widget(button_cancel)      


        self.add_widget(screen)

    def update_record_screen(self):

        self.play_button.disabled = self.recording
        self.record_button.disabled = self.playing
        self.save_button.disabled = self.playing or self.recording or self.cur_layer.data is None
        self.play_button.text = 'Stop' if self.playing else 'Play'
        self.record_button.text = 'Stop' if self.recording else 'Record'

        text = u'{} layer{}'.format(len(self.layers),
                u'' if len(self.layers) == 1 else u's')
        if self.recording:
            text += u' + recording'
        elif self.cur_layer.data is not None:
            text += u' + 1'

        if self.playing:
            text = u'Playing ' + text
        if self.playing or self.recording:
            text += u' ({})'.format(self.engine_playing_text) if self.engine_playing_text else ''
        self.record_label.text = text

    
    def make_tracks_screen(self):
        screen = ScreenWithBackground('tracks')

        label1 = Label(text='All Saved Tracks',
                font_size = 100,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                color=dark_teal)
        button_back = make_bg_button('Back', .1, .15, .01, .02)
        button_back.bind(on_press=self.go_to_callback('record'))

        screen.add_widget(label1)
        screen.add_widget(button_back)
        self.add_widget(screen)


    def make_record_screen(self):
        screen = ScreenWithBackground('record')

        self.record_label = Label(text='...',
                font_size = 50,
                size_hint=(.5, .1), pos_hint={'x':.25, 'y':.75},

                color=(0, 0.5, 0.6, 1))
        self.play_button = make_button('Play', .2, .1, .25, .85, 80)
        self.record_button = make_button('Record', .2, .1, .55, .85, 80)
        self.save_button = make_button('Save', .2, .1, .6, .1, 100)

        self.background_gain_slider = Slider(
                min=0, max=100, value=100, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .1, 'y': .15})
        self.layer_gain_slider = Slider(
                min=0, max=100, value=100, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .2, 'y': .15})
        def get_background_gain():
            return int(round(self.background_gain_slider.value))
        def change_layer_gain(instance, value):
            self.cur_layer.gain = int(round(value))
        self.layer_gain_slider.bind(value=change_layer_gain)

        self.pitch_snap_slider = Slider(
                min=0, max=100, value=0, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .3, 'y': .15})
        self.rhythm_snap_slider = Slider(
                min=0, max=100, value=0, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .4, 'y': .15})
        
        label_background_gain = Label(text='background\nvolume',
                font_size = 30,
                size_hint=(.1, .05), pos_hint={'x':.1, 'y':.05},
                color=dark_teal)

        label_layer_gain = Label(text='solo\nvolume',
                font_size = 30,
                size_hint=(.1, .05), pos_hint={'x':.2, 'y':.05},
                color=dark_teal)

        label_pitch_snap = Label(text='pitch\nsnap',
                font_size = 30,
                size_hint=(.1, .05), pos_hint={'x':.3, 'y':.05},
                color=dark_teal)

        label_rhythm_snap = Label(text='rhythm\nsnap',
                font_size = 30,
                size_hint=(.1, .05), pos_hint={'x':.4, 'y':.05},
                color=dark_teal)


        self.engine_playing_text = ""
        def engine_text_callback(text):
            self.engine_playing_text = text
            self.update_record_screen()

        def prep_engine():
            try:
                chords = [int(s) for s in self.chords_input.text.split(',')]
            except ValueError:
                print('chords broken: ' + repr(self.chords_input.text))
                chords = [1, 5, 6, 4]

            mode = 'minor' if self.mode_group.pressed == self.button_minor else 'major'
            self.engine.set_chords(chords, [self.key_input.text.lower(), mode], self.rhythm)

        def stop_layers():
            self.mixer.remove(self.layers_mixer)
            del self.layers_mixer
        def play_layers(layers):
            self.layers_mixer = Mixer()
            for layer in layers:
                data_array = WaveArray(layer.data, 2)
                instrument = layer.instrument
                processed_data, raw_pitches, processed_pitches = self.engine.process(data_array, instrument, layer.gain)
                self.raw_segments_widget.display.set_segments(raw_pitches)
                self.processed_segments_widget.display.set_segments(processed_pitches)
                self.layers_mixer.add(WaveGenerator(WaveArray(processed_data, 2)))
            self.mixer.add(self.layers_mixer)

        def play(instance):
            if self.playing:
                self.playing = False
                self.engine_stop()
                stop_layers()
            else:
                prep_engine()
                self.playing = True
                self.engine_stop = self.engine.play_lines(self.synth, self.sched, get_background_gain, engine_text_callback)
                layers = self.layers
                if self.cur_layer.data is not None: layers = layers + [self.cur_layer]
                play_layers(layers)

            self.update_record_screen()

        def save(instance):
            if self.cur_layer.data is not None:
                self.layers.append(self.cur_layer)
                self.cur_layer = Layer(self.instrument_input.int_value, None, self.layer_gain_slider.value)

            self.update_record_screen()

        def record(instance):
            if self.recording:
                self.recording = False
                self.engine_stop()
                stop_layers()
                self.cur_layer.data = combine_buffers(self.input_buffers)
            else:
                prep_engine()
                self.input_buffers = []
                self.recording = True
                self.engine_stop = self.engine.play_lines(self.synth, self.sched, get_background_gain, engine_text_callback)
                play_layers(self.layers)

            self.update_record_screen()

        self.play_button.bind(on_press=play)
        self.save_button.bind(on_press=save)
        self.record_button.bind(on_press=record)

        button_instrument = make_button('   Change\nInstrument', .18, .15, .5, .3, 50)
        button_instrument.bind(on_press=self.go_to_callback('instrument'))

        button_all_tracks = make_button('All Tracks', .18, .15, .7, .3, 50)
        button_all_tracks.bind(on_press=self.go_to_callback('tracks'))

        button_cancel = make_bg_button('Cancel',.1, .1, .85, .02)
        button_cancel.bind(on_press=self.go_to_callback('start'))

        screen.add_widget(self.record_label)
        screen.add_widget(self.play_button)
        screen.add_widget(self.save_button)
        screen.add_widget(self.record_button)
        screen.add_widget(button_cancel)
        screen.add_widget(button_all_tracks)
        screen.add_widget(button_instrument)
        screen.add_widget(self.background_gain_slider)
        screen.add_widget(self.layer_gain_slider)
        screen.add_widget(self.pitch_snap_slider)
        screen.add_widget(self.rhythm_snap_slider)
        screen.add_widget(label_background_gain)
        screen.add_widget(label_layer_gain)
        screen.add_widget(label_pitch_snap)
        screen.add_widget(label_rhythm_snap)       

        self.graph_widget = GraphDisplayWidget(
                size_hint=(.5, .3), pos_hint={'x':.12, 'y':.5})
        self.raw_segments_widget = SegmentsDisplayWidget(
                size_hint=(.5, .3), pos_hint={'x':.12, 'y':.5},
                color= coral)
        self.processed_segments_widget = SegmentsDisplayWidget(
                size_hint=(.5, .3), pos_hint={'x':.12, 'y':.5},
                color= coral) #it's still just red rn?
        screen.add_widget(self.graph_widget)
        screen.add_widget(self.raw_segments_widget)
        screen.add_widget(self.processed_segments_widget)
        self.add_widget(screen)

        self.update_record_screen()


    def go_to_callback(self, name):
        def callback(instance):
            self.current = name
        return callback

    def mood_callback(self, chords, key, rhythm):
        def callback(button):
            self.chords_input.text = ','.join(str(c) for c in chords)
            key_bass, key_mode = key
            self.key_input.text = key_bass
            if key_mode == 'major':
                self.mode_group.press(self.button_major)
            elif key_mode == 'minor':
                self.mode_group.press(self.button_minor)
            else:
                print('error! unrecognized mode: ' + repr(key_mode))
            self.rhythm = rhythm # TODO
            # self.engine.set_chords(chords, key, rhythm)
        return callback

    def receive_audio(self, frames, num_channels) :
        # get one channel from input
        if num_channels == 2:
            mono = frames[self.channel_select::2] # pick left or right channel
        else:
            mono = frames

        # pitch detection: get pitch and display on meter and graph
        self.cur_pitch = self.pitch.write(mono)
        self.graph_widget.graph.add_point(self.cur_pitch)

        if self.recording:
            self.input_buffers.append(frames)

    def on_update(self, dt):
        # self.w1.on_update()
        self.audio.on_update()
    def on_key_down(self, keycode, modifiers):
        self.w1.on_key_down(keycode, modifiers)

REC_INDEX = [1,2,3,4,5,6,7,8,9]
# pass in which MainWidget to run as a command-line arg
run(MainMainWidget1)
