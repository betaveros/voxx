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
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.gridlayout import GridLayout

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

        self.graph = GraphDisplay(self.pos, 300, 30, (30, 90), (.9,.1,.3))
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
light_teal = (0.164, 0.517, 0.552, 0.5)
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
    def __init__(self, instrument, data, gain, pitch_snap, note_ticks):
        self._instrument = instrument
        self._data = data
        self._gain = gain
        self._pitch_snap = pitch_snap
        self._note_ticks = note_ticks
        self._segments_cache = None
        self._rendered_cache = None

    @property
    def instrument(self): return self._instrument
    @instrument.setter
    def instrument(self, i): self._instrument = i; self._rendered_cache = None
    @property
    def data(self): return self._data
    @data.setter
    def data(self, d): self._data = d; self._segments_cache = None
    @property
    def gain(self): return self._gain
    @gain.setter
    def gain(self, g): self._gain = g; self._rendered_cache = None
    @property
    def pitch_snap(self): return self._pitch_snap
    @pitch_snap.setter
    def pitch_snap(self, p): self._pitch_snap = p; self._segments_cache = None
    @property
    def note_ticks(self): return self._note_ticks
    @note_ticks.setter
    def note_ticks(self, n): self._note_ticks = n; self._segments_cache = None

    def process_with(self, engine):
        if self._segments_cache is None:
            data_array = WaveArray(self.data, 1)
            self._segments_cache = engine.process(data_array, self.pitch_snap, self.note_ticks, 100)
            self._rendered_cache = None
        return self._segments_cache
    def render_with(self, engine):
        raw, proc = self.process_with(engine)
        if self._rendered_cache is None:
            self._rendered_cache = engine.render(proc, self.instrument, self.gain)
        return self._rendered_cache

PLAYING = 'playing'
RECORDING = 'recording'
PLAYING_SELECTED = 'playing selected'
PLAYING_ALL = 'playing all'

class MainMainWidget1(ScreenManager):

    def __init__(self):
        super(MainMainWidget1, self).__init__()
        self.audio = Audio(NUM_CHANNELS, input_func=self.receive_audio)
        self.pitch = PitchDetector()

        self.input_mode = None
        self.mood = None

        self.status = None
        self.partial = None
        self.layers = []
        self.cur_layer_index = None
        self.cur_layer = Layer(40, None, 100, 20, 80)

        self.channel_select = 0


        self.make_start_screen()

        self.make_mood_screen_1()
        self.make_progression_screen()

        self.make_input_screen()
        self.make_rhythm_screen()

        self.make_record_screen()
        self.make_tracks_screen()

        self.make_instrument_screen()

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

        self.update_record_screen()
        self.update_chord_template()

    def make_start_screen(self):
        screen = ScreenWithBackground('start')
        label1 = Label(text=u'V\u00f6XX!',
                font_size = 300,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color= dark_teal)

        label2 = Label(text='Set Background Track',
                font_size = 100,
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
                font_size = 120,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=(0, 0.5, 0.6, 1))

        mood_group = CoralButtonGroup()
        def add_mood_button(name, sx, sy, px, py, chords, key, rhythm):
            button = mood_group.make_button(name, sx, sy, px, py,
                    callback=self.mood_callback(chords, key, rhythm))
            screen.add_widget(button)

        add_mood_button('Happy', .15, .15, .08, .4, [6, 4, 1, 5], ['C', 'major'], 240)
        add_mood_button('Sad'  , .15, .15, .31, .4, [1, 7, 5, 4], ['C', 'minor'], 1920)
        add_mood_button('Epic' , .15, .15, .54, .4, [4, 1, 6, 5], ['D', 'major'], 480)
        add_mood_button('Chill', .15, .15, .77, .4, [1, 7, 6, 5], ['F', 'minor'], 960)

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
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)

        label2 = Label(text='Set length of the background loop',
                font_size = 100,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.5},
                font_name = 'fonts/AmaticSC-Regular.ttf',
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
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color= dark_teal)

        label2 = Label(text='for your masterpiece',
                font_size = 100,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.55},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)


        label3 = Label(text='Set Tempo:',
                font_size = 100,
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.5},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        label4 = Label(text='Set Key:',
                font_size = 100,
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.38},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        label5 = Label(text='Set Chords:',
                font_size = 100,
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.26},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        label6 = Label(text='BPM',
                font_size = 80,
                size_hint=(.3, .05), pos_hint={'x':.5, 'y':.5},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        self.bpm_input = IntInput(
                text = '120',
                size_hint=(.2, .1), pos_hint={'x':.4, 'y':.5})
        self.bpm_input.bind(text=lambda instance, value: self.update_chord_template())

        self.key_input = LineTextInput(
                text = 'C',
                size_hint=(.2, .1), pos_hint={'x':.4, 'y':.38})
        self.key_input.bind(text=lambda instance, value: self.update_chord_template())

        self.chords_input = LineTextInput(
                text = '1,4,5,1',
                size_hint=(.5, .1), pos_hint={'x':.4, 'y':.26})
        self.chords_input.bind(text=lambda instance, value: self.update_chord_template())

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
                font_size = 120,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)

        label2 = Label(text='Select the fastest notes you want in your chord',
                font_size = 70,
                size_hint=(.6, .2), pos_hint={'x':.2, 'y':.62},
                font_name = 'fonts/Caveat-Regular.ttf',
                color=dark_teal)


        self.speed_group = CoralButtonGroup()
        button_slow = self.speed_group.make_button('Slow\n (1/4 note)',   .2, .25, .1, .32)
        button_mid  = self.speed_group.make_button('Medium\n (1/8 note)', .2, .25, .4, .32)
        button_fast = self.speed_group.make_button('Fast\n (1/16 note)',  .2, .25, .7, .32)

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
                font_size = 150,
                size_hint=(.7, .2), pos_hint={'x':.15, 'y':.8},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)

        label2 = Label(text='Quick Selection',
                font_size = 100,
                size_hint=(.38, .2), pos_hint={'x':.5, 'y':.65},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=black)


        label3 = Label(text='MIDI Number',
                font_size = 100,
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.65},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=black)

        self.instrument_input = IntInput(
                text = '40', #Violin
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.5})


        instr_group = CoralButtonGroup()
        def add_instrument_button(num, name, sx, sy, px, py):
            def cb(instance):
                self.instrument_input.text = str(num)
            button = instr_group.make_button(name, sx, sy, px, py)
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

    def engine_text_callback(self, i, text):
        self.engine_playing_text = "[{}] {}".format(str(i), text)
        self.update_record_screen()

    def update_record_screen(self):

        self.play_button.disabled = self.status not in [None, PLAYING]
        self.record_button.disabled = self.status not in [None, RECORDING]
        self.play_selected_button.disabled = self.status not in [None, PLAYING_SELECTED]
        self.play_all_button.disabled = self.status not in [None, PLAYING_ALL]

        self.play_button.text = 'Stop' if self.status == PLAYING else 'Play'
        self.record_button.text = 'Stop' if self.status == RECORDING else 'Record'
        self.play_selected_button.text = 'Stop' if self.status == PLAYING_SELECTED else 'Play'
        self.play_all_button.text = 'Stop' if self.status == PLAYING_ALL else 'Play All'

        self.save_button.disabled = self.status is not None or self.cur_layer.data is None

        text = u'{} layer{}'.format(len(self.layers),
                u'' if len(self.layers) == 1 else u's')
        if self.cur_layer_index is None:
            if self.status == RECORDING:
                text += u' + recording'
            elif self.cur_layer.data is not None:
                text += u' + 1'

            if self.status == PLAYING:
                text = u'Playing ' + text
        else:
            text = u'{} of '.format(self.cur_layer_index + 1) + text
            if self.status == RECORDING:
                text = u'Recording ' + text
            elif self.status == PLAYING:
                text = u'Playing all: ' + text
        if self.status in [PLAYING, RECORDING]:
            text += u' ({})'.format(self.engine_playing_text) if self.engine_playing_text else ''
        self.record_label.text = text

    def track_callback(self, i):
        def f(instance):
            if 0 <= i < len(self.layers):
                self.cur_layer_index = i
                self.cur_layer = self.layers[i]
                self.update_record_screen()
                self.current = 'record'
        return f

    def update_all_saved_layers(self):
        for i, tb in enumerate(self.track_buttons):
            tb.disabled = i >= len(self.layers)
        for i, sb in enumerate(self.select_boxes):
            sb.disabled = i >= len(self.layers)
            sb.active = sb.active and i < len(self.layers)

    def make_tracks_screen(self):
        screen = ScreenWithBackground('tracks')

        label1 = Label(text='All Saved Tracks',
                font_size = 100,
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.75},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)
        N=8
        self.y_pos = []
        self.track_buttons = []
        self.select_boxes = []
        self.track_labels = []
        for i in range(N):
            self.y_pos.append(0.65-0.08*i)
            track_button = make_button('',0.7,0.05,0.2,self.y_pos[i])
            track_button.disabled = True
            track_button.bind(on_press=self.track_callback(i))
            self.track_buttons.append(track_button)

            def on_checkbox_active(checkbox, value):
                if value:
                    print('The checkbox', checkbox, 'is active')
                else:
                    print('The checkbox', checkbox, 'is inactive')

            # checkbox.bind(active=on_checkbox_active)
            select_box = CheckBox(size_hint=(0.05, 0.05), pos_hint={'x': 0.13, 'y': self.y_pos[i]}) # TODO
            select_box.disabled = True

            self.select_boxes.append(select_box)

            self.label = Label(text='Track  '+ str(i+1),
                font_size = 40,
                size_hint=(.08, .05), pos_hint={'x':.03, 'y':self.y_pos[i]},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=black)
            self.track_labels.append(self.label)

            screen.add_widget(self.track_buttons[i])
            screen.add_widget(self.select_boxes[i])
            screen.add_widget(self.track_labels[i])

        def play_selected(instance):
            if self.status == PLAYING_SELECTED:
                self.status = None
                self.engine_stop(); self.stop_layers()
            else:
                self.status = PLAYING_SELECTED
                self.engine_stop = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                # play all saved layers plus the current layer
                selected_layers = []
                for i, layer in enumerate(self.layers):
                    if self.select_boxes[i].active:
                        selected_layers.append(layer)
                self.play_layers(selected_layers)

            self.update_record_screen()

        def play_all(instance):
            if self.status == PLAYING_ALL:
                self.status = None
                self.engine_stop(); self.stop_layers()
            else:
                self.status = PLAYING_ALL
                self.engine_stop = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                self.play_layers(self.layers)

            self.update_record_screen()

        self.play_selected_button = make_button('Play', .1, .07, .33, .75, 50)
        self.play_selected_button.bind(on_press=play_selected)
        self.play_all_button = make_button('Play All', .2, .07, .48, .75, 50)
        self.play_all_button.bind(on_press=play_all)

        button_back = make_bg_button('Back', .1, .15, .01, .85)
        button_back.bind(on_press=self.go_to_callback('record'))

        screen.add_widget(label1)
        screen.add_widget(button_back)
        screen.add_widget(self.play_selected_button)
        screen.add_widget(self.play_all_button)
        self.add_widget(screen)

    def update_chord_template(self):
        # type: () -> None
        try:
            bpm = int(self.bpm_input.text)
        except ValueError:
            print('bpm broken: ' + repr(self.bpm_input.text))
            bpm = 120

        try:
            chords = [int(s) for s in self.chords_input.text.split(',')]
        except ValueError:
            print('chords broken: ' + repr(self.chords_input.text))
            chords = [1, 5, 6, 4]

        key_root = self.key_input.text.upper()
        key_mode = 'minor' if self.mode_group.pressed == self.button_minor else 'major'

        self.engine.bpm = bpm
        self.engine.set_chord_template(chords_gen.chord_generater(chords, (key_root, key_mode), self.rhythm))

    def get_note_ticks(self):
        return (40, 60, 80, 120, 240)[int(round(self.layer_note_ticks_slider.value))]

    def play_layers(self, layers):
        self.layers_mixer = Mixer()
        for layer in layers:
            raw_pitches, processed_pitches = layer.process_with(self.engine)
            rendered_data = layer.render_with(self.engine)

            # self.raw_segments_widget.display.set_segments(raw_pitches)
            # self.processed_segments_widget.display.set_segments(processed_pitches)
            self.layers_mixer.add(WaveGenerator(WaveArray(rendered_data, 2)))
        self.mixer.add(self.layers_mixer)

    def get_background_gain(self):
        return int(round(self.background_gain_slider.value))

    def stop_layers(self):
        self.mixer.remove(self.layers_mixer)
        del self.layers_mixer

    def make_record_screen(self):
        screen = ScreenWithBackground('record')

        self.record_label = Label(text='...',
                font_size = 70,
                size_hint=(.3, .1), pos_hint={'x':.35, 'y':.85},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=(0, 0.5, 0.6, 1))

        self.play_button = make_button('Play', .2, .1, .1, .85, 80)

        self.record_button = make_button('Record', .2, .1, .7, .85, 80)

        button_instrument = make_button('   Change\nInstrument', .18, .1, .5, .35, 40)
        button_instrument.bind(on_press=self.go_to_callback('instrument'))

        button_background = make_button('   Change\nProgression', .18, .1, .7, .35, 40)
        button_background.bind(on_press=self.go_to_callback('mood1'))

        self.new_button = make_button('New Track', .18, .1, .5, .23, 45)

        button_all_tracks = make_button('All Tracks', .18, .1, .7, .23, 45)
        button_all_tracks.bind(on_press=self.go_to_callback('tracks'))

        self.delete_button = make_button('Delete Track', .18, .1, .5, .11, 45)

        self.save_button = make_button('Save', .18, .1, .7, .11, 60, bg_color = darker_teal)

        button_cancel = make_bg_button('Cancel',.08, .09, .85, .02, 40)
        button_cancel.bind(on_press=self.go_to_callback('start'))

        self.background_gain_slider = Slider(
                min=0, max=100, value=100, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .1, 'y': .15})
        self.layer_gain_slider = Slider(
                min=0, max=100, value=100, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .2, 'y': .15})
        def change_layer_gain(instance, value):
            self.cur_layer.gain = int(round(value))
        self.layer_gain_slider.bind(value=change_layer_gain)

        self.layer_pitch_snap_slider = Slider(
                min=0, max=20, value=20, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .3, 'y': .15})
        def change_layer_pitch_snap(instance, value):
            self.cur_layer.pitch_snap = value
        self.layer_pitch_snap_slider.bind(value=change_layer_pitch_snap)
        self.layer_note_ticks_slider = Slider(
                min=0, max=4, value=2, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .4, 'y': .15})
        def change_note_ticks_value(instance, value):
            self.cur_layer.note_ticks = self.get_note_ticks()
        self.layer_note_ticks_slider.bind(value=change_note_ticks_value)

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

        def play(instance):
            if self.status == PLAYING:
                self.status = None
                self.engine_stop(); self.stop_layers()
            else:
                self.status = PLAYING
                self.engine_stop = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                # play all saved layers plus the current layer
                layers = self.layers
                cur_layer_is_extra = self.cur_layer_index is None
                cur_layer_has_data = self.cur_layer.data is not None
                if cur_layer_is_extra and cur_layer_has_data:
                    layers = layers + [self.cur_layer]
                self.play_layers(layers)

            self.update_record_screen()

        def save(instance):
            if self.cur_layer_index is None:
                if self.cur_layer.data is not None:
                    self.layers.append(self.cur_layer)
                    self.update_all_saved_layers()

            self.cur_layer_index = None
            self.cur_layer = Layer(self.instrument_input.int_value, None,
                    self.layer_gain_slider.value,
                    self.layer_pitch_snap_slider.value,
                    self.get_note_ticks())

            self.update_record_screen()

        def record(instance):
            if self.status == RECORDING:
                self.status = None
                self.engine_stop(); self.stop_layers()
                self.cur_layer.data = combine_buffers(self.partial.all_buffers)
            else:
                self.status = RECORDING
                self.partial = self.engine.make_partial(self.cur_layer.pitch_snap, self.cur_layer.note_ticks, 100)
                self.engine_stop = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                # play all layers except maybe the current one
                layers = self.layers
                if self.cur_layer_index is not None:
                    layers = layers[:self.cur_layer_index] + layers[self.cur_layer_index + 1:]
                self.play_layers(layers)

            self.update_record_screen()

        self.play_button.bind(on_press=play)
        self.save_button.bind(on_press=save)
        self.record_button.bind(on_press=record)

        #Make the chord bars below recording graph that shows which background chord you're on
        self.nProgression = 4 #This needs to be passed on the total number of chords in the current progression
        self.x_pos = []
        self.chord_bars =[]
        x_start = 0.12
        x_end = 0.88

        for i in range (self.nProgression):
            bar_length = (x_end - x_start)/self.nProgression
            self.x_pos.append(x_start + i * bar_length)
            self.chord_bar = make_button('',bar_length, .05, self.x_pos[i], 0.5)
            self.chord_bars.append(self.chord_bar)
            screen.add_widget(self.chord_bars[i])



        screen.add_widget(self.record_label)
        screen.add_widget(self.play_button)
        screen.add_widget(self.record_button)
        screen.add_widget(self.save_button)
        screen.add_widget(self.delete_button)
        screen.add_widget(self.new_button)
        screen.add_widget(button_cancel)
        screen.add_widget(button_all_tracks)
        screen.add_widget(button_instrument)
        screen.add_widget(button_background)
        screen.add_widget(self.background_gain_slider)
        screen.add_widget(self.layer_gain_slider)
        screen.add_widget(self.layer_pitch_snap_slider)
        screen.add_widget(self.layer_note_ticks_slider)
        screen.add_widget(label_background_gain)
        screen.add_widget(label_layer_gain)
        screen.add_widget(label_pitch_snap)
        screen.add_widget(label_rhythm_snap)

        self.graph_widget = GraphDisplayWidget(
                size_hint=(.05, 2), pos_hint={'x':.9, 'y':.58})
        self.raw_segments_widget = SegmentsDisplayWidget(
                size_hint=(.5, .2), pos_hint={'x':.12, 'y':.58},
                color= coral)
        self.processed_segments_widget = SegmentsDisplayWidget(
                size_hint=(.5, .2), pos_hint={'x':.12, 'y':.58},
                color= coral) #it's still just red rn?
        screen.add_widget(self.graph_widget)
        screen.add_widget(self.raw_segments_widget)
        screen.add_widget(self.processed_segments_widget)
        self.add_widget(screen)

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
            self.rhythm = rhythm

            self.update_chord_template()
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

        if self.status == RECORDING:
            self.partial.append(frames, 2)
            self.raw_segments_widget.display.set_segments(self.partial.all_segments)
            self.processed_segments_widget.display.set_segments(self.partial.all_processed_segments)

    def on_update(self, dt):
        self.audio.on_update()
    def on_key_down(self, keycode, modifiers):
        pass

REC_INDEX = [1,2,3,4,5,6,7,8,9]
# pass in which MainWidget to run as a command-line arg
run(MainMainWidget1)
