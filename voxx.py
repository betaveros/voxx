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
import midi_names

import random
from pitch_detector import PitchDetector

NUM_CHANNELS = 2

CHORD_CHANNEL = 1

add_mic_graph = False
font_mult = 1
def font_size_for(x): return int(round(font_mult * x))

for arg in sys.argv:
    if arg.startswith("fs="):
        font_mult = float(arg[3:])
    if arg == 'mic':
        add_mic_graph = True

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
                print('IOBuffer:overrun')
            else:
                print('IOBuffer:underrun')

        else:
            tmp = self.buffer

        # clear out buffer because we just used it
        self.buffer = None
        return tmp, True

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

class ChordBar(BaseWidget):
    def __init__(self, **kwargs):
        super(ChordBar, self).__init__(**kwargs)

        # self.size = kwargs['size']
        # self.pos = kwargs['pos']
        self.bg_color = kwargs['bg_color']
        self.fg_color = kwargs['fg_color']
        self.bg_rect = Rectangle(size=self.size, pos=self.pos)
        self.fg_rect = Rectangle(size=(0, 0), pos=self.pos)
        g = InstructionGroup()
        # g.add(PushMatrix())
        g.add(Color(*self.bg_color))
        g.add(self.bg_rect)
        g.add(Color(*self.fg_color))
        g.add(self.fg_rect)
        # g.add(PopMatrix())
        self.canvas.add(g)
        self.index = None
        self.total = None
        self.bind(pos=self.redraw)

    def set_chord(self, index, total):
        self.index = index
        self.total = total
        self.recompute()

    def recompute(self):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        if self.index is None or self.total is None:
            self.fg_rect.size = (0, 0)
        else:
            x, y = self.pos
            w, h = self.size
            self.fg_rect.pos = (x + self.index*(1.0*w/self.total), y)
            self.fg_rect.size = (1.0*w/self.total, h)

    def redraw(self, x, y):
        self.recompute()

class SegmentsDisplay(InstructionGroup):
    """Displays a sequence of horizontal segments.

    segments is a list of (value, length) pairs."""
    def __init__(self, pos, height, x_scale, segments, in_range, color):
        super(SegmentsDisplay, self).__init__()

        self.height = height
        self.x_scale = x_scale
        self.range = in_range

        self.translate = Translate(*pos)
        self.line = Line( width = 1.5 )
        self.now_bar = Line( width = 1.5 )
        self.add(PushMatrix())
        self.add(self.translate)
        self.add(Color(*color))
        self.add(self.line)
        self.add(Color(1, 1, 1))
        self.add(self.now_bar)
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

    def set_now(self, now):
        x = self.x_scale * now
        self.now_bar.points = [x, 0, x, self.height]

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
                font_size = font_size_for(100),
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


def make_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size=font_size_for(50), color=light_pink, bg_color=dark_teal, font_name = 'fonts/AmaticSC-Regular.ttf'):
    return Button(text=text, font_size=font_size, color=color, size_hint=(size_hint_x, size_hint_y), pos_hint={'x': pos_hint_x, 'y': pos_hint_y}, background_normal='', background_color=bg_color, font_name = font_name)

class CoralButtonGroup(object):
    def __init__(self):
        super(CoralButtonGroup, self).__init__()
        self.pressed = None
        self.cb = None

    def press(self, button):
        if self.pressed is None:
            button.background_color = coral
            self.pressed = button
        if button != self.pressed:
            self.pressed.background_color = dark_teal
            button.background_color = coral
            self.pressed = button

        if self.cb: self.cb()

    def clear_pressed(self):
        self.pressed.background_color = dark_teal
        self.pressed = None

        if self.cb: self.cb()

    def make_button(self, text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size=font_size_for(90), callback=None):
        button = make_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size)
        def cur_press(instance):
            if callback is not None: callback(instance)
            self.press(instance)
        button.bind(on_press=cur_press)
        return button

    def bind_press(self, cb):
        self.cb = cb

def make_bg_button(text, size_hint_x, size_hint_y, pos_hint_x, pos_hint_y, font_size=font_size_for(80)):
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

    def clear_cache(self): self._segments_cache = None

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
        self.cur_layer = Layer(40, None, 100, 0, 40)

        self.channel_select = 0


        self.make_start_screen()

        self.make_mood_screen_1()
        #self.make_progression_screen()

        self.make_input_screen()
        self.make_rhythm_screen()

        self.make_record_screen()
        self.make_tracks_screen()

        self.make_instrument_screen()

        self.engine = VoxxEngine()
        self.engine_status = None

        self.mixer = Mixer()
        self.mixer.set_gain(1)
        try:
            self.synth = Synth('data/FluidR3_GM.sf2')
        except Exception as e:
            raise Exception("Error opening Synth from data/FluidR3_GM.sf2. Please find the Fluid Synth Sound Font online and put it in the data directory.")
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
                font_size = font_size_for(300),
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                color= dark_teal)

        label2 = Label(text='Set Background Track',
                font_size = font_size_for(100),
                size_hint=(.7, .3), pos_hint={'x':.15, 'y':.4},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=dark_teal)

        g = CoralButtonGroup()
        button1 = g.make_button('Templates',  .25, .15, .2 , .25, 100, self.go_to_callback('mood1'))
        button2 = g.make_button('Advanced', .25, .15, .55, .25, 100, self.go_to_callback('input'))

        button3 = make_bg_button('Skip Background Track', .25, .15, .7, .04, 70)
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
                font_size = font_size_for(120),
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.6},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=(0, 0.5, 0.6, 1))

        mood_group = CoralButtonGroup()
        def add_mood_button(name, sx, sy, px, py, chords, key, rhythm_template):
            button = mood_group.make_button(name, sx, sy, px, py,
                    callback=self.mood_callback(chords, key, rhythm_template))
            screen.add_widget(button)

        add_mood_button('Happy', .15, .15, .08, .4, [6, 4, 1, 5], ['C', 'major'], chords_gen.RhythmTemplate.from_string(8, "x-xxxxxx x-xx-xx- x-xx-xx-"))
        add_mood_button('Sad'  , .15, .15, .31, .4, [1, 7, 5, 4], ['C', 'minor'], chords_gen.RhythmTemplate.from_string(1, "x x x"))
        add_mood_button('Epic' , .15, .15, .54, .4, [4, 1, 6, 5], ['D', 'major'], chords_gen.RhythmTemplate.from_string(4, "x-xx x-xx x--x"))
        add_mood_button('Chill', .15, .15, .77, .4, [1, 2, 7, 6], ['F', 'major'], chords_gen.DemoRhythmTemplate())

        button_back = make_bg_button('Back', .1, .15, .01, .02)
        button_next = make_bg_button('Next', .1, .15, .89, .02)
        button_advanced = make_bg_button('Advanced', .2, .15, .4, .02)

        button_next.bind(on_press=self.go_to_callback('instrument'))
        button_back.bind(on_press=self.go_to_callback('start'))
        button_advanced.bind(on_press=self.go_to_callback('input'))

        screen.add_widget(label)
        screen.add_widget(button_back)
        screen.add_widget(button_next)
        screen.add_widget(button_advanced)
        self.add_widget(screen)

    # def make_progression_screen(self):
    #     screen = ScreenWithBackground('length')
    #     label1 = Label(text='Almost done!',
    #             font_size = 150,
    #             size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
    #             font_name = 'fonts/AmaticSC-Bold.ttf',
    #             color=dark_teal)

    #     label2 = Label(text='Set length of the background loop',
    #             font_size = 100,
    #             size_hint=(.5, .3), pos_hint={'x':.25, 'y':.5},
    #             font_name = 'fonts/AmaticSC-Regular.ttf',
    #             color=dark_teal)

    #     self.measure_group = CoralButtonGroup()
    #     self.button_short = self.measure_group.make_button('Short\n (4 measures)',  .2, .15, .1, .3)
    #     self.button_mid   = self.measure_group.make_button('Medium\n (6 measures)', .2, .15, .4, .3)
    #     self.button_long  = self.measure_group.make_button('Long\n (8 measures)',   .2, .15, .7, .3)

    #     button_back = make_bg_button('Back', .1, .15, .01, .02)
    #     button_next = make_bg_button('Next', .1, .15, .89, .02)
    #     button_advanced = make_bg_button('Advanced', .2, .15, .4, .02)


    #     button_next.bind(on_press=self.go_to_callback('instrument'))
    #     button_back.bind(on_press=self.go_to_callback('mood1'))
    #     button_advanced.bind(on_press=self.go_to_callback('input'))


    #     screen.add_widget(label1)
    #     screen.add_widget(label2)
    #     screen.add_widget(self.button_short)
    #     screen.add_widget(self.button_mid)
    #     screen.add_widget(self.button_long)
    #     screen.add_widget(button_back)
    #     screen.add_widget(button_next)
    #     screen.add_widget(button_advanced)
    #     self.add_widget(screen)

    def finish_set_instrument(self, instance):
        self.cur_layer.instrument = self.instrument_input.int_value
        self.current = 'record'
        self.update_all_saved_layers()

    def make_input_screen(self):
        screen = ScreenWithBackground('input')
        label1 = Label(text='Advanced Settings',
                font_size = font_size_for(150),
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color= dark_teal)

        label2 = Label(text='for your masterpiece',
                font_size = font_size_for(100),
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.55},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)


        label3 = Label(text='Set Tempo:',
                font_size = font_size_for(100),
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.5},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        label4 = Label(text='Set Key:',
                font_size = font_size_for(100),
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.38},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        label5 = Label(text='Set Chords:',
                font_size = font_size_for(100),
                size_hint=(.3, .1), pos_hint={'x':.1, 'y':.26},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color= black)

        label6 = Label(text='BPM',
                font_size = font_size_for(80),
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

        # FIXME defaults are dangerous
        self.rhythm_template = chords_gen.RhythmTemplate.randomize(8, regular=False, dense=False, unison=False)

        button_back = make_bg_button('Back', .1, .15, .01, .02)
        button_next = make_bg_button('Next', .1, .15, .89, .02)

        button_next.bind(on_press=self.go_to_callback('rhythm'))
        button_back.bind(on_press=self.go_to_callback('mood1'))

        self.mode_group = CoralButtonGroup()
        self.button_major = self.mode_group.make_button('Major', .13, .1, .62, .38, 70)
        self.button_minor = self.mode_group.make_button('Minor', .13, .1, .78, .38, 70)

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
        # FIXME
        screen = ScreenWithBackground('rhythm')
        label1 = Label(text='What rhythm would you like?',
                font_size = font_size_for(120),
                size_hint=(.5, .3), pos_hint={'x':.25, 'y':.7},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)

        label2 = Label(text='Select the fastest notes you want in your chord',
                font_size = font_size_for(70),
                size_hint=(.6, .2), pos_hint={'x':.2, 'y':.65},
                font_name = 'fonts/Caveat-Regular.ttf',
                color=dark_teal)


        self.speed_group = CoralButtonGroup()
        self.button_slow = self.speed_group.make_button('Slow\n(1/4 note)', .2, .22, .16, .48)
        self.button_mid  = self.speed_group.make_button('Medium\n(1/8 note)', .2, .22, .39, .48)
        self.button_fast = self.speed_group.make_button('Fast\n(1/16 note)', .2, .22, .62, .48)

        self.regularity_group = CoralButtonGroup()
        self.button_irregular = self.regularity_group.make_button('Irregular', .2, .1, .295, .35)
        self.button_regular   = self.regularity_group.make_button('Regular',   .2, .1, .505, .35)

        self.density_group = CoralButtonGroup()
        self.button_sparse = self.density_group.make_button('Sparse', .2, .1, .295, .23)
        self.button_dense  = self.density_group.make_button('Dense',  .2, .1, .505, .23)

        self.unison_group = CoralButtonGroup()
        self.button_staggered = self.unison_group.make_button('Staggered', .2, .1, .295, .11)
        self.button_unison    = self.unison_group.make_button('Unison',    .2, .1, .505, .11)

        for button in [self.button_slow, self.button_mid, self.button_fast, self.button_regular, self.button_irregular, self.button_dense, self.button_sparse, self.button_staggered, self.button_unison]:
            screen.add_widget(button)

        for group in [self.speed_group, self.regularity_group, self.density_group, self.unison_group]:
            group.bind_press(self.update_rhythm_template)

        button_preview = make_button('Preview', .2, .06, 0.4, 0.02)

        def preview_rhythm(instance):
            try:
                bpm = int(self.bpm_input.text)
            except ValueError:
                print('bpm broken: ' + repr(self.bpm_input.text))
                bpm = 120
            chords = self.chords_input.text.split(',')[:1] or ["1"]

            key_root = self.key_input.text.upper()
            key_mode = 'minor' if self.mode_group.pressed == self.button_minor else 'major'

            self.mixer.add(WaveGenerator(WaveArray(self.engine.render_chord_demo(bpm,
                chords_gen.chord_generater(chords, (key_root, key_mode), self.rhythm_template)), 2)))
        button_preview.bind(on_press=preview_rhythm)

        button_back = make_bg_button('Back', .1, .15, .01, .02)

        button_next = make_bg_button('Next', .1, .15, .89, .02)

        button_next.bind(on_press=self.go_to_callback('instrument'))
        button_back.bind(on_press=self.go_to_callback('input'))

        # self.rhythm_template = RhythmTemplate(8, regular=False, dense=False, unison=False)

        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(button_preview)
        screen.add_widget(button_back)
        screen.add_widget(button_next)
        self.add_widget(screen)


    def make_instrument_screen(self):
        screen = ScreenWithBackground('instrument')
        label1 = Label(text='Select Instrument',
                font_size = font_size_for(150),
                size_hint=(.7, .2), pos_hint={'x':.15, 'y':.8},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=dark_teal)

        label2 = Label(text='Quick Selection',
                font_size = font_size_for(100),
                size_hint=(.38, .2), pos_hint={'x':.5, 'y':.65},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=black)


        label3 = Label(text='MIDI Number',
                font_size = font_size_for(100),
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.65},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=black)

        first_instrument = 40 # violin
        self.instrument_input = IntInput(
                text = str(first_instrument),
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.5})

        instrument_label = Label(
                text=midi_names.name_for(first_instrument),
                font_size = font_size_for(100),
                size_hint=(.18, .15), pos_hint={'x':.15, 'y':.35},
                font_name = 'fonts/AmaticSC-Regular.ttf',
                color=black)

        def instrument_change(instance, value):
            try:
                instrument_label.text = midi_names.name_for(int(value))
            except ValueError:
                instrument_label.text = '???'
        self.instrument_input.bind(text=instrument_change)


        instr_group = CoralButtonGroup()
        def add_instrument_button(num, name, sx, sy, px, py):
            def cb(instance):
                self.instrument_input.text = str(num)
            button = instr_group.make_button(name, sx, sy, px, py,75)
            button.bind(on_press=cb)
            screen.add_widget(button)

        add_instrument_button( 0, 'Piano'    , .18, .15, .5,  .5)
        add_instrument_button(24, 'Guitar'   , .18, .15, .7,  .5)
        add_instrument_button(40, 'Violin'   , .18, .15, .5, .32)
        add_instrument_button(42, 'Cello'    , .18, .15, .7, .32)
        add_instrument_button(32, 'Bass'     , .18, .15, .5, .14)
        add_instrument_button(65, 'Saxophone', .18, .15, .7, .14)

        def preview_instrument(instance):
            instr = self.instrument_input.int_value
            self.mixer.add(WaveGenerator(WaveArray(self.engine.render_demo(instr), 2)))

        button_preview = make_button('Preview', .18, .15, .08, .14, 90, bg_color = darker_teal)
        button_preview.bind(on_press=preview_instrument)
        button_done    = make_button('Done', .18, .15, .28, .14, 90, bg_color = darker_teal)
        button_done.bind(on_press=self.finish_set_instrument)


        button_cancel = make_bg_button('Cancel',.1, .1, .85, .02)
        button_cancel.bind(on_press=self.go_to_callback('start'))

        screen.add_widget(label1)
        screen.add_widget(label2)
        screen.add_widget(label3)
        screen.add_widget(self.instrument_input)
        screen.add_widget(instrument_label)
        screen.add_widget(button_preview)
        screen.add_widget(button_done)
        screen.add_widget(button_cancel)


        self.add_widget(screen)

    def engine_text_callback(self, i, total, text):
        self.chord_bar.set_chord(i, total)
        self.engine_playing_text = text
        self.update_record_screen()

    def update_record_screen(self):

        self.play_button.disabled = self.status not in [None, PLAYING]
        self.record_button.disabled = self.status not in [None, RECORDING]
        self.play_selected_button.disabled = self.status not in [None, PLAYING_SELECTED]
        self.play_all_button.disabled = self.status not in [None, PLAYING_ALL]

        self.button_instrument.disabled = self.status is not None
        self.button_background.disabled = self.status is not None
        # self.button_all_tracks.disabled = self.status is not None

        self.play_button.text = 'Stop' if self.status == PLAYING else 'Play'
        self.record_button.text = 'Stop' if self.status == RECORDING else 'Record'
        self.play_selected_button.text = 'Stop' if self.status == PLAYING_SELECTED else 'Play'
        self.play_all_button.text = 'Stop' if self.status == PLAYING_ALL else 'Play All'

        self.save_button.disabled = self.status is not None or self.cur_layer_index is not None or self.cur_layer.data is None
        self.new_button.disabled = self.status is not None or self.cur_layer_index is None
        self.all_new_layer_button.disabled =  self.status is not None or self.cur_layer_index is None
        self.delete_button.disabled = self.status is not None or (self.cur_layer_index is None and self.cur_layer.data is None)

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
            text += u' ({})'.format(self.engine_playing_text) if self.engine_playing_text else u''
        self.record_label.text = text

    def track_callback(self, i):
        def f(instance):
            if 0 <= i < len(self.layers):
                self.cur_layer_index = i
                self.cur_layer = self.layers[i]
                raw_pitches, processed_pitches = self.cur_layer.process_with(self.engine)
                self.raw_segments_widget.display.set_segments(raw_pitches)
                self.processed_segments_widget.display.set_segments(processed_pitches)
                if self.engine_status and self.engine_status.stop_flag: self.engine_status = None
                self.update_sliders_from_layer()
                self.update_record_screen()
                self.current = 'record'
        return f

    def update_all_saved_layers(self):
        counter = Counter()
        for i, tb in enumerate(self.track_buttons):
            if i < len(self.layers):
                tb.disabled = False
                instrument = self.layers[i].instrument
                counter[instrument] += 1
                tb.text = "{} {}".format(midi_names.name_for(instrument), counter[instrument])
            else:
                tb.disabled = True
                tb.text = ""
        for i, sb in enumerate(self.select_boxes):
            sb.disabled = i >= len(self.layers)
            sb.active = sb.active and i < len(self.layers)

    def make_tracks_screen(self):
        screen = ScreenWithBackground('tracks')

        label1 = Label(text='All Saved Tracks',
                font_size = font_size_for(100),
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
                font_size = font_size_for(40),
                size_hint=(.08, .05), pos_hint={'x':.03, 'y':self.y_pos[i]},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=black)
            self.track_labels.append(self.label)

            screen.add_widget(self.track_buttons[i])
            screen.add_widget(self.select_boxes[i])
            screen.add_widget(self.track_labels[i])

        def play_selected(instance):
            if self.status == PLAYING_SELECTED:
                self.stop()
            else:
                self.status = PLAYING_SELECTED
                self.engine_status = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                # play all saved layers plus the current layer
                selected_layers = []
                for i, layer in enumerate(self.layers):
                    if self.select_boxes[i].active:
                        selected_layers.append(layer)
                self.play_layers(selected_layers)

            self.update_record_screen()

        def play_all(instance):
            if self.status == PLAYING_ALL:
                self.stop()
            else:
                self.status = PLAYING_ALL
                self.engine_status = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                self.play_layers(self.layers)

            self.update_record_screen()

        def export(instance):
            export_mixer = Mixer()
            size = 0
            for layer in self.layers:
                rendered_data = layer.render_with(self.engine)
                size = max(size, rendered_data.size)
                export_mixer.add(WaveGenerator(WaveArray(rendered_data, 2)))

            rendered_chords = self.engine.render_chords(size // 2, self.get_background_gain())
            size = max(size, rendered_chords.size)
            export_mixer.add(WaveGenerator(WaveArray(rendered_chords, 2)))

            writer = AudioWriter('processed')
            writer.start()
            data, _ = export_mixer.generate(size // 2, 2)
            writer.add_audio(data, 2)
            writer.stop()

        self.play_selected_button = make_button('Play', .1, .07, .25, .75, 60)
        self.play_selected_button.bind(on_press=play_selected)
        self.play_all_button = make_button('Play All', .2, .07, .4, .75, 60)
        self.play_all_button.bind(on_press=play_all)
        self.all_new_layer_button = make_button('New', .1, .07, .65, .75, 60)
        self.all_new_layer_button.bind(on_press=self.new_layer)
        self.export_button = make_button('Export', .1, .07, .85, .9, 60)
        self.export_button.bind(on_press=export)

        button_back = make_bg_button('Back', .1, .15, .01, .85)
        button_back.bind(on_press=self.go_to_callback('record'))

        screen.add_widget(label1)
        screen.add_widget(button_back)
        screen.add_widget(self.play_selected_button)
        screen.add_widget(self.play_all_button)
        screen.add_widget(self.all_new_layer_button)
        screen.add_widget(self.export_button)
        self.add_widget(screen)

    def update_rhythm_template(self):
        beat = 8
        if self.speed_group.pressed == self.button_slow:
            beat = 4
        elif self.speed_group.pressed == self.button_fast:
            beat = 16
        self.rhythm_template = chords_gen.RhythmTemplate.randomize(
                beats_per_measure = beat,
                regular = self.regularity_group.pressed == self.button_regular,
                dense = self.density_group.pressed == self.button_dense,
                unison = self.unison_group.pressed == self.button_unison,
                )
        self.update_chord_template()

    def update_chord_template(self):
        # type: () -> None
        try:
            bpm = int(self.bpm_input.text)
        except ValueError:
            print('bpm broken: ' + repr(self.bpm_input.text))
            bpm = 120

        chords = self.chords_input.text.split(',')

        key_root = self.key_input.text.upper()
        key_mode = 'minor' if self.mode_group.pressed == self.button_minor else 'major'

        self.engine.bpm = bpm
        self.tempo_map.bpm = bpm # !?!?
        self.engine.set_chord_template(chords_gen.chord_generater(chords, (key_root, key_mode), self.rhythm_template))

    def get_note_ticks(self):
        return (40, 60, 80, 120, 240)[int(round(self.layer_note_ticks_slider.value))]

    def clear_segments_display(self):
        self.raw_segments_widget.display.set_segments([])
        self.processed_segments_widget.display.set_segments([])
        if self.engine_status and self.engine_status.stop_flag: self.engine_status = None

    def play_layers(self, layers, focus=None):
        self.layers_mixer = Mixer()
        self.layers_mixer.set_gain(1)
        self.clear_segments_display()
        for layer in layers:
            raw_pitches, processed_pitches = layer.process_with(self.engine)
            rendered_data = layer.render_with(self.engine)

            if layer is focus:
                self.raw_segments_widget.display.set_segments(raw_pitches)
                self.processed_segments_widget.display.set_segments(processed_pitches)
            self.layers_mixer.add(WaveGenerator(WaveArray(rendered_data, 2)))
        self.mixer.add(self.layers_mixer)

    def get_background_gain(self):
        return int(round(self.background_gain_slider.value))

    def stop(self):
        self.status = None
        self.engine_status.stop()
        self.mixer.remove(self.layers_mixer)
        del self.layers_mixer
        self.chord_bar.set_chord(None, None)

    def new_layer(self, instance):
        self.cur_layer_index = None
        self.cur_layer = Layer(self.instrument_input.int_value, None,
                int(round(self.layer_gain_slider.value)),
                self.layer_pitch_snap_slider.value,
                self.get_note_ticks())
        self.clear_segments_display()
        self.update_all_saved_layers()
        self.update_record_screen()
        self.current = 'record'

    def update_sliders_from_layer(self):
        self.layer_gain_slider.value = self.cur_layer.gain
        self.layer_pitch_snap_slider.value = self.cur_layer.pitch_snap
        try:
            self.layer_note_ticks_slider.value = (40, 60, 80, 120, 240).index(self.cur_layer.note_ticks)
        except ValueError:
            self.layer_note_ticks_slider.value = 2 # idk

    def make_record_screen(self):
        screen = ScreenWithBackground('record')

        self.record_label = Label(text='...',
                font_size = font_size_for(70),
                size_hint=(.3, .1), pos_hint={'x':.35, 'y':.85},
                font_name = 'fonts/AmaticSC-Bold.ttf',
                color=(0, 0.5, 0.6, 1))

        self.play_button = make_button('Play', .2, .1, .1, .85, 90)

        self.record_button = make_button('Record', .2, .1, .7, .85, 90)

        self.button_instrument = make_button('   Change\nInstrument', .18, .1, .5, .35, 45)
        self.button_instrument.bind(on_press=self.go_to_callback('instrument'))

        self.button_background = make_button('   Change\nProgression', .18, .1, .7, .35, 45)
        self.button_background.bind(on_press=self.go_to_callback('mood1'))

        self.new_button = make_button('New Track', .18, .1, .5, .23, 65)

        self.button_all_tracks = make_button('All Tracks', .18, .1, .7, .23, 65)
        self.button_all_tracks.bind(on_press=self.go_to_callback('tracks'))

        self.delete_button = make_button('Delete Track', .18, .1, .5, .11, 65)

        self.save_button = make_button('Save', .18, .1, .7, .11, 80, bg_color = darker_teal)

        button_cancel = make_bg_button('Cancel',.08, .09, .85, .02, 60)
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
                min=0, max=1, value=0, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .3, 'y': .15})
        def change_layer_pitch_snap(instance, value):
            self.cur_layer.pitch_snap = value
        self.layer_pitch_snap_slider.bind(value=change_layer_pitch_snap)
        self.layer_note_ticks_slider = Slider(
                min=0, max=4, value=0, orientation='vertical',
                size_hint=(.1, .3),
                pos_hint={'x': .4, 'y': .15})
        def change_note_ticks_value(instance, value):
            self.cur_layer.note_ticks = self.get_note_ticks()
        self.layer_note_ticks_slider.bind(value=change_note_ticks_value)

        label_background_gain = Label(text='background\nvolume',
                font_size = font_size_for(30),
                size_hint=(.1, .05), pos_hint={'x':.1, 'y':.05},
                color=dark_teal)

        label_layer_gain = Label(text='solo\nvolume',
                font_size = font_size_for(30),
                size_hint=(.1, .05), pos_hint={'x':.2, 'y':.05},
                color=dark_teal)

        label_pitch_snap = Label(text='pitch\nsnap',
                font_size = font_size_for(30),
                size_hint=(.1, .05), pos_hint={'x':.3, 'y':.05},
                color=dark_teal)

        label_rhythm_snap = Label(text='rhythm\nsnap',
                font_size = font_size_for(30),
                size_hint=(.1, .05), pos_hint={'x':.4, 'y':.05},
                color=dark_teal)


        self.engine_playing_text = ""

        def play(instance):
            if self.status == PLAYING:
                self.stop()
            else:
                self.status = PLAYING
                self.engine_status = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                # play all saved layers plus the current layer
                layers = self.layers
                cur_layer_is_extra = self.cur_layer_index is None
                cur_layer_has_data = self.cur_layer.data is not None
                if cur_layer_is_extra and cur_layer_has_data:
                    layers = layers + [self.cur_layer]
                self.play_layers(layers, self.cur_layer)

            self.update_record_screen()

        def save(instance):
            if self.cur_layer_index is None:
                if self.cur_layer.data is not None:
                    self.layers.append(self.cur_layer)
                    self.cur_layer_index = len(self.layers) - 1
                    self.update_all_saved_layers()

            self.update_record_screen()

        def delete_layer(instance):
            if self.cur_layer_index is not None:
                del self.layers[self.cur_layer_index]

            self.cur_layer_index = None
            self.cur_layer = Layer(self.instrument_input.int_value, None,
                    int(round(self.layer_gain_slider.value)),
                    self.layer_pitch_snap_slider.value,
                    self.get_note_ticks())
            self.clear_segments_display()
            # self.update_sliders_from_layer() # don't need, same values.
            self.update_all_saved_layers()
            self.update_record_screen()

        def record(instance):
            if self.status == RECORDING:
                self.stop()
                self.cur_layer.data = combine_buffers(self.partial.all_buffers)
            else:
                self.status = RECORDING
                self.partial = self.engine.make_partial(self.cur_layer.pitch_snap, self.cur_layer.note_ticks, 100)
                self.engine_status = self.engine.play_lines(self.synth, self.sched, self.get_background_gain, self.engine_text_callback)
                # play all layers except maybe the current one
                layers = self.layers
                if self.cur_layer_index is not None:
                    layers = layers[:self.cur_layer_index] + layers[self.cur_layer_index + 1:]
                self.play_layers(layers)

            self.update_record_screen()

        self.play_button.bind(on_press=play)
        self.save_button.bind(on_press=save)
        self.new_button.bind(on_press=self.new_layer)
        self.delete_button.bind(on_press=delete_layer)
        self.record_button.bind(on_press=record)

        #Make the chord bars below recording graph that shows which background chord you're on
        self.chord_bar = ChordBar(bg_color=dark_teal, fg_color=coral, size_hint=(0.76, 0.05), pos_hint={'x': 0.12, 'y': 0.5})
        screen.add_widget(self.chord_bar)

        screen.add_widget(self.record_label)
        screen.add_widget(self.play_button)
        screen.add_widget(self.record_button)
        screen.add_widget(self.save_button)
        screen.add_widget(self.delete_button)
        screen.add_widget(self.new_button)
        screen.add_widget(button_cancel)
        screen.add_widget(self.button_all_tracks)
        screen.add_widget(self.button_instrument)
        screen.add_widget(self.button_background)
        screen.add_widget(self.background_gain_slider)
        screen.add_widget(self.layer_gain_slider)
        screen.add_widget(self.layer_pitch_snap_slider)
        screen.add_widget(self.layer_note_ticks_slider)
        screen.add_widget(label_background_gain)
        screen.add_widget(label_layer_gain)
        screen.add_widget(label_pitch_snap)
        screen.add_widget(label_rhythm_snap)

        self.raw_segments_widget = SegmentsDisplayWidget(
                size_hint=(.5, .2), pos_hint={'x':.12, 'y':.58},
                color= bright_blue)
        self.processed_segments_widget = SegmentsDisplayWidget(
                size_hint=(.5, .2), pos_hint={'x':.12, 'y':.58},
                color= coral) #it's still just red rn?
        if add_mic_graph:
            self.graph_widget = GraphDisplayWidget(
                    size_hint=(.05, 2), pos_hint={'x':.9, 'y':.58})
            screen.add_widget(self.graph_widget)
        else:
            self.graph_widget = None
        screen.add_widget(self.raw_segments_widget)
        screen.add_widget(self.processed_segments_widget)
        self.add_widget(screen)

    def go_to_callback(self, name):
        def callback(instance):
            self.current = name
        return callback

    def mood_callback(self, chords, key, rhythm_template):
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
            self.rhythm_template = rhythm_template

            for layer in self.layers:
                layer.clear_cache()

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
        if self.graph_widget:
            self.graph_widget.graph.add_point(self.cur_pitch)

        if self.status == RECORDING:
            self.partial.append(frames, 2)
            self.raw_segments_widget.display.set_segments(self.partial.all_segments)
            self.processed_segments_widget.display.set_segments(self.partial.all_processed_segments)

    def on_update(self, dt):
        self.audio.on_update()
        self.processed_segments_widget.display.set_now(self.engine_status.get_frame() if self.engine_status else 0)

    def on_key_down(self, keycode, modifiers):
        pass

REC_INDEX = [1,2,3,4,5,6,7,8,9]
# pass in which MainWidget to run as a command-line arg



run(MainMainWidget1)
