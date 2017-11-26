#Hacking arts - Xueqi Zhang
# Nov. 11, 2017


import sys
sys.path.append('..')
from common.core import *
from common.audio import *
from common.synth import *

from common.gfxutil import *
from common.clock import *
from common.metro import *
from common.noteseq import *
import random


EMOTION = [""]
pitch_dic = {'c': 60, 'd': 62, 'e': 64, 'f': 65, 'g': 67, 'a': 69, 'b': 71, 'csharp': 61, 'dflat':61, 'dsharp': 63, 'eflat':63, 'fsharp': 66, 'gflat':66, 'gsharp': 68, 'aflat':68, 'asharp': 70, 'bflat':70}
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

WHOLE = [[1920]]
HALF = [[960, 960]]
QUA = [[480, 480, 480, 480],[960, 480, 480], [480, 960, 480]]
EIGHTH = [[480, 480, 240, 240, 240, 240], [480, 240, 480, 240, 240, 240], [480, 240, 480, 240, 480], [480, 240, 240, 240, 240, 240, 240], [240, 480, 240, 480, 240, 240], [480, 240, 240, 240, 240, 480], [480, 480, 240, 480, 240]]

RHYTHM = {1920: WHOLE, 960: HALF, 480: QUA, 240: EIGHTH}



# Test NoteSequencer: a class that plays a single sequence of notes.
#240 ticks: eighth note, 

#chords is a list of chord progressions [1,4,5,1], each number stands for one measure
#key is a list of strings representing the key, eg: ['c', 'major']
#rhythm is a number stands for the fastest note in the progression eg: 120, 240, 480, 960
#tempo is the bpm, eg: 120, 180 etc
def chord_generater(chords, key, rhythm, tempo):
    tempo_param = tempo / 120.0
    starting_note = pitch_dic[key[0]]
    if key[1] == 'major':
        scale = MAJOR
    else:
        scale = MINOR
    scale_notes = []
    for pmt in scale:
        scale_notes.append(starting_note + pmt)
    print(scale_notes)
    top_line= []
    mid_line = []
    root_line = []
    for chord in chords:
        top_line.append(scale_notes[(chord + 3) % 7])
        mid_line.append(scale_notes[(chord + 1) % 7])
        root_line.append(scale_notes[(chord -1) % 7])
    print("top notes", top_line)
    print("mid notes", mid_line)
    print("rt notes", root_line)

    one_measure = RHYTHM[rhythm][random.randint(0, len(RHYTHM[rhythm]) -1 )]

    top_dur_pitch =  []
    middle_dur_pitch =  []
    root_dur_pitch =  []

    which = []
    for x in range(len(top_line)):
        for note in one_measure:
            top_dur_pitch.append([note / tempo_param, top_line[x]])
            middle_dur_pitch.append([note / tempo_param, mid_line[x]])
            root_dur_pitch.append([note / tempo_param, root_line[x]])
            which.append([note / tempo_param, top_line[x], mid_line[x], root_line[x]])

    print(which)
    return [top_dur_pitch, middle_dur_pitch, root_dur_pitch, which]



#emotion is a string representing the emotion: eg: 'happy', 'sad'
#measure is a number 
def progression(emotion, measure):
    pass


class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        self.audio = Audio(2)
        self.synth = Synth('data/FluidR3_GM.sf2')

        # create TempoMap, AudioScheduler
        self.tempo_map  = SimpleTempoMap(120)  #120 bpm
        self.sched = AudioScheduler(self.tempo_map) #scheduler and a clock built into one class

        # connect scheduler into audio system
        self.audio.set_generator(self.sched) 
        self.sched.set_generator(self.synth)

        # create the metronome:
        self.metro = Metronome(self.sched, self.synth)

        progression = chord_generater([1, 4, 5, 1], ['eflat', 'minor'], 240, 129)
        # create a NoteSequencer:
        self.seq = NoteSequencer(self.sched, self.synth, 1, (0,0), progression[0])
        self.seq1 = NoteSequencer(self.sched, self.synth, 2, (0,0), progression[1])
        self.seq2 = NoteSequencer(self.sched, self.synth, 3, (0,0), progression[2])

        # and text to display our status
        self.label = topleft_label()
        self.add_widget(self.label)

    

    def on_key_down(self, keycode, modifiers):
        if keycode[1] == 'm':
            self.metro.toggle()

        if keycode[1] == 's':
            self.seq.toggle()
            self.seq1.toggle()
            self.seq2.toggle()

        bpm_adj = lookup(keycode[1], ('up', 'down'), (10, -10)) #where is the lookup function defined
        if bpm_adj: #adjustment
            new_tempo = self.tempo_map.get_tempo() + bpm_adj
            self.tempo_map.set_tempo(new_tempo, self.sched.get_time()) #set_tempo takes in a tempo in seconds

    def on_update(self) :
        self.audio.on_update()
        self.label.text = self.sched.now_str() + '\n'
        self.label.text += 'tempo:%d\n' % self.tempo_map.get_tempo()
        self.label.text += 'm: toggle Metronome\n'
        self.label.text += 's: toggle Sequence\n'
        self.label.text += 'up/down: change speed\n'        


run(MainWidget)

