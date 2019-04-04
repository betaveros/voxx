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
import demo_chords
MYPY = False
if MYPY: from typing import List, Tuple

EMOTION = [""]
pitch_dic = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
MAJOR_NUMERALS = u'I ii iii IV V vi vii\u00b0'.split()
MINOR_NUMERALS = u'i ii\u00b0 III iv v VI VII'.split()

ROMAN_NUMERAL_DICT = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7}

WHOLE = [[1920]]
HALF = [[960, 960]]
QUA = [[480, 480, 480, 480],[960, 480, 480], [480, 960, 480]]
EIGHTH = [[480, 480, 240, 240, 240, 240], [480, 240, 480, 240, 240, 240], [480, 240, 480, 240, 480], [480, 240, 240, 240, 240, 240, 240], [240, 480, 240, 480, 240, 240], [480, 240, 240, 240, 240, 480], [480, 480, 240, 480, 240]]

RHYTHM = {1920: WHOLE, 960: HALF, 480: QUA, 240: EIGHTH}

class Chord(object):
	def __init__(self, duration, pitches):
		self.duration = duration # type: int
		self.pitches = pitches # type: Dict[int, int]
		# maps pitches in the chord or scale to a weight saying how "in the
		# chord" it is

class DurationText(object):
	def __init__(self, duration, text):
		self.duration = duration
		self.text = text

class ChordTemplate(object):
	def __init__(self, lines, chords, duration_texts):
		# type: (List[List[Tuple[int, int]]], List[Chord], List[DurationText]) -> None
		self.lines = lines # type: List[List[Tuple[int, int]]]
		self.chords = chords # type: List[Chord]
		self.duration_texts = duration_texts # type: List[DurationText]

# Test NoteSequencer: a class that plays a single sequence of notes.
#240 ticks: eighth note, 

def chord_split(s): # split into root and inversion stuff
	if s and s[0].isdigit():
		return s[:1], s[1:]
	for i, c in enumerate(s):
		if c.isdigit():
			return s[:i], s[i:]
	return s, ""

# hopefully supported formats: 1, 2, 3, ...; I, i, II, ii, ...; C, C#, Db, ...
def parse_chord(chord_str, tonic, mode):
	root_str, inv_str = chord_split(chord_str)
	print(root_str, inv_str)
	if mode == 'major':
		scale = MAJOR
		numerals = MAJOR_NUMERALS
	else:
		scale = MINOR
		numerals = MINOR_NUMERALS
	scale_notes = [tonic + i for i in scale]

	major_seven = False

	if not root_str: root_str = '1' # shrug
	if root_str in 'nN': root_str = 'bII6' # shrug

	final_semitone_shift = 0
	while len(root_str) >= 2 and root_str[0] == 'b':
		root_str = root_str[1:]
		final_semitone_shift -= 1
	while len(root_str) >= 2 and root_str[0] == '#':
		root_str = root_str[1:]
		final_semitone_shift += 1

	if root_str.isdigit():
		# classic voxx, invert arbitrarily
		return (
				scale_notes[(int(root_str) - 1) % 7],
				scale_notes[(int(root_str) + 1) % 7],
				scale_notes[(int(root_str) + 3) % 7],
				numerals[(int(root_str) - 1) % 7]
		)
	elif root_str[0] in 'ivIV':
		# roman numeral (probably)
		pure = ''.join(c for c in root_str if c in 'ivIV').upper()
		num = ROMAN_NUMERAL_DICT.get(pure, 1)
		root = scale_notes[num - 1]
		if root_str.endswith('o') or root_str.endswith(u'\u00b0'):
			mid = root + 3
			top = root + 6
			name = pure.lower() + u'\u00b0'
		elif root_str.endswith('+'):
			mid = root + 4
			top = root + 8
			name = pure.upper() + '+'
		elif root_str[0].islower():
			mid = root + 3
			top = root + 7
			name = pure.lower()
		else:
			mid = root + 4
			top = root + 7
			name = pure.upper()
	elif root_str[0] in 'abcdefgABCDEFG':
		root_key = root_str[0].upper()
		root = pitch_dic[root_key]
		quality = "major"
		name = root_key

		for c in root_str[1:]:
			if c == 'b':
				root -= 1
				name += 'b'
			elif c == '#':
				root += 1
				name += '#'
			elif c == 'm':
				if quality == 'major':
					quality = 'minor'
			elif c == 'j':
				quality = 'major'
				major_seven = True
			elif c == 'a':
				quality = 'augmented'
			elif c == 'd':
				quality = 'diminished'

		if quality == 'minor':
			name += 'm'
			mid = root + 3
			top = root + 7
		elif quality == 'diminished':
			name += 'dim'
			mid = root + 3
			top = root + 6
		elif quality == 'augmented':
			name += 'aug'
			mid = root + 4
			top = root + 8
		else:
			mid = root + 4
			top = root + 7

	if inv_str == '6':
		root, mid, top = mid, top, root + 12
		name = name + '6'
	elif inv_str == '64':
		root, mid, top = top - 12, root, mid
		name = name + '64'
	elif inv_str == '7':
		top = root + (11 if major_seven else 10)
		name = name + '7'
	elif inv_str == '65':
		root, mid, top = mid, root + (11 if major_seven else 10), root + 12
		name = name + '65'
	elif inv_str == '43':
		root, mid, top = top - 12, root - (1 if major_seven else 2), root
		name = name + '43'
	elif inv_str == '24' or inv_str == '2':
		root, mid, top = root - (1 if major_seven else 2), root, mid
		name = name + '24'

	root += final_semitone_shift
	mid += final_semitone_shift
	top += final_semitone_shift
	if final_semitone_shift >= 0:
		name = '#'*final_semitone_shift + name
	else:
		name = 'b'*(-final_semitone_shift) + name
	return root, mid, top, name

# chord_strs is a list of chord progressions represented flexibly as strings;
# each number stands for one measure.
# key is a list of strings representing the key, eg: ['c', 'major']
# rhythm is a number stands for the fastest note in the progression eg: 120, 240, 480, 960


def chord_generater(chord_strs, key, rhythm):
	# type: (List[int], Tuple[str, str], int) -> ChordTemplate

	if chord_strs == ["1","2","7","6"] and key == ('F', 'major') and rhythm == 960:
		return demo_chords.demo

	key_root, key_mode = key
	starting_note = pitch_dic[key_root]
	if key_mode == 'major':
		scale = MAJOR
	else:
		scale = MINOR
	all_notes = [starting_note + i for i in range(12)]
	scale_notes = [starting_note + i for i in scale]
	print(scale_notes)
	top_line= []
	mid_line = []
	root_line = []
	names = []
	for chord in chord_strs:
		try:
			r, m, t, name = parse_chord(chord, starting_note, key_mode)
		except Exception as e:
			print(e)
			r, m, t, name = parse_chord('1', starting_note, key_mode)
		if root_line and not (min(root_line) <= r <= max(root_line)):
			if r >= root_line[-1] + 7:
				r -= 12
				m -= 12
				t -= 12
			if r <= root_line[-1] - 7:
				r += 12
				m += 12
				t += 12

		top_line.append(t)
		mid_line.append(m)
		root_line.append(r)
		names.append(name)

	print("top notes", top_line)
	print("mid notes", mid_line)
	print("rt notes", root_line)

	one_measure = RHYTHM[rhythm][random.randint(0, len(RHYTHM[rhythm]) -1 )]

	top_dur_pitch    = [] # type: List[Tuple[int, int]]
	middle_dur_pitch = [] # type: List[Tuple[int, int]]
	root_dur_pitch   = [] # type: List[Tuple[int, int]]
	chords = [] # type: List[Chord]
	duration_texts = [] # type: List[DurationText]
	for x in range(len(top_line)):
		note_total = 0
		for note in one_measure:
			top_dur_pitch.append((note, top_line[x]))
			middle_dur_pitch.append((note, mid_line[x]))
			root_dur_pitch.append((note, root_line[x]))
			pitch_dict = dict((m, 0) for m in all_notes)
			for m in scale_notes: pitch_dict[m] = 1
			pitch_dict[top_line[x]] = 2
			pitch_dict[mid_line[x]] = 2
			pitch_dict[root_line[x]] = 2
			chords.append(Chord(note, pitch_dict))
			note_total += note
		duration_texts.append(DurationText(note_total, names[x]))

	return ChordTemplate([top_dur_pitch, middle_dur_pitch, root_dur_pitch], chords, duration_texts)

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

        progression = chord_generater([1, 3, 6, 4, 2, 7], ['e', 'minor'], 240)
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


if __name__ == "__main__":
    run(MainWidget)
