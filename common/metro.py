#####################################################################
#
# metro.py
#
# Copyright (c) 2016, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

from common.clock import kTicksPerQuarter, quantize_tick_up

class Metronome(object):
    """Plays a steady click every beat.
    """
    def __init__(self, sched, synth, channel = 0, patch=(128, 0), pitch = 60):
        super(Metronome, self).__init__()
        self.sched = sched
        self.synth = synth
        self.channel = channel
        self.patch = patch
        self.pitch = pitch
        self.beat_len = kTicksPerQuarter

        # run-time variables
        self.on_cmd = None
        self.off_cmd = None
        self.playing = False

    def start(self):
        if self.playing:
            return

        self.playing = True

        # set up the correct sound (program change)
        self.synth.program(self.channel, self.patch[0], self.patch[1])

        # find the tick of the next beat, and make it "beat aligned"
        now = self.sched.get_tick()
        next_beat = quantize_tick_up(now, self.beat_len)

        # now, post the _noteon function (and remember this command)
        self.on_cmd = self.sched.post_at_tick(next_beat, self._noteon)

    def stop(self):
        if not self.playing:
            return 
            
        self.playing = False

        # in case there is a note on hanging, turn it off immediately
        if self.off_cmd:
            self.off_cmd.execute()

        # cancel anything pending in the future.
        self.sched.remove(self.on_cmd)
        self.sched.remove(self.off_cmd)

        # reset these so we don't have a reference to old commands.
        self.on_cmd = None
        self.off_cmd = None

    def toggle(self):
        if self.playing:
            self.stop()
        else:
            self.start()

    def _noteon(self, tick, ignore):
        # play the note right now:
        self.synth.noteon(self.channel, self.pitch, 100)

        # post the note off for half a beat later:
        self.off_cmd = self.sched.post_at_tick(tick + self.beat_len/2, self._noteoff, self.pitch)

        # schedule the next noteon for one beat later
        next_beat = tick + self.beat_len
        self.on_cmd = self.sched.post_at_tick(next_beat, self._noteon)

    def _noteoff(self, tick, pitch):
        # just turn off the currently sounding note.
        self.synth.noteoff(self.channel, pitch)
