#####################################################################
#
# clock.py
#
# Copyright (c) 2015, Eran Egozy
#
# Released under the MIT License (http://opensource.org/licenses/MIT)
#
#####################################################################

import time
import numpy as np
from common.audio import Audio


# Simple time keeper object. It starts at 0 and knows how to pause
class Clock(object):
    def __init__(self):
        super(Clock, self).__init__()
        self.paused = True
        self.offset = 0
        self.start()

    def is_paused(self):
        return self.paused

    def get_time(self):
        if self.paused:
            return self.offset
        else:
            return self.offset + time.time()

    def set_time(self, t):
        if self.paused:
            self.offset = t
        else:
            self.offset = t - time.time()

    def start(self):
        if self.paused:
            self.paused = False
            self.offset -= time.time()

    def stop(self):
        if not self.paused:
            self.paused = True
            self.offset += time.time()

    def toggle(self):
        if self.paused:
            self.start()
        else:
            self.stop()


# For tempo maps - converting bpm to ticks
kTicksPerQuarter = 480

class SimpleTempoMap(object):

    def __init__(self, bpm = 120) :
        super(SimpleTempoMap, self).__init__()
        self.bpm = bpm
        self.tick_offset = 0

    def time_to_tick(self, time) :
        slope = (kTicksPerQuarter * self.bpm) / 60.
        tick = slope * time + self.tick_offset
        return tick

    def tick_to_time(self, tick) :
        slope = (kTicksPerQuarter * self.bpm) / 60.
        time = (tick - self.tick_offset) / slope
        return time

    # buggy way of setting new tempo
    # def set_tempo(self, bpm, cur_time):
    #    self.bpm = bpm

    # better way of setting new tempo: maintain tick continuity
    def set_tempo(self, bpm, cur_time):
        cur_tick = self.time_to_tick(cur_time)
        self.bpm = bpm
        slope = (kTicksPerQuarter * self.bpm) / 60.
        self.tick_offset = cur_tick - cur_time * slope

    def get_tempo(self):
        return self.bpm

def tick_str(tick) :
    beat = float(tick) / kTicksPerQuarter
    return "tick:%d\nbeat:%.2f" % (tick, beat)


# data passed into tempo map is a list of points
# where each point is (time, tick)
# optionally pass in filepath instead which will
# read the file to create the list of (time, tick) points
# TempoMap will linearly interpolate this graph
class TempoMap(object):
    def __init__(self, data = None, filepath = None):
        super(TempoMap, self).__init__()

        if data == None:
            data = self._read_tempo_data(filepath)

        assert(data[0] == (0,0))
        assert(len(data) > 1)

        self.times, self.ticks = zip(*data)

    def time_to_tick(self, time) :
        tick = np.interp(time, self.times, self.ticks)
        return tick

    def tick_to_time(self, tick) :
        time = np.interp(tick, self.ticks, self.times)
        return time

    def _read_tempo_data(self, filepath):
        data = [(0,0)]

        for line in open(filepath).readlines():
            (time, beats) = line.strip().split('\t')
            time = float(time)

            delta_tick = float(beats) * kTicksPerQuarter
            last_tick = data[-1][1]
            data.append( (time, last_tick + delta_tick))

        return data


class Scheduler(object):
    def __init__(self, clock, tempo_map) :
        super(Scheduler, self).__init__()
        self.clock = clock
        self.tempo_map = tempo_map
        self.commands = []

    def get_time(self) :
        return self.clock.get_time()

    def get_tick(self) :
        sec = self.get_time()
        return self.tempo_map.time_to_tick(sec)

    # add a record for the function to call at the particular tick
    # keep the list of commands sorted from lowest to hightest tick
    # make sure tick is the first argument so sorting will work out
    # properly
    def post_at_tick(self, tick, func, arg = None) :
        now_tick = self.get_tick()

        if tick <= now_tick:
            func(tick, arg)
            return None
        else:
            cmd = Command(tick, func, arg)
            self.commands.append(cmd)
            self.commands.sort(key = lambda x: x.tick)
            return cmd

    # attempt a removal. Does nothing if cmd is not found
    def remove(self, cmd):
        if cmd in self.commands:
            idx = self.commands.index(cmd)
            del self.commands[idx]

    # on_update should be called as often as possible.
    # the only trick here is to make sure we remove the command BEFORE
    # calling the command's function.
    def on_update(self):
        now_tick = self.get_tick()
        while self.commands:
            if self.commands[0].tick <= now_tick:
                command = self.commands.pop(0)
                command.execute()
            else:
                break

    def now_str(self):
        time = self.get_time()
        tick = self.get_tick()
        beat = float(tick) / kTicksPerQuarter
        txt = "time:%.2f\ntick:%d\nbeat:%.2f" % (time, tick, beat)
        return txt


# AudioScheduler is a Scheduler and Clock built into one class.
# It is ALSO a Generator. For it to work, it must be inserted into
# and Audio generator chain.
class AudioScheduler(object):
    def __init__(self, tempo_map) :
        super(AudioScheduler, self).__init__()
        self.tempo_map = tempo_map
        self.commands = []

        self.generator = None
        self.cur_frame = 0

    def set_generator(self, gen) :
        self.generator = gen

    def generate(self, num_frames, num_channels) :
        output = np.empty(num_channels * num_frames, dtype = np.float32)
        o_idx = 0

        # the current period of time goes from self.cur_frame to end_frame
        end_frame = self.cur_frame + num_frames

        # advance time and fire off commands for this time frame
        while self.commands:
            # find the exact frame at which the next command should happen
            cmd_tick = self.commands[0].tick
            cmd_time = self.tempo_map.tick_to_time(cmd_tick)
            cmd_frame = int(cmd_time * Audio.sample_rate)

            if cmd_frame < end_frame:
                o_idx = self._generate_until(cmd_frame, num_channels, output, o_idx)
                command = self.commands.pop(0)
                command.execute()
            else:
                break

        self._generate_until(end_frame, num_channels, output, o_idx)

        return output, True

    # generate audio from self.cur_frame to to_frame
    def _generate_until(self, to_frame, num_channels, output, o_idx) :
        num_frames = to_frame - self.cur_frame
        if num_frames > 0:
            if self.generator:
                data, cont = self.generator.generate(num_frames, num_channels)
            else:
                data = np.zeros(num_channels * num_frames, dtype=np.float32)

            next_o_idx = o_idx+(num_channels * num_frames)
            output[o_idx : next_o_idx] = data
            self.cur_frame += num_frames
            return next_o_idx
        else:
            return o_idx


    def get_time(self) :
        return self.cur_frame / float(Audio.sample_rate)

    def get_tick(self) :
        return self.tempo_map.time_to_tick(self.get_time())

    # add a record for the function to call at the particular tick
    def post_at_tick(self, tick, func, arg = None) :
        now_time  = self.get_time()
        post_time = self.tempo_map.tick_to_time(tick)

        if post_time <= now_time:
            func(tick, arg)
            return None
        else:
            # create a command to hold the function/arg and sort by tick
            cmd = Command(tick, func, arg)
            self.commands.append(cmd)
            self.commands.sort(key = lambda x: x.tick)
            return cmd

    # attempt a removal. Does nothing if cmd is not found
    def remove(self, cmd):
        if cmd in self.commands:
            idx = self.commands.index(cmd)
            del self.commands[idx]

    def now_str(self):
        time = self.get_time()
        tick = self.tempo_map.time_to_tick(time)
        beat = float(tick) / kTicksPerQuarter
        txt = "time:%.2f\ntick:%d\nbeat:%.2f" % (time, tick, beat)
        return txt


class Command(object):
    def __init__(self, tick, func, arg):
        super(Command, self).__init__()
        self.tick = int(tick)
        self.func = func
        self.arg = arg
        self.did_it = False

    def execute(self):
        # ensure that execute only gets called once.
        if not self.did_it:
            self.did_it = True
            self.func( self.tick, self.arg )

    def __repr__(self):
        return 'cmd:%d' % self.tick

# helper function for quantization:
def quantize_tick_up(tick, grid) :
    return tick - (tick % grid) + grid

