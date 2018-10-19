# VöXX

Final project for 21M.385 Interactive Music Systems, by Brian Chen, Dou Dou, and Xueqi Zhang.

Uses `common` code that is Copyright (c) 2017, Eran Egozy, released under the MIT License (http://opensource.org/licenses/MIT)

# Usage

You should install Python2, `kivy`, `pyaudio`, `aubio`, and maybe other things. Run `common/audio.py` and change the values in `common/config.cfg` to reflect the right input and output channels. Get the Fluid synth soundbank `FluidR3_GM.sf2` (it's a Sound Font file, you can find it online or with some Linux package managers) and put it in or symlink it into the folder `data/`. Finally:

```
python2 voxx.py
```
