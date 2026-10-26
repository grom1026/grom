# Гром (Grom, "thunder")

A spoken debug loop for blind coders. Python on Windows, with VS Code.

When a script crashes, Grom says the error out loud, starting with what matters: the error and the line number. A small neural net then explains the error in plain words. You step through the rest with keys, and one key opens VS Code right at the broken line.

If NVDA, JAWS or another screen reader is running, Grom talks through it, so you only ever hear one voice.

Free, for anyone.

---

## The files

| File | What it is |
|---|---|
| Гром_GROM.py | The whole program |
| grom_explain_net.json | The explaining net's trained weights |
| README.md | This |

Keep all three in the same folder.

---

## Turn Grom on (once)

In the Grom folder, in the command window:

```
python Гром_GROM.py
```

Listen for "Grom is on." From then on it starts by itself when you log in.

This also installs what Grom needs (the voice and the screen reader link), and hooks Grom into Python so every crash gets spoken.

---

## Keys

These work from any program while Grom is on.

| Keys | What happens |
|---|---|
| Ctrl+Shift+Alt+N | Next piece of the last crash |
| Ctrl+Shift+Alt+B | Back one piece |
| Ctrl+Shift+Alt+R | Repeat this piece |
| Ctrl+Shift+Alt+J | Open VS Code at the broken line |

NVDA and JAWS don't use any Ctrl+Shift+Alt keys, so these don't clash with your screen reader. If another program already uses one of them, Grom tells you when it starts.

## What you hear after a crash

1. The error and the line number. This plays by itself.
2. What the error means, in plain words. This is the explaining net.
3. The broken line, read out.
4. The rest of the trail, one step at a time.

If the net doesn't recognize the error, it says it isn't sure, instead of guessing.

---

## Other commands

**Test it.** This crashes Python on purpose:

```
python -c "print(frames)"
```

You should hear the error. Ctrl+Shift+Alt+N then gives the explanation.

**Turn Grom off**, and stop it starting at login:

```
python Гром_GROM.py off
```

**Retrain the explaining net** (needs numpy, `pip install numpy`):

```
python Гром_GROM.py train
```

It runs thousands of small broken scripts, learns from the real errors, tests itself on errors it never saw, and saves a new grom_explain_net.json.

**If Ctrl+Shift+Alt+J doesn't open VS Code**, check this works:

```
code --version
```

If it says "not recognized," open VS Code, press Ctrl+Shift+P, and run **Shell Command: Install 'code' command in PATH**, or reinstall VS Code with "Add to PATH" checked.

---

## Better voice (optional)

With no screen reader running, Grom uses the Windows voice. For the Piper voice instead:

```
pip install piper-tts
```

Then download `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json` and put them in:

```
%APPDATA%\SpokenDebug\voices
```

Grom switches to it by itself.

---

## Known issues

- Only tested so far by a sighted user. Feedback from blind coders is very welcome.

## Coming

- A second net that notices when a run looks wrong even without a crash, and says how sure it is.
