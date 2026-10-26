# Гром_GROM.py - Гром (GROM, "thunder"): a spoken debug loop for blind coders.
#
# This one file is the whole program:
#   - the crash catcher: when any Python script crashes, it says the error out loud
#   - the explaining net: says what the error means, in plain words
#   - the run watcher: when a script finishes without crashing but looks off
#     (NaN, a loss going the wrong way, stuck loops, odd run times, warnings),
#     a second net says what looked off and how sure it is
#   - the key listener: steps through the crash, jumps to the broken line
#   - setup and retraining
#
# Commands (run in this folder):
#   python Гром_GROM.py           set up and turn Grom on (run once)
#   python Гром_GROM.py train     retrain both nets (needs numpy)
#   python Гром_GROM.py off       turn Grom off and stop it starting at login
#
# Keys (work from any program while Grom is on):
#   Ctrl+Shift+Alt+N  next piece of the last crash
#   Ctrl+Shift+Alt+B  back one piece
#   Ctrl+Shift+Alt+R  repeat this piece
#   Ctrl+Shift+Alt+J  jump to the broken line in VS Code
#
# If NVDA, JAWS or another screen reader is running, Grom talks through it.

import os
import io
import re
import sys
import json
import math
import time
import zlib
import wave
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(HERE, 'grom_explain_net.json')
DATA_DIR = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'SpokenDebug')
CRASH_FILE = os.path.join(DATA_DIR, 'last_crash.json')


# =================================================================
# VOICE - uses the Piper voice if it's downloaded, the Windows voice if not
# =================================================================

VOICE = 'en_US-lessac-medium'
VOICE_MODEL = os.path.join(DATA_DIR, 'voices', VOICE + '.onnx')

_piper = None
_piper_failed = False
_windows_voice = None


def _say_piper(text):
    global _piper, _piper_failed
    if _piper_failed or not os.path.exists(VOICE_MODEL):
        return False
    try:
        if _piper is None:
            from piper import PiperVoice
            _piper = PiperVoice.load(VOICE_MODEL)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            _piper.synthesize_wav(text, wav)
        import winsound
        winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)
        return True
    except Exception:
        _piper_failed = True
        return False


def _say_windows(text):
    global _windows_voice
    import pyttsx3
    if _windows_voice is None:
        _windows_voice = pyttsx3.init()
    _windows_voice.say(text)
    _windows_voice.runAndWait()


_readers = None


def _screen_reader():
    # the person's own screen reader (NVDA, JAWS and others), if one is running
    global _readers
    if _readers is False:
        return None          # the screen reader library isn't installed
    if _readers is None:
        try:
            from accessible_output2.outputs.auto import Auto
            _readers = Auto()
        except Exception:
            _readers = False
            return None
    try:
        found = _readers.get_first_available_output()
        if found and not found.is_system_output():
            return found
    except Exception:
        pass
    return None


def say(text):
    # 1. a screen reader is running: talk through it, so there's only one voice.
    #    New speech cuts off old speech, the way screen reader users expect.
    reader = _screen_reader()
    if reader:
        try:
            reader.speak(text, interrupt=True)
            return
        except Exception:
            pass
    # 2. no screen reader: Grom's own voice
    if not _say_piper(text):
        _say_windows(text)


def using_screen_reader():
    return _screen_reader() is not None


# =================================================================
# EXPLAINING NET - reads the error and picks the plain-words explanation
# =================================================================

SIZE = 2048          # how many input slots the net has
SURE_ENOUGH = 0.5    # below this, Grom says it isn't sure

# ---------------------------------------------------------------
# what the net reads: the error kind and message, chopped into words
# ---------------------------------------------------------------

def _words(kind, message):
    text = message.lower()
    text = re.sub(r"\(.*?, line \d+\)$", ' ', text)   # "(script.py, line 4)"
    text = re.sub(r"\([^()]*[\\/][^()]*\)", ' ', text)  # a file path in brackets
    text = re.sub(r'\d+', ' # ', text)                 # numbers all look alike
    words = re.findall(r"[a-z_]+|[^\sa-z_]", text)
    out = ['KIND_' + kind]
    out += words
    out += [a + '|' + b for a, b in zip(words, words[1:])]   # word pairs
    return out


def features(kind, message):
    slots = {}
    for w in _words(kind, message):
        i = zlib.crc32(w.encode('utf-8')) % SIZE
        slots[i] = slots.get(i, 0.0) + 1.0
    scale = 1.0 / math.sqrt(sum(v * v for v in slots.values()) or 1.0)
    return {i: v * scale for i, v in slots.items()}


# ---------------------------------------------------------------
# the plain-words explanations. {a} and {b} are filled with the
# first and second quoted thing in the error message.
# ---------------------------------------------------------------

EXPLAIN = {
    'name_not_defined': "{a} is used before it was made. Check the spelling, or make {a} earlier in the script.",
    'unbound_local': "{a} gets a value inside this function, but it's used before that happens. Give it a value first. If you meant the {a} outside the function, add a global line for it.",
    'none_attribute': "Something here is None, meaning empty, so it has no {b}. Usually a function gave nothing back, or a lookup found nothing.",
    'attribute_missing': "A {a} doesn't have anything called {b}. Check the spelling, or this thing isn't the kind you think it is.",
    'module_attribute': "The module {a} has nothing called {b}. Check the spelling. Also check you don't have your own file named {a}, which would hide the real one.",
    'module_not_found': "Python can't find a module called {a}. Either it isn't installed, or the name is spelled wrong. Installing is usually pip install, then the name.",
    'import_name': "{b} has nothing called {a} to import. Check the spelling, or your version of {b} may not have it.",
    'key_error': "The dictionary has no key {a}. Check the spelling, or that key was never added.",
    'index_error': "You asked for a spot past the end. Counting starts at zero, so the last spot is one less than the length.",
    'pop_empty': "You took an item out of something that was already empty.",
    'zero_div': "Something got divided by zero. Check the number after the divide sign.",
    'str_concat': "You joined text and something that isn't text with a plus sign. Wrap the other thing in str() first, or use an f-string.",
    'unsupported_operand': "That math sign doesn't work between a {a} and a {b}. Convert one so they match.",
    'not_callable': "There are round brackets after a {a}, so Python tried to run it like a function. Often a variable has the same name as a function.",
    'not_subscriptable': "There are square brackets after a {a}, and a {a} can't be looked into that way.",
    'not_iterable': "Something tried to go through a {a} one item at a time, like a loop, but a {a} isn't a list or anything like one.",
    'missing_arg': "The function was called without everything it needs. It's missing {a}.",
    'too_many_args': "The function got more things than it takes. If it's inside a class, check that self is first in its def line.",
    'unexpected_kwarg': "The function has no setting called {a}. Check the spelling.",
    'not_a_number': "You tried to turn text into a number, but the text {a} isn't a number.",
    'unpack_count': "The number of names on the left of the equals doesn't match the number of things on the right.",
    'not_in_list': "You looked for something in a list, and it isn't there.",
    'file_not_found': "There's no file at {a}. Check the name and the folder. Python looks in the folder you ran the script from.",
    'permission': "Windows won't let Python use {a}. It may be open in another program, or it's a folder, not a file.",
    'missing_colon': "A colon is missing at the end of the line, after an if, for, while, def or class.",
    'unclosed_bracket': "A bracket was opened and never closed. Count the opening and closing brackets on that line.",
    'unterminated_string': "A piece of text was started with a quote mark and never closed.",
    'syntax_general': "Python can't read this line. Look for a missing comma, bracket, quote or colon, or a typo.",
    'indentation': "The spaces at the start of a line are off. Lines in the same block need the same indent, and the line after a colon needs to move in.",
    'tabs_spaces': "This file mixes tabs and spaces at the start of lines. Use only spaces.",
    'recursion': "A function keeps calling itself and never stops. Check the part that's supposed to end it.",
    'assertion': "An assert line checked something and found it false.",
    'decode': "The file has characters Python couldn't read with the default setting. Open it with encoding equals utf-8.",
    'changed_size': "Something was added to or removed from a dictionary or set while it was being looped over. Loop over a copy instead.",
    'json_bad': "The text isn't valid JSON. Often the file is empty or cut off.",
    'stop_iteration': "next() asked for another item, but there were none left.",
    'unhashable': "A {a} can't be a dictionary key or go in a set. Use a tuple instead of a list.",
    'math_domain': "A math function got a number it can't take, like the square root of a negative.",
    'overflow': "A number got too big for Python to handle.",
    'item_assign': "A {a} can't be changed one piece at a time. Make a new one instead.",
    'compare': "Two things got compared that can't be compared, like text and a number.",
    'bad_convert_type': "A conversion like int() or float() got a {a}, and it can't turn that into a number.",
}

UNSURE = "Grom isn't sure what this one means. Hear the full error for details."


def _quoted(message):
    found = [a or b for a, b in re.findall(r"'([^']*)'|\"([^\"]*)\"", message)]
    if not found:
        found = re.findall(r"non-iterable (\w+)", message)
    return found


# ---------------------------------------------------------------
# the net itself
# ---------------------------------------------------------------

_net = None


def _load():
    global _net
    if _net is None:
        with open(NET_FILE, encoding='utf-8') as fh:
            _net = json.load(fh)
    return _net


def guess(kind, message):
    """Returns (label, how_sure) - how_sure is 0 to 1."""
    net = _load()
    x = features(kind, message)
    W1, b1, W2, b2 = net['W1'], net['b1'], net['W2'], net['b2']
    hidden = list(b1)
    for i, v in x.items():
        row = W1[i]
        for j in range(len(hidden)):
            hidden[j] += row[j] * v
    hidden = [h if h > 0 else 0.0 for h in hidden]
    scores = list(b2)
    for j, h in enumerate(hidden):
        if h:
            row = W2[j]
            for k in range(len(scores)):
                scores[k] += row[k] * h
    top = max(scores)
    ex = [math.exp(s - top) for s in scores]
    total = sum(ex)
    k = max(range(len(ex)), key=ex.__getitem__)
    return net['labels'][k], ex[k] / total


def explain(kind, message):
    """The plain-words explanation for one error."""
    label, sure = guess(kind, message)
    if sure < SURE_ENOUGH:
        return UNSURE
    q = _quoted(message)
    if label == 'key_error':
        q = [message.strip().strip('\'"')]
    elif label == 'missing_arg' and ': ' in message:
        q = [message.split(': ', 1)[1].replace("'", '')]
    try:
        return EXPLAIN[label].format(a=q[0] if q else 'it',
                                     b=q[1] if len(q) > 1 else 'that')
    except Exception:
        return UNSURE


# =================================================================
# CRASH CATCHER - usercustomize.py calls catch_crashes() every time Python starts
# =================================================================

_original_hook = sys.excepthook


def _spoken_kind(name):
    # NameError -> "name error"
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', name).lower()


def _is_real_file(filename):
    return not filename.startswith('<')


def _is_yours(filename):
    # True if the line is in YOUR code, not Python's or a library's
    if filename.startswith('<frozen'):
        return False
    name = os.path.normcase(filename)
    skip = [os.path.normcase(p) for p in (sys.prefix, sys.base_prefix)]
    skip.append('site-packages')
    return not any(s in name for s in skip)


def _where(f):
    if _is_real_file(f.filename):
        return 'line ' + str(f.lineno) + ', in ' + os.path.basename(f.filename)
    return 'line ' + str(f.lineno)


def _group(exc, first):
    # the pieces for one error, bottom part first
    frames = traceback.extract_tb(exc.__traceback__)
    yours = [f for f in frames if _is_yours(f.filename)]
    main = yours[-1] if yours else (frames[-1] if frames else None)

    head = _spoken_kind(type(exc).__name__) + '. ' + str(exc)
    if main:
        head += '. ' + _where(main)
    if not first:
        head = 'Before that, another error. ' + head
    pieces = [head + '.']

    # the explaining net: what the error means, in plain words
    try:
        pieces.append(explain(type(exc).__name__, str(exc)))
    except Exception:
        pass  # no net file, or it broke - Grom still works without it

    if main and main.line:
        pieces.append('The line says: ' + main.line)

    # the rest of the trail, going upward
    for f in reversed(frames):
        if f is main or f.filename.startswith('<frozen'):
            continue
        label = 'Called from ' if _is_yours(f.filename) else 'Inside a library, '
        text = label + _where(f)
        if f.line:
            text += ': ' + f.line
        pieces.append(text)

    jump = None
    if main and _is_real_file(main.filename):
        jump = {'file': os.path.abspath(main.filename), 'line': main.lineno}

    # a typo error points at its own line, not the trail
    if isinstance(exc, SyntaxError) and exc.filename and _is_real_file(exc.filename):
        jump = {'file': os.path.abspath(exc.filename), 'line': exc.lineno or 1}
        pieces[0] = '{}{}. {}. line {}, in {}.'.format(
            '' if first else 'Before that, another error. ', _spoken_kind(type(exc).__name__),
            exc.msg, jump['line'], os.path.basename(exc.filename))
        pieces = [p for p in pieces if not p.startswith('The line says: ')]
        if exc.text and exc.text.strip():
            pieces.insert(2 if len(pieces) > 1 else 1, 'The line says: ' + exc.text.strip())
    return pieces, jump


def build_pieces(exc):
    # handles stacked errors: newest first, each one its own group
    pieces, jump = _group(exc, first=True)
    seen = {id(exc)}
    cur = exc
    while True:
        nxt = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        more, _ = _group(nxt, first=False)
        pieces += more
        cur = nxt
    return pieces, jump


def _speak_crash(exc_type, exc, tb):
    global _crashed
    _original_hook(exc_type, exc, tb)  # normal error printout first
    if issubclass(exc_type, KeyboardInterrupt):
        return   # stopped by hand: the run watcher still checks it (handy for stuck loops)
    _crashed = True
    try:
        if exc.__traceback__ is None:
            exc.__traceback__ = tb
        pieces, jump = build_pieces(exc)
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CRASH_FILE, 'w', encoding='utf-8') as fh:
            json.dump({'time': time.time(), 'pieces': pieces, 'jump': jump}, fh)
        # if Grom is on, it says the error itself, so there's only ever one voice
        if not _grom_is_on():
            say(pieces[0])
    except Exception:
        pass  # never let the talking part cause its own crash


def _grom_is_on():
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenMutexW(0x00100000, False, 'SpokenDebugLoop')
        if handle:
            kernel32.CloseHandle(handle)
            return True
    except Exception:
        pass
    return False


def catch_crashes():
    sys.excepthook = _speak_crash
    watch_runs()


# =================================================================
# RUN WATCHER - the second net. When one of your scripts finishes
# WITHOUT crashing, Grom looks at what it printed and how long it took,
# and says if something looks off, and how sure it is.
# =================================================================

RUN_NET_FILE = os.path.join(HERE, 'grom_run_net.json')
HISTORY_FILE = os.path.join(DATA_DIR, 'run_history.json')
SPEAK_ABOVE = 0.6      # the net has to be at least this sure before Grom says anything

# each sign gets its own small net that only sees the evidence that matters for it,
# so one sign can't drag another off (a timing quirk can't hide a blown-up loss)
SIGN_GROUPS = [
    (['nan'], list(range(0, 3))),                              # numbers gone bad
    (['loss_up', 'loss_flat'], list(range(2, 13))),            # the loss
    (['no_output'], list(range(13, 17))),                      # how much it printed
    (['repeating'], [13] + list(range(17, 21))),               # repeats
    (['too_slow', 'too_fast'], list(range(21, 26))),           # time
    (['warnings'], list(range(26, 29))),                       # warnings
]

_run = None            # the run being watched right now
_crashed = False       # a crash already got reported, so the watcher stays quiet

SIGNS = ['nan', 'loss_up', 'loss_flat', 'no_output', 'repeating',
         'too_slow', 'too_fast', 'warnings']

NUMBER_GONE_BAD = re.compile(r'(?i)(?<![a-z])(nan|-?inf(?:inity)?)(?![a-z])')
LOSS = re.compile(r"(?i)(?<![a-z])((?:train|training|val|valid|validation|test|eval)?[_ ]?"
                  r"(?:loss|cost|mse|nll))['\"]?\s*[:=]?\s*"
                  r"(-?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?|nan|-?inf(?:inity)?)(?![\w.])")
SOFT_WARNINGS = ('DeprecationWarning', 'PendingDeprecationWarning', 'FutureWarning',
                 'ImportWarning', 'ResourceWarning')


class _RunRecord:
    # keeps the first 200 and the last 3000 lines a script prints
    def __init__(self, script):
        import collections
        self.script = script
        self.start = time.time()
        self.first = []
        self.last = collections.deque(maxlen=3000)
        self.count = 0
        self.warnings = []     # (kind, file, line, message)
        self._part = ''

    def feed(self, text):
        if '\n' not in text:
            self._part += text
            if len(self._part) > 20000:
                self._part = self._part[-10000:]
            return
        text = self._part + text
        parts = text.split('\n')
        self._part = parts.pop()[-10000:]
        for p in parts:
            self._add(p)

    def _add(self, line):
        line = line.split('\r')[-1].rstrip()   # progress bars redraw with \r: keep the last drawing
        self.count += 1
        if len(self.first) < 200:
            self.first.append(line)
        else:
            self.last.append(line)

    def finish(self):
        if self._part:
            self._add(self._part)
            self._part = ''
        return self.first + list(self.last)


class _Tee:
    # passes everything through to the real output, and keeps a copy for the watcher
    def __init__(self, stream, record):
        self._stream = stream
        self._record = record

    def write(self, text):
        try:
            self._record.feed(text)
        except Exception:
            pass
        return self._stream.write(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def watch_runs():
    global _run
    try:
        script = sys.argv[0] if sys.argv else ''
        # only scripts run from a file: not "python -c", not "python -m", not the >>> prompt
        if not script.endswith(('.py', '.pyw')) or not os.path.isfile(script):
            return
        script = os.path.abspath(script)
        if os.path.basename(script) == os.path.basename(os.path.abspath(__file__)):
            return                      # Grom itself
        if not _is_yours(script) or '.vscode' in script.lower():
            return                      # a library's or an editor's script, not yours
        _run = _RunRecord(script)
        if sys.stdout is not None:
            sys.stdout = _Tee(sys.stdout, _run)
        if sys.stderr is not None:
            sys.stderr = _Tee(sys.stderr, _run)

        import warnings
        real_show = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            try:
                _run.warnings.append((category.__name__, filename, lineno, str(message)))
            except Exception:
                pass
            return real_show(message, category, filename, lineno, file, line)
        warnings.showwarning = show

        import atexit
        atexit.register(_check_run)
    except Exception:
        _run = None


def _load_history():
    try:
        with open(HISTORY_FILE, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _check_run():
    global _run
    record, _run = _run, None
    if record is None or _crashed:
        return
    try:
        lines = record.finish()
        seconds = time.time() - record.start
        history = _load_history()
        past = history.get(record.script, [])
        report = check_run(lines, record.count, seconds, past, record.warnings, record.script)

        # remember this run, so next time "usual" means something
        past.append({'t': round(seconds, 3), 'lines': record.count})
        history[record.script] = past[-20:]
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as fh:
            json.dump(history, fh)

        if report:
            pieces, jump = report
            with open(CRASH_FILE, 'w', encoding='utf-8') as fh:
                json.dump({'time': time.time(), 'pieces': pieces, 'jump': jump}, fh)
            if not _grom_is_on():
                say(pieces[0])
    except Exception:
        pass  # the watcher must never break anyone's script


# ---------------------------------------------------------------
# what the second net reads: numbers that describe the run
# ---------------------------------------------------------------

def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return 0.0 if not n else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def _loss_series(lines):
    found = {}
    for line in lines:
        for key, value in LOSS.findall(line):
            key = key.lower().replace(' ', '_').strip('_')
            found.setdefault(key, []).append(value)
    if not found:
        return None, []
    # the most-printed loss; training loss wins a tie over validation/test
    key = max(found, key=lambda k: (len(found[k]), not k.startswith(('val', 'test', 'eval'))))
    values = []
    for v in found[key]:
        try:
            values.append(float(v))
        except ValueError:
            values.append(float('nan'))
    return key, values


def run_features(lines, count, seconds, past, warns):
    """Turns one run into a list of numbers for the net, plus the facts Grom speaks."""
    clip = lambda x, lo=-5.0, hi=5.0: max(lo, min(hi, x))  # noqa: E731
    facts = {}

    # 1. numbers gone bad
    bad_lines = [ln for ln in lines if NUMBER_GONE_BAD.search(ln)]
    facts['bad_count'] = len(bad_lines)
    facts['bad_line'] = bad_lines[0] if bad_lines else ''
    f = [math.log1p(len(bad_lines)), len(bad_lines) / max(1, len(lines))]

    # 2. the loss
    key, values = _loss_series(lines)
    finite = [v for v in values if math.isfinite(v)]
    facts['loss_key'] = key
    f.append(math.log1p(len(values) - len(finite)))
    if len(finite) >= 4:
        n = len(finite)
        k = max(1, n // 10)
        raw_first, raw_last = sum(finite[:k]) / k, sum(finite[-k:]) / k
        scale = sum(abs(v) for v in finite) / n + 1e-8
        # the shape of the loss is measured on a squashed scale (like a log chart),
        # so a loss that explodes to huge numbers still reads as a clear climb
        mid = _median([abs(v) for v in finite]) + 1e-12
        sq = [math.asinh(v / mid) for v in finite]
        first, last = sum(sq[:k]) / k, sum(sq[-k:]) / k
        xs = [i / (n - 1) for i in range(n)]
        mx, my = sum(xs) / n, sum(sq) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, sq)) / (sum((x - mx) ** 2 for x in xs) or 1)
        lo, hi = min(sq), max(sq)
        ups = sum(1 for a, b in zip(sq, sq[1:]) if b > a) / (n - 1)
        third = sq[-max(2, n // 3):]
        resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, sq)]
        noise = (sum(r * r for r in resid) / n) ** 0.5
        f += [clip(slope / (noise + 1e-12) / 4), clip((last - first) / (hi - lo + 1e-12), -1, 1)]
        f += [1.0, math.log1p(n) / 5,
              clip(slope), clip((raw_last - raw_first) / (abs(raw_first) + 1e-8)),
              ups, clip((max(finite) - min(finite)) / scale, 0, 10),
              (last - lo) / (hi - lo + 1e-12),
              clip(third[-1] - third[0])]
        first, last, my = finite[0], finite[-1], sum(finite) / n   # what Grom reads out
        facts.update(loss_first=first, loss_last=last, loss_mean=my)
    else:
        f += [0.0] * 10

    # 3. output: how much, compared with usual, and repeats
    past_lines = [p.get('lines', 0) for p in past]
    usual_lines = _median(past_lines) if past_lines else 0
    facts['usual_lines'] = usual_lines
    f += [math.log1p(count) / 8, 1.0 if past_lines else 0.0,
          clip(math.log((count + 1) / (usual_lines + 1))) if past_lines else 0.0,
          1.0 if count == 0 else 0.0]

    streak, best, best_line, prev = 0, 0, '', None
    for ln in lines:
        if ln.strip() and ln == prev:
            streak += 1
        else:
            streak = 1 if ln.strip() else 0
        if streak > best:
            best, best_line = streak, ln
        prev = ln
    tail = [ln for ln in lines[-200:] if ln.strip()]
    same_next = sum(1 for a, b in zip(tail, tail[1:]) if a == b) / max(1, len(tail) - 1)
    facts['streak'], facts['streak_line'] = best, best_line
    f += [math.log1p(best) / 8, same_next, len(set(tail)) / max(1, len(tail)),
          math.log1p(streak) / 8]

    # 4. time, compared with usual
    past_t = [p.get('t', 0.0) for p in past if p.get('t', 0) > 0]
    usual_t = _median(past_t) if past_t else 0.0
    facts['usual_t'], facts['seconds'] = usual_t, seconds
    if len(past_t) >= 1:
        logs = [math.log(t + 0.05) for t in past_t]
        spread = (sum((x - sum(logs) / len(logs)) ** 2 for x in logs) / len(logs)) ** 0.5
        f += [min(len(past_t), 10) / 10, clip(math.log((seconds + 0.05) / (usual_t + 0.05))),
              min(spread, 3.0), math.log1p(abs(seconds - usual_t)) / 5, math.log1p(seconds) / 5]
    else:
        f += [0.0, 0.0, 0.0, 0.0, math.log1p(seconds) / 5]

    # 5. warnings
    hard = [w for w in warns if w[0] not in SOFT_WARNINGS]
    facts['warnings'] = hard
    f += [math.log1p(len(hard)), math.log1p(len(warns) - len(hard)),
          math.log1p(len(set((w[0], w[3]) for w in hard)))]
    return f, facts


# ---------------------------------------------------------------
# the second net, and what Grom says
# ---------------------------------------------------------------

_run_net = None


def _load_run_net():
    global _run_net
    if _run_net is None:
        with open(RUN_NET_FILE, encoding='utf-8') as fh:
            _run_net = json.load(fh)
    return _run_net


def run_guess(features):
    """How sure the nets are about each sign, 0 to 1."""
    out = {}
    for part in _load_run_net()['groups']:
        x = [(features[i] - m) / sd for i, m, sd in zip(part['features'], part['mean'], part['std'])]
        hidden = list(part['b1'])
        for i, v in enumerate(x):
            row = part['W1'][i]
            for j in range(len(hidden)):
                hidden[j] += row[j] * v
        hidden = [h if h > 0 else 0.0 for h in hidden]
        for k, label in enumerate(part['labels']):
            z = part['b2'][k] + sum(part['W2'][j][k] * h for j, h in enumerate(hidden))
            out[label] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
    return out


def _sure_words(p):
    return 'Very sure.' if p >= 0.9 else ('Fairly sure.' if p >= 0.75 else 'Not very sure.')


def _num(x):
    if abs(x) >= 1e6 or (0 < abs(x) < 1e-3):
        power = int(math.floor(math.log10(abs(x))))
        return 'about {:.2g} times 10 to the {}'.format(x / 10 ** power, power)
    if abs(x) >= 100 or x == int(x):
        return '{:,.0f}'.format(x)
    return '{:.3g}'.format(x)


def _time_words(s):
    if s < 1:
        return 'under a second'
    if s < 90:
        return '{:.0f} seconds'.format(s)
    if s < 5400:
        return '{:.0f} minutes'.format(s / 60)
    return '{:.1f} hours'.format(s / 3600)


def _usually(s):
    return 'under a second' if s < 1 else 'about ' + _time_words(s)


def _short(line):
    line = line.strip()
    return line if len(line) <= 100 else line[:100] + '...'


def _say_sign(sign, facts):
    if sign == 'nan':
        return ('numbers went bad. NaN or infinity showed up {} times'.format(facts['bad_count']),
                'First place it showed up: ' + _short(facts['bad_line']))
    if sign == 'loss_up':
        return ('the {} went up instead of down, from {} to {}'.format(
                    facts['loss_key'].replace('_', ' '), _num(facts['loss_first']), _num(facts['loss_last'])),
                'A loss going up usually means the learning rate is too high, or the data and labels are mixed up.')
    if sign == 'loss_flat':
        return ('the {} stayed flat, around {}, so it may not be learning'.format(
                    facts['loss_key'].replace('_', ' '), _num(facts['loss_mean'])),
                'A flat loss usually means the learning rate is too low, or the weights are not being updated.')
    if sign == 'no_output':
        return ('it printed nothing, and it usually prints about {} lines'.format(_num(facts['usual_lines'])),
                'It may have skipped the part that does the work, or quit early.')
    if sign == 'repeating':
        return ('the same line repeated {} times in a row'.format(facts['streak']),
                'The line was: ' + _short(facts['streak_line']))
    if sign == 'too_slow':
        return ('it took {}, and it usually takes {}'.format(
                    _time_words(facts['seconds']), _usually(facts['usual_t'])),
                'Something may be stuck, or working on far more than usual.')
    if sign == 'too_fast':
        return ('it finished in {}, and it usually takes {}'.format(
                    _time_words(facts['seconds']), _usually(facts['usual_t'])),
                'It may have stopped early, or skipped the main work.')
    if sign == 'warnings':
        w = facts['warnings']
        return ('Python gave {} warning{}'.format(len(w), '' if len(w) == 1 else 's'),
                'The first: {}. {}. Line {}, in {}.'.format(
                    _spoken_kind(w[0][0]), _short(w[0][3]), w[0][2], os.path.basename(w[0][1])))
    return None


def check_run(lines, count, seconds, past, warns, script):
    """Returns (pieces, jump) if the run looks off, or None if it looks fine."""
    features, facts = run_features(lines, count, seconds, past, warns)
    sure = run_guess(features)
    found = sorted((p, s) for s, p in sure.items() if p >= SPEAK_ABOVE)[::-1]
    if not found:
        return None
    pieces = []
    for n, (p, sign) in enumerate(found):
        said = _say_sign(sign, facts)
        if not said:
            continue
        head, detail = said
        if n == 0:
            pieces.append('The run of {} finished, but {}. {}'.format(
                os.path.basename(script), head, _sure_words(p)))
        else:
            pieces.append('Also, {}. {}'.format(head, _sure_words(p)))
        pieces.append(detail)
    jump = {'file': script, 'line': 1}
    if facts['warnings'] and _is_real_file(facts['warnings'][0][1]):
        jump = {'file': os.path.abspath(facts['warnings'][0][1]), 'line': facts['warnings'][0][2]}
    return (pieces, jump) if pieces else None


# =================================================================
# KEY LISTENER - runs in the background, answers the keys
# =================================================================

state = {'time': None, 'pieces': [], 'jump': None, 'i': 0}


def _tell(text):
    print('>>', text, flush=True)
    say(text)


def _refresh():
    # load the newest crash; a new crash starts back at piece one
    try:
        with open(CRASH_FILE, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    if data.get('time') != state['time']:
        state.update(time=data.get('time'), pieces=data.get('pieces', []),
                     jump=data.get('jump'), i=0)
    return bool(state['pieces'])


def _crash_stamp():
    try:
        return os.path.getmtime(CRASH_FILE)
    except OSError:
        return None


def next_piece():
    if not _refresh():
        return _tell('No crash yet.')
    if state['i'] + 1 < len(state['pieces']):
        state['i'] += 1
        _tell(state['pieces'][state['i']])
    else:
        _tell('End.')


def back_piece():
    if not _refresh():
        return _tell('No crash yet.')
    if state['i'] > 0:
        state['i'] -= 1
        _tell(state['pieces'][state['i']])
    else:
        _tell('Start. ' + state['pieces'][0])


def repeat_piece():
    if not _refresh():
        return _tell('No crash yet.')
    _tell(state['pieces'][state['i']])


def jump_to_line():
    import subprocess
    if not _refresh():
        return _tell('No crash yet.')
    j = state['jump']
    if not j:
        return _tell('No file to jump to.')
    subprocess.Popen('code -g "{}:{}"'.format(j['file'], j['line']), shell=True)
    _tell('Line ' + str(j['line']) + '.')


# Keys are Ctrl+Shift+Alt plus a letter. NVDA and JAWS don't use Ctrl+Shift+Alt,
# and these letters don't clash with VS Code.
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000
WM_HOTKEY = 0x0312
PM_REMOVE = 1
KEYS = {
    ord('N'): ('N', next_piece),     # next
    ord('B'): ('B', back_piece),     # back
    ord('R'): ('R', repeat_piece),   # repeat
    ord('J'): ('J', jump_to_line),   # jump
}


def listen():
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    # only one copy at a time
    kernel32.CreateMutexW(None, False, 'SpokenDebugLoop')
    if kernel32.GetLastError() == 183:
        say('Grom is already running.')
        return

    # claim the keys from Windows, so they reach Grom and not the program you're in
    taken = []
    for vk, (letter, _) in KEYS.items():
        if not user32.RegisterHotKey(None, vk, MOD_CONTROL | MOD_SHIFT | MOD_ALT | MOD_NOREPEAT, vk):
            taken.append(letter)

    msg = wintypes.MSG()

    def drain():
        # Grom's own voice makes you wait while it talks; key presses made
        # during that are thrown away, so no piece gets skipped.
        # (A screen reader doesn't make you wait, so nothing is thrown away then.)
        if not using_screen_reader():
            while user32.PeekMessageW(ctypes.byref(msg), None, WM_HOTKEY, WM_HOTKEY, PM_REMOVE):
                pass

    _refresh()   # know about the last crash, but don't say it
    last_seen = _crash_stamp()   # noted before talking, so a crash during it still gets said
    say('Grom is on.')
    if taken:
        say("Another program already uses Control Shift Alt " + ", ".join(taken) +
            ". Those keys won't work until it lets go of them.")
    while True:
        # a new crash: say its first piece
        stamp = _crash_stamp()
        if stamp != last_seen:
            last_seen = stamp
            time.sleep(0.1)   # let the crash finish saving
            old_time = state['time']
            if _refresh() and state['time'] != old_time:
                _tell(state['pieces'][0])
                drain()
            continue
        # a key press
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            if msg.message == WM_HOTKEY and msg.wParam in KEYS:
                try:
                    KEYS[msg.wParam][1]()
                except Exception as e:
                    print('problem:', e, flush=True)
                drain()
        time.sleep(0.03)


# =================================================================
# SETUP - hooks Grom into Python, starts it, and sets it to start at login
# =================================================================

HOOK_MARK = '# made by Гром_GROM.py'
HOOK = HOOK_MARK + ''' - hooks Grom into Python. Delete this file to unhook it.
import sys
sys.path.append({folder!r})
try:
    import Гром_GROM
    Гром_GROM.catch_crashes()
except Exception:
    pass
'''
RUN_KEY = r'Software\Microsoft\Windows\CurrentVersion\Run'


def _stop_old_grom():
    # stops any Grom already running in the background (old or new)
    import subprocess
    # the listener always runs under pythonw.exe, so only those are stopped
    ps = ("Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq 'pythonw.exe' -and "
          "$_.CommandLine -match 'GROM' -and $_.ProcessId -ne {} }} | "
          "ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force }}").format(os.getpid())
    subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                   capture_output=True, creationflags=0x08000000)
    time.sleep(1)


def _pythonw():
    p = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    return p if os.path.exists(p) else sys.executable


def setup():
    import site
    import subprocess
    import winreg

    print('Setting up Гром...')

    # 1. the voice, and the link to the person's screen reader
    for module, package, what in (('pyttsx3', 'pyttsx3', 'the Windows voice'),
                                  ('accessible_output2', 'accessible_output2',
                                   'the screen reader link')):
        try:
            __import__(module)
            print('  already have', what)
        except ImportError:
            print('  installing', what, '(' + package + ')...')
            subprocess.run([sys.executable, '-m', 'pip', 'install', package])

    # 2. hook Grom into Python
    user_site = site.getusersitepackages()
    # if an old Grom catcher is already loading from somewhere, replace that one
    loaded = getattr(sys.modules.get('usercustomize'), '__file__', None)
    if loaded and loaded.endswith('.py') and os.path.exists(loaded):
        with open(loaded, encoding='utf-8', errors='replace') as fh:
            text = fh.read()
        if 'SpokenDebug' in text or HOOK_MARK in text:
            user_site = os.path.dirname(loaded)
    os.makedirs(user_site, exist_ok=True)
    hook_file = os.path.join(user_site, 'usercustomize.py')
    if os.path.exists(hook_file):
        with open(hook_file, encoding='utf-8', errors='replace') as fh:
            old = fh.read()
        if HOOK_MARK not in old and 'SpokenDebug' not in old:
            sys.exit('There is already a usercustomize.py that is not Grom\'s:\n  ' + hook_file +
                     '\nGrom did not touch it. Rename it, then run this again.')
    with open(hook_file, 'w', encoding='utf-8') as fh:
        fh.write(HOOK.format(folder=HERE))
    # old separate Grom files there are replaced by this one file
    for name in ('grom_voice.py', 'grom_explain.py', 'grom_explain_net.json',
                 'Гром_GROM.py', 'start_GROM.py', 'debugloop.py'):
        old_file = os.path.join(user_site, name)
        if os.path.exists(old_file):
            os.remove(old_file)
            print('  removed the old copy of', name, 'from', user_site)
    print('  hooked into Python')

    # 3. start at login
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE)
    winreg.SetValueEx(key, 'Grom', 0, winreg.REG_SZ,
                      '"{}" "{}" listen'.format(_pythonw(), os.path.abspath(__file__)))
    winreg.CloseKey(key)
    print('  set to start when you log in')

    # 4. start it now
    _stop_old_grom()
    subprocess.Popen([_pythonw(), os.path.abspath(__file__), 'listen'],
                     creationflags=0x00000008)  # no window
    print('Done. Listen for "Grom is on."')


def off():
    import site
    import winreg
    _stop_old_grom()
    try:
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE)
        winreg.DeleteValue(key, 'Grom')
        winreg.CloseKey(key)
    except OSError:
        pass
    hook_file = os.path.join(site.getusersitepackages(), 'usercustomize.py')
    try:
        with open(hook_file, encoding='utf-8') as fh:
            if HOOK_MARK in fh.read():
                os.remove(hook_file)
    except OSError:
        pass
    print('Grom is off. Run "python Гром_GROM.py" to turn it back on.')


# =================================================================
# TRAINING - teaches the explaining net from real errors (needs numpy)
#   1. runs thousands of small broken bits of Python and catches the real errors
#   2. trains the net to tell which kind of mistake each error is
#   3. tests it on errors made with names it never saw
#   4. saves grom_explain_net.json next to this file
# =================================================================

import random  # noqa: E402

TRAIN_WORDS = ['count', 'total', 'data', 'user', 'item', 'name', 'result', 'score',
               'value', 'frames', 'path', 'line', 'word', 'vocab', 'model', 'weights',
               'row', 'col', 'idx', 'token', 'text', 'config', 'batch', 'loss', 'net',
               'img', 'frame', 'grid', 'layer', 'cell', 'step', 'size', 'box', 'key']
TEST_WORDS = ['banana', 'ship', 'tolstoy', 'cloud', 'glass', 'pixel', 'ladder', 'rung',
              'thunder', 'wing', 'fuel', 'bomb_bay', 'pylon', 'paint', 'arch', 'story']
FAKE_MODULES = ['neuralkit', 'fastcv', 'speechbox', 'tinyml', 'pixelz', 'markovlib']


def make_namer(words, rnd, gibberish=False):
    def n():
        if gibberish and rnd.random() < 0.35:
            # made-up names teach the net that names don't matter
            return ''.join(rnd.choice('abcdefghijklmnopqrstuvwxyz_')
                           for _ in range(rnd.randint(1, 10))).strip('_') or 'q'
        a = rnd.choice(words)
        r = rnd.random()
        if r < 0.3:
            return a + '_' + rnd.choice(words)
        if r < 0.45:
            return a + str(rnd.randint(1, 99))
        if r < 0.55:
            return a.upper()[:1] + a[1:] + rnd.choice(words).capitalize()
        return a
    return n


# ---------------------------------------------------------------
# broken code, one list per kind of mistake. n() makes a name.
# ---------------------------------------------------------------
def templates(n, rnd):
    v = lambda: rnd.choice(['1', '2.5', '"hi"', '[1, 2]', '{}', 'None', '(3, 4)', 'True'])  # noqa: E731,F841
    i = lambda: str(rnd.randint(2, 40))  # noqa: E731
    T = {}
    a, b, c = n(), n() + 'X', n() + 'Y'
    T['name_not_defined'] = [f'print({a})', f'{b} = {a} + 1', f'for q in {a}: pass',
                             f'def f():\n    return {a}\nf()', f'{a}.append(3)']
    T['unbound_local'] = [f'{a} = 1\ndef f():\n    {a} += 1\nf()',
                          f'def f():\n    print({a})\n    {a} = 2\nf()',
                          f'def f():\n    if False:\n        {a} = 1\n    return {a}\nf()']
    T['none_attribute'] = [f'{a} = None\n{a}.{b}', f'{a} = [1].sort()\n{a}.append(2)',
                           f'def f(): pass\nf().{b}()', f'{a} = {{}}.get("k")\n{a}.{b}']
    obj = rnd.choice(['5', '2.5', '"txt"', '[1]', '{}', '(1,)', 'True', 'range(3)', 'b"x"'])
    obj2 = rnd.choice(['5', '2.5', '"txt"', '[1]', '{}', '(1,)', '{1}', 'range(3)'])
    T['attribute_missing'] = [f'({obj}).{b}', f'{a} = {obj}\n{a}.{b}()', f'({obj2}).{c}',
                              f'{a} = {obj2}\nprint({a}.{b})',
                              f'class {a.capitalize()}Thing: pass\n{a.capitalize()}Thing().{b}']
    mod = rnd.choice(['math', 'os', 'json', 're', 'random', 'time', 'string'])
    T['module_attribute'] = [f'import {mod}\n{mod}.{b}', f'import {mod}\n{mod}.{b}()']
    fm = rnd.choice(FAKE_MODULES) + str(rnd.randint(0, 999))
    T['module_not_found'] = [f'import {fm}', f'from {fm} import {a}', f'import {fm}.{a}']
    T['import_name'] = [f'from {mod} import {b}', f'from {mod} import {a}, {b}']
    key = rnd.choice([f'"{a}"', i(), f"'{b}'", f'("{a}", 1)'])
    T['key_error'] = [f'{a} = {{}}\n{a}[{key}]', f'{a} = {{"x": 1}}\nprint({a}[{key}])',
                      f'{a} = {{}}\n{a}.pop({key})', f'del {{}}[{key}]']
    T['index_error'] = [f'{a} = [1, 2]\n{a}[{i()}]', f'"{a}"[{i()}]', f'(1, 2)[{i()}]',
                        f'{a} = []\n{a}[0]', f'{a} = [0]\n{a}[{i()}] = 5', f'[][-1]']
    T['pop_empty'] = [f'{a} = []\n{a}.pop()', 'set().pop()',
                      'import collections\ncollections.deque().pop()']
    T['zero_div'] = [f'{a} = 0\n5 / {a}', f'{a} = 0\n5 // {a}', f'{a} = 0\n5 % {a}',
                     f'{a} = 0.0\n2.5 / {a}', f'sum([]) and 1 / 0 or 1 / 0']
    T['str_concat'] = [f'"{a}" + {rnd.choice(["1", "2.5", "None", "[1]", "True"])}',
                       f'{a} = 3\nprint("n = " + {a})', f'{a} = "x"\n{a} += {i()}']
    ops = rnd.choice(['+', '-', '*', '/', '**', '%', '//', '@', '&'])
    pair = rnd.choice([('1', '"s"'), ('[1]', '1.5'), ('None', '2'), ('{}', '{}'),
                       ('(1,)', '1'), ('2', 'None'), ('1.5', '"t"'), ('"a"', '"b"')])
    T['unsupported_operand'] = [f'{pair[0]} {ops} {pair[1]}',
                                f'{a} = {pair[0]}\n{a} {ops}= {pair[1]}']
    T['not_callable'] = [f'{a} = {obj}\n{a}()', f'{a} = 5\n{a}(2)', f'"{a}"()',
                         f'list = [1]\nlist("ab")', f'{a} = None\n{a}()']
    T['not_subscriptable'] = [f'{a} = 5\n{a}[0]', f'{a} = None\n{a}[0]',
                              f'def {a}(): pass\n{a}[1]', f'{a} = 2.5\n{a}[{i()}]', f'len[0]']
    T['not_iterable'] = [f'for {a} in 5: pass', f'{a} = None\nfor q in {a}: pass',
                         f'{a}, {b} = 5', f'list(2.5)', f'{a} = 3\nsum({a})',
                         f'{a} = None\n{b}, {c} = {a}']
    T['missing_arg'] = [f'def {a}(x): pass\n{a}()', f'def {a}(x, y): pass\n{a}(1)',
                        f'class K:\n    def __init__(self, {b}): pass\nK()',
                        f'def {a}({b}, {c}): pass\n{a}()']
    T['too_many_args'] = [f'def {a}(): pass\n{a}(1)', f'def {a}(x): pass\n{a}(1, 2)',
                          f'class K:\n    def {a}(): pass\nK().{a}()',
                          f'class K:\n    def {a}(x): pass\nK().{a}(1)']
    T['unexpected_kwarg'] = [f'def {a}(x): pass\n{a}(x=1, {b}=2)', f'print(1, {b}=2)',
                             f'def {a}(): pass\n{a}({b}=1)', f'sorted([1], {b}=True)']
    num = rnd.choice([f'"{a}"', '"3.5"', '"1,000"', '""', '" "', '"12a"', '"x7"'])
    T['not_a_number'] = [f'int({num})', f'float("{a}")', f'{a} = input_text = {num}\nint({a})',
                         f'int("{rnd.randint(1, 99)}.5")']
    T['unpack_count'] = [f'{a}, {b} = 1, 2, 3', f'{a}, {b}, {c} = [1, 2]',
                         f'{a}, {b} = "{a}{b}xyz"', f'for {a}, {b} in [(1, 2, 3)]: pass',
                         f'{a}, {b} = [1]']
    T['not_in_list'] = [f'{a} = [1, 2]\n{a}.remove({i()})', f'[1, 2].index({i()})',
                        f'["x"].index("{a}")', f'["x"].remove("{a}")']
    T['file_not_found'] = [f'open("{a}.txt")', f'open("{a}/{b}.csv")', f'open(r"C:\\{a}\\{b}.json")',
                           f'import os\nos.listdir("{a}_folder")', f'open("{a}.py", "r")']
    T['missing_colon'] = [f'if {a} == 1\n    pass', f'for {a} in {b}\n    pass',
                          f'def {a}()\n    pass', f'while {a}\n    pass', f'class {a.capitalize()}\n    pass']
    T['unclosed_bracket'] = [f'print({a}', f'{a} = [1, 2,', f'{a} = {{"x": 1', f'print(({a} + 1)',
                             f'{a} = f({b}, [{c}]']
    T['unterminated_string'] = [f'{a} = "hello', f"{a} = 'hi", f'print("{a})', f'{a} = """open']
    T['syntax_general'] = [f'{a} = = 1', f'print {a}', f'{a} {b} = 1', f'{a} + = 2',
                           f'def = {a}', f'{a} = 1 +', f'import', f'return {a} {b}',
                           f'{a} = [1 2 3]', f'if {a} = 1:\n    pass', f'{a}++']
    T['indentation'] = [f'if {a}:\n{b} = 1', f'{a} = 1\n    {b} = 2',
                        f'def {a}():\nreturn 1', f'for q in {a}:\npass',
                        f'if {a}:\n        x = 1\n    y = 2']
    T['tabs_spaces'] = [f'if {a}:\n\t{b} = 1\n        {c} = 2', f'def {a}():\n        x = 1\n\treturn x']
    T['recursion'] = [f'def {a}(x):\n    return {a}(x + 1)\n{a}(0)',
                      f'def {a}():\n    {a}()\n{a}()',
                      f'def {a}(x):\n    if x == -1:\n        return 0\n    return {a}(x + 1)\n{a}(1)']
    T['assertion'] = [f'{a} = 1\nassert {a} > 2', f'assert False, "{a} was wrong"',
                      f'{a} = 1\nassert {a} == 2', f'assert len([]) > 0, "{a} empty"']
    T['decode'] = [f'b"\\xff\\xfe{a}".decode("utf-8")', f'b"\\x9d{a}".decode("ascii")',
                   f'b"\\x81".decode("cp1252")', f'b"\\xe9{a}".decode("utf-8")']
    T['changed_size'] = [f'{a} = {{1: 2}}\nfor k in {a}:\n    {a}[k + 1] = 0',
                         f'{a} = {{1, 2}}\nfor k in {a}:\n    {a}.add(k + 5)',
                         f'{a} = {{"x": 1, "y": 2}}\nfor k in {a}:\n    del {a}[k]']
    T['json_bad'] = [f'import json\njson.loads("")', f'import json\njson.loads("{{{a}: 1}}")',
                     f'import json\njson.loads("[1, 2")', f'import json\njson.loads("{a}")']
    T['stop_iteration'] = [f'next(iter([]))', f'{a} = iter([])\nnext({a})',
                           f'{a} = (q for q in [])\nnext({a})']
    T['unhashable'] = [f'{{[1]: 2}}', f'{a} = set()\n{a}.add([1])', f'{{{{}}: 1}}',
                       f'{a} = {{}}\n{a}[[1, 2]] = 3', f'hash({{1}})']
    T['math_domain'] = [f'import math\nmath.sqrt(-{i()})', f'import math\nmath.log(0)',
                        f'import math\nmath.log(-{i()})', f'import math\nmath.acos({i()})']
    T['overflow'] = [f'import math\nmath.exp({rnd.randint(800, 5000)})', f'2.0 ** {rnd.randint(2000, 9000)}',
                     f'import math\nmath.pow(10, {rnd.randint(400, 900)})']
    T['item_assign'] = [f'{a} = "{a}"\n{a}[0] = "x"', f'{a} = (1, 2)\n{a}[0] = 5',
                        f'{a} = "abc"\n{a}[{i()}] = "q"', f'{a} = b"xy"\n{a}[0] = 1']
    T['compare'] = [f'"{a}" < {i()}', f'None > 1', f'{a} = [1, "x"]\nsorted({a})',
                    f'max([1, "{a}"])', f'[1] < (1,)', f'{{}} > {{}}']
    T['bad_convert_type'] = [f'int(None)', f'float([1])', f'int({{}})', f'{a} = None\nint({a})',
                             f'float(None)']
    return T


def run_one(code):
    import io
    import contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, 'script.py', 'exec'), {'__name__': 'x'})
    except RecursionError as e:
        return type(e).__name__, str(e)
    except Exception as e:
        return type(e).__name__, str(e)
    return None


# phrasings from older Python versions (3.8 - 3.11), so it still works there
OLD_PYTHON = [
    ('unbound_local', 'UnboundLocalError', "local variable '{a}' referenced before assignment"),
    ('unclosed_bracket', 'SyntaxError', 'unexpected EOF while parsing (script.py, line 1)'),
    ('unterminated_string', 'SyntaxError', 'EOL while scanning string literal (script.py, line 1)'),
    ('unterminated_string', 'SyntaxError', 'unterminated string literal (detected at line 1) (script.py, line 1)'),
    ('name_not_defined', 'NameError', "name '{a}' is not defined"),
    ('indentation', 'IndentationError', 'expected an indented block (script.py, line 2)'),
    ('str_concat', 'TypeError', 'can only concatenate str (not "{t}") to str'),
    ('not_subscriptable', 'TypeError', "'{t}' object is not subscriptable"),
    ('missing_arg', 'TypeError', "{a}() missing 1 required positional argument: '{b}'"),
    ('too_many_args', 'TypeError', '{a}() takes 0 positional arguments but 1 was given'),
    ('unexpected_kwarg', 'TypeError', "{a}() got an unexpected keyword argument '{b}'"),
    ('bad_convert_type', 'TypeError', "int() argument must be a string, a bytes-like object or a number, not '{t}'"),
    ('bad_convert_type', 'TypeError', "float() argument must be a string or a number, not '{t}'"),
    ('not_iterable', 'TypeError', "cannot unpack non-iterable {t} object"),
]
TYPES = ['int', 'float', 'NoneType', 'list', 'dict', 'tuple', 'function', 'bool', 'str', 'method']

# which real error kind must come back for each label (catches broken templates)
EXPECT = {
    'name_not_defined': {'NameError'}, 'unbound_local': {'UnboundLocalError'},
    'none_attribute': {'AttributeError'}, 'attribute_missing': {'AttributeError'},
    'module_attribute': {'AttributeError'}, 'module_not_found': {'ModuleNotFoundError'},
    'import_name': {'ImportError'}, 'key_error': {'KeyError'}, 'index_error': {'IndexError'},
    'pop_empty': {'IndexError', 'KeyError'}, 'zero_div': {'ZeroDivisionError'},
    'str_concat': {'TypeError'}, 'unsupported_operand': {'TypeError'},
    'not_callable': {'TypeError'}, 'not_subscriptable': {'TypeError'},
    'not_iterable': {'TypeError'}, 'missing_arg': {'TypeError'}, 'too_many_args': {'TypeError'},
    'unexpected_kwarg': {'TypeError'}, 'not_a_number': {'ValueError'},
    'unpack_count': {'ValueError'}, 'not_in_list': {'ValueError'},
    'file_not_found': {'FileNotFoundError'}, 'missing_colon': {'SyntaxError'},
    'unclosed_bracket': {'SyntaxError'}, 'unterminated_string': {'SyntaxError'},
    'syntax_general': {'SyntaxError'}, 'indentation': {'IndentationError'},
    'tabs_spaces': {'TabError'}, 'recursion': {'RecursionError'},
    'assertion': {'AssertionError'}, 'decode': {'UnicodeDecodeError'},
    'changed_size': {'RuntimeError'}, 'json_bad': {'JSONDecodeError'},
    'stop_iteration': {'StopIteration'}, 'unhashable': {'TypeError'},
    'math_domain': {'ValueError'}, 'overflow': {'OverflowError'},
    'item_assign': {'TypeError'}, 'compare': {'TypeError'}, 'bad_convert_type': {'TypeError'},
    'permission': {'PermissionError'},
}


# words the message must have, for kinds that share an error type
MUST_SAY = {
    'str_concat': 'concatenate', 'unsupported_operand': 'unsupported operand',
    'not_callable': 'not callable', 'not_subscriptable': 'not subscriptable',
    'not_iterable': 'iterable', 'missing_arg': 'missing', 'too_many_args': 'positional argument',
    'unexpected_kwarg': 'unexpected keyword', 'not_a_number': 'convert|invalid literal',
    'unpack_count': 'unpack', 'not_in_list': 'not in', 'unhashable': 'unhashable',
    'math_domain': 'domain', 'item_assign': 'assignment', 'compare': 'not supported between',
    'bad_convert_type': 'argument must be', 'none_attribute': 'NoneType',
    'attribute_missing': "^'(?!NoneType)", 'module_attribute': '^module',
    'missing_colon': "expected ':'", 'unclosed_bracket': 'never closed|was not closed',
    'unterminated_string': 'unterminated', 'changed_size': 'changed size',
    'syntax_general': "^(?!expected ':')(?!.*never closed)(?!.*unterminated)",
}


def permission_errors(n, rnd):
    # can't make these reliably by running code, so write the real message shape
    rows = []
    for _ in range(4):
        name = rnd.choice([n() + '.txt', n() + '.csv', 'C:\\Users\\me\\' + n(), n() + '_folder',
                           n() + '.json', 'C:\\Users\\me\\Desktop\\' + n() + '.png'])
        rows.append(('permission', 'PermissionError', "[Errno 13] Permission denied: '{}'".format(name)))
    return rows


def make_set(words, seed, rounds):
    rnd = random.Random(seed)
    n = make_namer(words, rnd, gibberish=(words is TRAIN_WORDS))
    rows, bad = [], {}
    for _ in range(rounds):
        for label, codes in templates(n, rnd).items():
            for code in codes:
                got = run_one(code)
                if (got is None or got[0] not in EXPECT[label] or
                        (label in MUST_SAY and not re.search(MUST_SAY[label], got[1]))):
                    bad.setdefault(label, []).append((code, got))
                    continue
                rows.append((label, got[0], got[1]))
        rows += permission_errors(n, rnd)
        for label, kind, msg in OLD_PYTHON:
            rows.append((label, kind, msg.format(a=n(), b=n(), t=rnd.choice(TYPES))))
    return rows, bad


def to_matrix(rows, labels):
    import numpy as np
    X = np.zeros((len(rows), SIZE), dtype=np.float32)
    y = np.array([labels.index(r[0]) for r in rows])
    for k, (_, kind, msg) in enumerate(rows):
        for i, v in features(kind, msg).items():
            X[k, i] = v
    return X, y


def train_net(X, y, n_labels, hidden=64, epochs=60, lr=0.05, seed=0):
    import numpy as np
    rs = np.random.RandomState(seed)
    W1 = rs.randn(X.shape[1], hidden).astype(np.float32) * 0.05
    b1 = np.zeros(hidden, np.float32)
    W2 = rs.randn(hidden, n_labels).astype(np.float32) * 0.1
    b2 = np.zeros(n_labels, np.float32)
    for ep in range(epochs):
        order = rs.permutation(len(X))
        total = 0.0
        for s in range(0, len(X), 64):
            idx = order[s:s + 64]
            xb, yb = X[idx], y[idx]
            h = np.maximum(xb @ W1 + b1, 0)
            z = h @ W2 + b2
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            total += -np.log(p[np.arange(len(yb)), yb] + 1e-9).sum()
            g = p.copy()
            g[np.arange(len(yb)), yb] -= 1
            g /= len(yb)
            gW2 = h.T @ g
            gh = (g @ W2.T) * (h > 0)
            gW1 = xb.T @ gh
            W2 -= lr * gW2 + lr * 1e-4 * W2
            b2 -= lr * g.sum(0)
            W1 -= lr * gW1 + lr * 1e-4 * W1
            b1 -= lr * gh.sum(0)
        if ep % 10 == 0 or ep == epochs - 1:
            print('  lesson round {:2d}  error {:.4f}'.format(ep + 1, total / len(X)))
    return W1, b1, W2, b2


# ---------------------------------------------------------------
# training the second net (the run watcher)
#   1. makes thousands of runs: real little learning loops that go
#      right, blow up or stall, plus ordinary output, stuck loops,
#      warnings, and runs that take far longer or shorter than usual
#   2. trains the net to say which signs each run shows
#   3. tests it on log styles and words it never saw
#   4. saves grom_run_net.json next to this file
# ---------------------------------------------------------------

LOG_STYLES_TRAIN = [
    'epoch {i} loss {v:.4f}',
    'Epoch {i}/{n} - loss: {v:.4f}',
    'step {i:5d} | loss: {v:.4f}',
    'iter {i}: loss={v:.5f}',
    '[{i}] cost = {v:.4f}',
    'epoch {i:3d}  train_loss {v:.4f}  val_loss {u:.4f}',
    '{i}/{n} mse: {v:.3e}',
    'batch {i} loss {v:.6f} lr 0.001',
]
LOG_STYLES_TEST = [
    "{{'loss': {v:.4f}, 'learning_rate': 5e-05, 'epoch': {i}}}",
    'Epoch {i}: 100%|##########| 32/32 [00:01<00:00, train_loss={v:.3f}]',
    'it {i} -- training loss {v:.4f}',
    'Loss: {v:.4f} (epoch {i})',
]
PLAIN_TRAIN = ['Loaded {k} rows from {w}.csv', 'Processing {w} ...', 'Saved {w}.png', 'Found {k} files',
               '{w}: {k}', 'Done.', 'Total: {f:.2f}', 'accuracy {f:.3f}', 'result = [{k}, {k2}, {k3}]',
               'INFO: started {w}', '{w} {w2} {w3}', '-' * 20, '', 'writing {w}.json', 'best score {f:.4f}']
PLAIN_TEST = ['Reading {w}.txt', 'got {k} {w}s', 'average {f:.3f}', 'model saved to {w}/', '>>> {w2} ok',
              'financial {w} report', 'nancy {w}', 'information: {k}']
WORDS_TRAIN = ['frames', 'data', 'vocab', 'model', 'user', 'grid', 'tokens', 'scores', 'images', 'net']
WORDS_TEST = ['ship', 'tolstoy', 'cloud', 'pixel', 'pylon', 'thunder', 'paint']
STUCK_LINES = ['waiting for data...', 'None', '0', 'retrying', 'Connection refused, retrying',
               'still running', 'ERROR: queue empty', 'tick', '[]']
HARD_WARNINGS = [('RuntimeWarning', 'overflow encountered in multiply'),
                 ('RuntimeWarning', 'invalid value encountered in divide'),
                 ('RuntimeWarning', 'divide by zero encountered in log'),
                 ('UserWarning', 'The given NumPy array is not writable'),
                 ('RuntimeWarning', 'Mean of empty slice')]
SOFT_WARNING_LIST = [('DeprecationWarning', 'This function is deprecated'),
                     ('FutureWarning', 'The default value will change in a future version'),
                     ('PendingDeprecationWarning', 'matrix subclass is not the recommended way')]


def _learning_run(rnd, mode, style):
    # a real little learning loop (linear regression), run for real
    import warnings
    import numpy as np
    rs = np.random.RandomState(rnd.randint(0, 10 ** 8))
    d, n = rnd.randint(3, 40), rnd.randint(60, 500)
    X = rs.randn(n, d)
    y = X @ rs.randn(d) + rs.randn(n) * rnd.choice([0.01, 0.1, 0.5, 1.0])
    lr = {'good': 10 ** rnd.uniform(-2.3, -0.8), 'high': 10 ** rnd.uniform(0.3, 1.3),
          'low': 10 ** rnd.uniform(-7, -5.5), 'stuck': 0.0,
          'wrong_sign': 10 ** rnd.uniform(-3.3, -1.8)}[mode]
    sign = 1 if mode == 'wrong_sign' else -1     # the bug: + instead of -
    scale = 10 ** rnd.uniform(-2, 2)
    offset = 0.0 if rnd.random() < 0.7 else rnd.uniform(-300, 300) * scale
    epochs = rnd.randint(8, 300) if mode != 'high' or rnd.random() < 0.5 else rnd.randint(8, 60)
    every = max(1, epochs // rnd.randint(5, 60))
    batch = rnd.choice([n, 32, 64])
    w = np.zeros(d)
    cap = 10 ** rnd.uniform(2, 250) if mode == 'high' and rnd.random() < 0.5 else None
    lines, values = [], []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for ep in range(1, epochs + 1):
            idx = rs.choice(n, min(batch, n), replace=False)
            err = X[idx] @ w - y[idx]
            loss = float(np.mean(err ** 2)) * scale + offset
            w = w + sign * lr * (2 * X[idx].T @ err / len(idx))
            if ep == 1 or ep % every == 0:
                values.append(loss)
                u = loss * rnd.uniform(1.0, 1.3) if math.isfinite(loss) else loss
                lines.append(style.format(i=ep, n=epochs, v=loss, u=u))
            if cap and loss > cap:
                break
    warns = [(c.category.__name__, 'train.py', rnd.randint(5, 80), str(c.message)) for c in caught]
    finite = [v for v in values if math.isfinite(v)]
    up = mode in ('high', 'wrong_sign') and _visibly_up(finite)
    flat = mode in ('low', 'stuck') and len(finite) >= 4
    return lines, warns, up, flat


def _visibly_up(finite):
    # the end of the loss is clearly above its start, judged on the same
    # log-like scale the net uses (so explosive climbs count)
    if len(finite) < 4:
        return False
    mid = _median([abs(v) for v in finite]) + 1e-12
    sq = [math.asinh(v / mid) for v in finite]
    k = max(1, len(sq) // 10)
    first, last = sum(sq[:k]) / k, sum(sq[-k:]) / k
    return last - first > 0.5 * (max(sq) - min(sq)) and \
        sum(finite[-k:]) / k - sum(finite[:k]) / k > 0.02 * (sum(abs(v) for v in finite) / len(finite))


def _plain_lines(rnd, templates, words, how_many):
    out = []
    for _ in range(how_many):
        t = rnd.choice(templates)
        out.append(t.format(w=rnd.choice(words), w2=rnd.choice(words), w3=rnd.choice(words),
                            k=rnd.randint(0, 5000), k2=rnd.randint(0, 99), k3=rnd.randint(0, 99),
                            f=rnd.uniform(0, 100)))
    return out


def make_run(rnd, test=False):
    """One made-up run of a script: what it printed, how long, its past runs, and the true signs."""
    styles = LOG_STYLES_TEST if test else LOG_STYLES_TRAIN
    plain = PLAIN_TEST if test else PLAIN_TRAIN
    words = WORDS_TEST if test else WORDS_TRAIN
    label = dict.fromkeys(SIGNS, 0)
    warns = []

    # what it printed
    if rnd.random() < 0.45:
        mode = rnd.choice(['good', 'good', 'good', 'high', 'wrong_sign', 'low', 'stuck'])
        lines, warns, up, flat = _learning_run(rnd, mode, rnd.choice(styles))
        label['loss_up'], label['loss_flat'] = int(up), int(flat)
        lines = _plain_lines(rnd, plain, words, rnd.randint(0, 5)) + lines + \
            _plain_lines(rnd, plain, words, rnd.randint(0, 3))
    else:
        lines = _plain_lines(rnd, plain, words, rnd.choice([1, 3, 10, 40, 150, 600]))
        if rnd.random() < 0.08:
            lines.insert(rnd.randint(0, len(lines)), 'value: {}'.format(rnd.choice(['nan', 'inf', '-inf', 'NaN'])))

    # a stuck loop, or a harmless short repeat
    if rnd.random() < 0.2:
        stuck = rnd.choice(STUCK_LINES)
        if rnd.random() < 0.6:
            many = rnd.choice([30, 60, 200, 1000, 3000])
            label['repeating'] = 1
        else:
            many = rnd.randint(2, 8)
        at = rnd.randint(0, len(lines))
        lines = lines[:at] + [stuck] * many + (lines[at:] if rnd.random() < 0.5 else [])

    # warnings
    if rnd.random() < 0.2:
        kinds = HARD_WARNINGS if rnd.random() < 0.6 else SOFT_WARNING_LIST
        for _ in range(rnd.randint(1, 3)):
            k, m = rnd.choice(kinds)
            warns.append((k, 'script.py', rnd.randint(1, 200), m))

    # its past runs: how long they took, how much they printed
    usual_t = 10 ** rnd.uniform(-1.6, 3.5)
    usual_lines = max(0, int(len(lines) * rnd.uniform(0.6, 1.6)))
    spread = rnd.uniform(0.03, 0.4)
    past = [{'t': usual_t * math.exp(rnd.gauss(0, spread)),
             'lines': max(0, int(usual_lines * math.exp(rnd.gauss(0, spread))))}
            for _ in range(rnd.choice([0, 0, 1, 2, 3, 5, 10, 20]))]

    # printed nothing at all
    if rnd.random() < 0.08:
        lines = []
        if past and _median([p['lines'] for p in past]) >= 3:
            label['no_output'] = 1
        label['loss_up'] = label['loss_flat'] = label['repeating'] = 0

    # how long this run took
    r = rnd.random()
    if r < 0.15:
        seconds = usual_t * rnd.uniform(5, 60)
        if len(past) >= 3 and seconds - usual_t >= 3:
            label['too_slow'] = 1
    elif r < 0.3:
        seconds = usual_t / rnd.uniform(5, 60)
        if len(past) >= 3 and usual_t - seconds >= 1:
            label['too_fast'] = 1
        if lines and rnd.random() < 0.5:
            lines = lines[:max(1, int(len(lines) * rnd.uniform(0.05, 0.6)))]   # quit early, printed less
    else:
        seconds = usual_t * math.exp(rnd.gauss(0, 0.3))

    # a loss sign only counts if the loss can actually be seen in what got printed
    _, seen = _loss_series(lines)
    seen = [v for v in seen if math.isfinite(v)]
    if len(seen) < 4:
        label['loss_up'] = label['loss_flat'] = 0
    elif label['loss_up'] and not _visibly_up(seen):
        label['loss_up'] = 0
    if any(NUMBER_GONE_BAD.search(ln) for ln in lines):
        label['nan'] = 1
    if any(w[0] not in SOFT_WARNINGS for w in warns):
        label['warnings'] = 1
    return lines, len(lines), seconds, past, warns, [label[s] for s in SIGNS]


def make_run_set(seed, count, test=False):
    import warnings
    warnings.filterwarnings('ignore')
    rnd = random.Random(seed)
    X, Y = [], []
    for _ in range(count):
        lines, n, seconds, past, warns, y = make_run(rnd, test)
        X.append(run_features(lines, n, seconds, past, warns)[0])
        Y.append(y)
    return X, Y


def _train_small_net(X, Y, hidden, epochs, lr, seed):
    import numpy as np
    mean, std = X.mean(0), np.maximum(X.std(0), 1e-3)
    Xn = (X - mean) / std
    rs = np.random.RandomState(seed)
    W1 = rs.randn(X.shape[1], hidden) * 0.3
    b1 = np.zeros(hidden)
    W2 = rs.randn(hidden, Y.shape[1]) * 0.3
    b2 = np.zeros(Y.shape[1])
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t, total = 0, 0.0
    for ep in range(epochs):
        order = rs.permutation(len(X))
        total = 0.0
        for s in range(0, len(X), 128):
            idx = order[s:s + 128]
            xb, yb = Xn[idx], Y[idx]
            h = np.maximum(xb @ W1 + b1, 0)
            p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -30, 30)))
            total += -(yb * np.log(p + 1e-9) + (1 - yb) * np.log(1 - p + 1e-9)).sum()
            g = (p - yb) / len(idx)
            gh = (g @ W2.T) * (h > 0)
            grads = [xb.T @ gh + 1e-4 * W1, gh.sum(0), h.T @ g + 1e-4 * W2, g.sum(0)]
            t += 1
            for k, (pa, gr) in enumerate(zip(params, grads)):   # Adam
                m[k] = 0.9 * m[k] + 0.1 * gr
                v[k] = 0.999 * v[k] + 0.001 * gr * gr
                pa -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return {'mean': mean.round(6).tolist(), 'std': std.round(6).tolist(),
            'W1': W1.round(5).tolist(), 'b1': b1.round(5).tolist(),
            'W2': W2.round(5).tolist(), 'b2': b2.round(5).tolist()}, total / len(X) / Y.shape[1]


def train_run_net(X, Y, hidden=24, epochs=60, lr=0.01, seed=0):
    import numpy as np
    X, Y = np.array(X, np.float64), np.array(Y, np.float64)
    groups = []
    for labels, feats in SIGN_GROUPS:
        cols = [SIGNS.index(lb) for lb in labels]
        part, err = _train_small_net(X[:, feats], Y[:, cols], hidden, epochs, lr, seed)
        part.update(labels=labels, features=feats)
        groups.append(part)
        print('  trained the net for {:22s} error {:.4f}'.format(' and '.join(labels), err))
    return {'groups': groups}


def retrain_run():
    import numpy as np
    global _run_net
    print('Making training runs (real little learning loops, stuck loops, warnings, timing)...')
    X, Y = make_run_set(seed=1, count=40000)
    print('  {} training runs'.format(len(X)))
    print('Making test runs with log styles and words the net never saw...')
    Xt, Yt = make_run_set(seed=2, count=4000, test=True)
    print('  {} test runs'.format(len(Xt)))
    print('Training...')
    net = train_run_net(X, Y)
    _run_net = net
    P = np.array([[run_guess(x)[s] for s in SIGNS] for x in Xt])
    Yt = np.array(Yt)
    said = P >= SPEAK_ABOVE
    print('Test, on runs it never saw:')
    print('  fine runs Grom stayed quiet on: {:.1f}%'.format(100 * (~said[Yt.sum(1) == 0].any(1)).mean()))
    for k, s in enumerate(SIGNS):
        real = Yt[:, k] == 1
        caught = (said[real, k]).mean() if real.any() else float('nan')
        false = (said[~real, k]).mean()
        print('  {:10s} caught {:5.1f}% of {:4d}   wrongly said {:4.1f}%'.format(
            s, 100 * caught, int(real.sum()), 100 * false))
    with open(RUN_NET_FILE, 'w', encoding='utf-8') as fh:
        json.dump(net, fh, separators=(',', ':'))
    print('Saved grom_run_net.json')



def retrain():
    import warnings
    import numpy as np
    warnings.filterwarnings('ignore')
    sys.setrecursionlimit(200)

    print('Making training errors by running broken code...')
    train_rows, _ = make_set(TRAIN_WORDS, seed=1, rounds=40)
    print('  {} training errors'.format(len(train_rows)))
    print('Making test errors with names the net never saw...')
    test_rows, _ = make_set(TEST_WORDS, seed=2, rounds=8)
    print('  {} test errors'.format(len(test_rows)))

    labels = sorted(EXPLAIN)
    X, y = to_matrix(train_rows, labels)
    Xt, yt = to_matrix(test_rows, labels)

    print('Training...')
    W1, b1, W2, b2 = train_net(X, y, len(labels))

    h = np.maximum(Xt @ W1 + b1, 0)
    pred = (h @ W2 + b2).argmax(1)
    print('Test: {:.1f}% right on errors it never saw'.format(100 * (pred == yt).mean()))

    net = {'labels': labels,
           'W1': np.round(W1, 4).tolist(), 'b1': np.round(b1, 4).tolist(),
           'W2': np.round(W2, 4).tolist(), 'b2': np.round(b2, 4).tolist()}
    with open(NET_FILE, 'w', encoding='utf-8') as fh:
        json.dump(net, fh, separators=(',', ':'))
    print('Saved grom_explain_net.json')
    print()
    print('Now the second net, the run watcher.')
    retrain_run()


# =================================================================

if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'setup'
    if command == 'listen':
        listen()
    elif command == 'train':
        retrain()
    elif command == 'off':
        off()
    else:
        setup()
