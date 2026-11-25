# r: numpy
# ---- Grasshopper Python 3 component (Rhino 8): bird songs (.wav) -> 3D geometry ----
# The peak detection now happens HERE, straight from the .wav files.
#
# INPUTS  : folder (str)    folder that holds the .wav files
#           song (int)        0, 1, 2 ... picks a song.  -1 = show ALL songs side by side
#           mm_per_s (float)  how long one second of song is (e.g. 15)
#           max_gap (float)   shorten silences longer than this many seconds (e.g. 1). 0 = keep real silences
#           y_len (float)     total height of the pitch axis (e.g. 100)
#           z_len (float)     total height of the loudness axis (e.g. 60)
#           r_min (float)     smallest sphere radius (quietest peak)  (e.g. 1.5)
#           r_max (float)     biggest sphere radius (loudest peak)    (e.g. 5)
#           spacing (float)   gap between songs when song = -1 (e.g. 40)
# OUTPUTS : pts  spheres  curve  info
#
# Needs numpy. In Rhino 8 the "# r: numpy" line on top installs it automatically
# the first time the component runs (give it a minute). Rhino 7's GhPython can't use numpy.
import os, glob, struct
import numpy as np
import Rhino.Geometry as rg
import scriptcontext as sc

# ---- SHARED SCALES: the same for every song, so songs are directly comparable ----
FREQ_LO, FREQ_HI = 2500.0, 8000.0   # pitch range in Hz  -> 0 .. y_len
DB_LO,   DB_HI   = -30.0,  0.0      # loudness range in dB -> 0 .. z_len (and sphere size)

# ---- PEAK-DETECTION SETTINGS (same as the bird_peaks.py script; edit here) ----
N_FFT      = 2048      # FFT window size
HOP        = 256       # step between frames
BAND_LO    = 1500.0    # lowest frequency searched for peaks, Hz (2500 drops low noise)
BAND_HI    = 11000.0   # highest frequency searched for peaks, Hz
PROMINENCE = 10.0      # how much a peak must stand out from its surroundings, dB
MIN_GAP_S  = 0.05      # minimum time between two peaks, seconds


def clamp01(v):
    return max(0.0, min(1.0, v))


# ================= 1. READ A .WAV FILE (no extra libraries) =================
def read_wav(path):
    """Returns (mono float samples, sample rate). Handles 8/16/24/32-bit PCM and 32/64-bit float."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a WAV file: %s" % path)
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, channels, sr = struct.unpack("<HHI", body[0:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if tag == 0xFFFE:                               # WAVE_FORMAT_EXTENSIBLE
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, channels, sr, bits)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)                        # chunks are padded to even length
    if fmt is None or raw is None:
        raise ValueError("Broken WAV file: %s" % path)

    tag, channels, sr, bits = fmt
    width = bits // 8
    raw = raw[:len(raw) - len(raw) % (width * channels)]
    if tag == 3:                                            # float
        y = np.frombuffer(raw, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
    elif tag == 1 and bits == 8:
        y = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif tag == 1 and bits == 16:
        y = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        y = v.astype(np.float64) / float(1 << 23)
    elif tag == 1 and bits == 32:
        y = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raise ValueError("Unsupported WAV format (tag %d, %d-bit): %s" % (tag, bits, path))
    if channels > 1:
        y = y.reshape(-1, channels).mean(axis=1)            # stereo -> mono
    return y, sr


# ================= 2. FIND THE PEAKS =================
def find_peaks(x, distance, min_prominence):
    """Simple version of scipy.signal.find_peaks(x, distance=..., prominence=...)."""
    n = len(x)
    # a) local maxima (flat tops count once, at their middle)
    cand = []
    i = 1
    while i < n - 1:
        if x[i - 1] < x[i]:
            j = i
            while j + 1 < n - 1 and x[j + 1] == x[i]:
                j += 1
            if x[j + 1] < x[i]:
                cand.append((i + j) // 2)
                i = j
        i += 1
    cand = np.array(cand, dtype=int)
    if len(cand) == 0:
        return cand, np.array([])

    # b) keep only the highest peak inside every 'distance' window
    keep = np.ones(len(cand), dtype=bool)
    for k in np.argsort(x[cand])[::-1]:
        if not keep[k]:
            continue
        lo = np.searchsorted(cand, cand[k] - distance + 1)
        hi = np.searchsorted(cand, cand[k] + distance)          # peaks closer than distance
        keep[lo:hi] = False
        keep[k] = True
    cand = cand[keep]

    # c) prominence = height above the higher of the two lowest points on either side
    proms = np.empty(len(cand))
    for k, p in enumerate(cand):
        h = x[p]
        left = p
        left_min = h
        while left > 0 and x[left - 1] <= h:
            left -= 1
            left_min = min(left_min, x[left])
        right = p
        right_min = h
        while right < n - 1 and x[right + 1] <= h:
            right += 1
            right_min = min(right_min, x[right])
        proms[k] = h - max(left_min, right_min)
    good = proms >= min_prominence
    return cand[good], proms[good]


def analyse_wav(path):
    """Returns a list of (time_s, frequency_hz, intensity_db), one per detected peak."""
    y, sr = read_wav(path)

    # spectrogram (same as librosa.stft: Hann window, centred frames)
    pad = N_FFT // 2
    y = np.pad(y, pad)
    n_frames = 1 + (len(y) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    S = np.abs(np.fft.rfft(y[idx] * window, axis=1))          # frames x freqs
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    times = np.arange(n_frames) * HOP / float(sr)

    # amplitude -> dB, 0 dB = loudest point of the whole recording (like librosa ref=np.max)
    S_db = 20 * np.log10(np.maximum(S, 1e-5)) - 20 * np.log10(max(S.max(), 1e-5))
    S_db = np.maximum(S_db, -80.0)

    # loudest frequency in the band, per frame
    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    S_band, f_band = S_db[:, band], freqs[band]
    dom_freq = f_band[S_band.argmax(axis=1)]
    dom_db = S_band.max(axis=1)

    dist = max(1, int(MIN_GAP_S * sr / HOP))
    peak_idx, _ = find_peaks(dom_db, dist, PROMINENCE)
    return [(float(times[i]), float(dom_freq[i]), float(dom_db[i])) for i in peak_idx]


def get_peaks(path):
    """Analysis is cached, so moving sliders doesn't redo the FFT. Re-runs if the file
    or a detection setting changes."""
    key = ("birdpeaks", path, os.path.getmtime(path), BAND_LO, BAND_HI, PROMINENCE, MIN_GAP_S)
    if key not in sc.sticky:
        sc.sticky[key] = analyse_wav(path)
    return sc.sticky[key]


# ================= 3. BUILD THE GEOMETRY =================
files = sorted(glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.WAV")))
files = sorted(set(files))
if not files:
    raise ValueError("No .wav files found in: %s" % folder)

if song is None or song < 0:
    chosen = list(range(len(files)))          # all songs
else:
    chosen = [int(song) % len(files)]         # one song (wraps round if the slider goes too high)

pts, spheres, curve = [], [], []
info_lines = []

for slot, idx in enumerate(chosen):
    name = os.path.splitext(os.path.basename(files[idx]))[0]
    rows = sorted(get_peaks(files[idx]))
    if not rows:
        info_lines.append("%d: %s (no peaks found)" % (idx, name))
        continue

    y_off = slot * (y_len + spacing) if len(chosen) > 1 else 0.0   # side-by-side when showing all

    # TIME -> X, starting at 0, with long silences shortened if max_gap > 0
    song_pts = []
    t_now, t_prev = 0.0, rows[0][0]
    for (ti, fi, di) in rows:
        dt = ti - t_prev
        if max_gap and max_gap > 0:
            dt = min(dt, max_gap)
        t_now += dt
        t_prev = ti

        x = t_now * mm_per_s                                                # time    -> X
        y = clamp01((fi - FREQ_LO) / (FREQ_HI - FREQ_LO)) * y_len + y_off   # pitch   -> Y
        n = clamp01((di - DB_LO) / (DB_HI - DB_LO))                         # 0 quiet, 1 loud
        z = n * z_len                                                       # loudness -> Z

        p = rg.Point3d(x, y, z)
        song_pts.append(p)
        spheres.append(rg.Sphere(p, r_min + n * (r_max - r_min)))           # loudness -> size

    pts.extend(song_pts)
    if len(song_pts) > 3:
        curve.append(rg.Curve.CreateInterpolatedCurve(song_pts, 3))
    info_lines.append("%d: %s (%d peaks, %.0f mm long)" % (idx, name, len(rows), song_pts[-1].X))

info = " | ".join(info_lines)