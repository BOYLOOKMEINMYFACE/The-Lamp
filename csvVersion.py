# ---- Grasshopper Python component: bird songs -> 3D geometry, switchable ----
# INPUTS  : folder (str)    folder that holds the peaks_*.csv files
#           song (int)      0, 1, 2 ... picks a song.  -1 = show ALL songs side by side
#           mm_per_s (float)  how long one second of song is (e.g. 15)
#           max_gap (float)   shorten silences longer than this many seconds (e.g. 1). 0 = keep real silences
#           y_len (float)   total height of the pitch axis (e.g. 100)
#           z_len (float)   total height of the loudness axis (e.g. 60)
#           r_min (float)   smallest sphere radius (quietest peak)  (e.g. 1.5)
#           r_max (float)   biggest sphere radius (loudest peak)    (e.g. 5)
#           spacing (float) gap between songs when song = -1 (e.g. 40)
# OUTPUTS : pts  spheres  curve  info
import os, glob, csv
import Rhino.Geometry as rg

# ---- SHARED SCALES: the same for every song, so songs are directly comparable ----
FREQ_LO, FREQ_HI = 2500.0, 8000.0   # pitch range in Hz  -> 0 .. y_len
DB_LO,   DB_HI   = -30.0,  0.0      # loudness range in dB -> 0 .. z_len (and sphere size)

print(folder)

def clamp01(v):
    return max(0.0, min(1.0, v))

# 1. FIND the song files
files = sorted(glob.glob(os.path.join(folder, "peaks_*.csv")))
if not files:
    raise ValueError("No peaks_*.csv files found in: %s" % folder)

# 2. WORK OUT which songs to build
if song is None or song < 0:
    chosen = list(range(len(files)))          # all songs
else:
    chosen = [int(song) % len(files)]         # one song (wraps round if the slider goes too high)

pts, spheres, curve = [], [], []
info_lines = []

for slot, idx in enumerate(chosen):
    name = os.path.basename(files[idx])[6:-4]
    rows = []
    with open(files[idx]) as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["time_s"]), float(r["frequency_hz"]), float(r["intensity_db"])))
    rows.sort()

    y_off = slot * (y_len + spacing) if len(chosen) > 1 else 0.0   # side-by-side when showing all

    # 3. TIME -> X, starting at 0, with long silences shortened if max_gap > 0
    song_pts = []
    t_now, t_prev = 0.0, rows[0][0]
    for (ti, fi, di) in rows:
        dt = ti - t_prev
        if max_gap and max_gap > 0:
            dt = min(dt, max_gap)
        t_now += dt
        t_prev = ti

        x = t_now * mm_per_s                                   # time      -> X
        y = clamp01((fi - FREQ_LO) / (FREQ_HI - FREQ_LO)) * y_len + y_off   # pitch -> Y
        n = clamp01((di - DB_LO) / (DB_HI - DB_LO))            # 0 = quiet, 1 = loud
        z = n * z_len                                          # loudness  -> Z

        p = rg.Point3d(x, y, z)
        song_pts.append(p)
        spheres.append(rg.Sphere(p, r_min + n * (r_max - r_min)))   # loudness -> size

    pts.extend(song_pts)
    if len(song_pts) > 3:
        curve.append(rg.Curve.CreateInterpolatedCurve(song_pts, 3))
    info_lines.append("%d: %s (%d peaks, %.0f mm long)" % (idx, name, len(rows), song_pts[-1].X))

# info = " | ".join(info_lines)
info = info_lines
print(info)