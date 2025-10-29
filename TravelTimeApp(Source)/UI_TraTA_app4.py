# UI_TraTA_app4.py
# Tkinter UI for TravelTimeAnalysis2 with progress bar + figure gallery (no external windows)
# Double-click or run: python UI_TraTA_app4.py
# Requires: Pillow, matplotlib, and your original TravelTimeAnalysis.py next to this file.

import os, sys, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# keep UI headless-friendly
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None  # suppress any accidental plt.show()

# --- Stable output folder setup ---
# If running as a frozen .exe, APP_ROOT = folder containing the exe
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    APP_ROOT = Path(sys.executable).parent
else:
    APP_ROOT = Path(__file__).resolve().parent

# Always use the same "maps" directory next to the app
MAPS_DIR = APP_ROOT / "maps"
MAPS_DIR.mkdir(exist_ok=True)

# Optional but recommended: make this the working directory
os.chdir(APP_ROOT)


try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Use progress-enabled wrapper
import TravelTimeAnalysis2 as TTA2

APP_TITLE = 'Travel Time Map – UI (Progress + Gallery)'
DEFAULT_POIS = 'NUS University Town, Singapore Botanic Gardens, HortPark'
LAYER_ORDER = ['drive', 'bike', 'walk']   # bottom -> top

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry('980x780')

        # ---- Top form
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill='x', anchor='nw')

        ttk.Label(frm, text='Project Name (for output filenames):').grid(row=0, column=0, sticky='w')
        self.var_project = tk.StringVar(value='travel_time_session')
        ttk.Entry(frm, textvariable=self.var_project, width=40).grid(row=0, column=1, sticky='w', padx=8)

        ttk.Label(frm, text='Target POIs (3 names, separated by commas):').grid(row=1, column=0, sticky='w', pady=(8,0))
        self.var_pois = tk.StringVar(value=DEFAULT_POIS)
        ttk.Entry(frm, textvariable=self.var_pois, width=80).grid(row=1, column=1, sticky='w', padx=8, pady=(8,0))

        ttk.Label(frm, text='Travel Mode(s):').grid(row=2, column=0, sticky='nw', pady=(8,0))
        modes_frame = ttk.Frame(frm)
        modes_frame.grid(row=2, column=1, sticky='w', padx=8, pady=(8,0))

        self.var_walk = tk.BooleanVar(value=True)
        self.var_bike = tk.BooleanVar(value=False)
        self.var_drive = tk.BooleanVar(value=False)

        ttk.Checkbutton(modes_frame, text='Walk', variable=self.var_walk).pack(side='left', padx=(0,10))
        ttk.Checkbutton(modes_frame, text='Bike', variable=self.var_bike).pack(side='left', padx=(0,10))
        ttk.Checkbutton(modes_frame, text='Drive', variable=self.var_drive).pack(side='left', padx=(0,10))

        quick = ttk.Frame(modes_frame)
        quick.pack(side='left', padx=10)
        ttk.Button(quick, text='All', command=lambda: self.set_modes(True, True, True)).pack(side='left')
        ttk.Button(quick, text='None', command=lambda: self.set_modes(False, False, False)).pack(side='left', padx=(6,0))

        ttk.Label(frm, text='Departure Time Budget (minutes):').grid(row=3, column=0, sticky='w', pady=(8,0))
        self.var_minutes = tk.StringVar(value='15')
        ttk.Spinbox(frm, from_=1, to=240, textvariable=self.var_minutes, width=10).grid(
            row=3, column=1, sticky='w', padx=8, pady=(8,0)
        )

        # Run + folder
        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, sticky='w', pady=12)
        self.btn_run = ttk.Button(btns, text='Run', command=self.on_run_clicked)
        self.btn_run.pack(side='left')
        ttk.Button(btns, text='Open Maps Folder', command=self.open_maps_folder).pack(side='left', padx=8)

        # ---- Status + progress
        self.var_status = tk.StringVar(value='Ready.')
        ttk.Label(self, textvariable=self.var_status, foreground='#555').pack(fill='x', anchor='w', padx=12, pady=(6,0))

        barrow = ttk.Frame(self, padding=(12, 4))
        barrow.pack(fill='x', anchor='w')
        self.prog = ttk.Progressbar(barrow, mode='determinate', maximum=100)
        self.prog.pack(side='left', fill='x', expand=True)
        self.var_pmsg = tk.StringVar(value='')
        ttk.Label(barrow, textvariable=self.var_pmsg, width=28, anchor='e').pack(side='left', padx=(8,0))

        # ---- Gallery controls
        gal = ttk.Frame(self, padding=(12, 0))
        gal.pack(fill='x', anchor='w')
        self.btn_prev = ttk.Button(gal, text='◀ Prev', command=self.show_prev, state='disabled')
        self.btn_prev.pack(side='left')
        self.btn_next = ttk.Button(gal, text='Next ▶', command=self.show_next, state='disabled')
        self.btn_next.pack(side='left', padx=8)
        ttk.Label(gal, text='Autoplay (sec):').pack(side='left', padx=(16, 4))
        self.var_interval = tk.StringVar(value='2')
        ttk.Spinbox(gal, from_=1, to=30, textvariable=self.var_interval, width=4).pack(side='left')
        self.btn_play = ttk.Button(gal, text='▶ Play', command=self.toggle_play, state='disabled')
        self.btn_play.pack(side='left', padx=8)
        self.lbl_counter = ttk.Label(gal, text='')
        self.lbl_counter.pack(side='left', padx=(16, 0))

        # ---- Image preview
        self.preview = ttk.Label(self)
        self.preview.pack(fill='both', expand=True, padx=12, pady=12)
        self._img_ref = None

        # gallery state
        self.gallery = []
        self.idx = -1
        self.autoplay = False
        self.after_id = None

        # keyboard shortcuts
        self.bind('<Left>', lambda e: self.show_prev())
        self.bind('<Right>', lambda e: self.show_next())
        self.bind('<space>', lambda e: self.toggle_play())

    # ------------- helpers -------------
    def set_modes(self, w, b, d):
        self.var_walk.set(w); self.var_bike.set(b); self.var_drive.set(d)

    def validate_inputs(self):
        project = (self.var_project.get() or '').strip() or 'travel_time_session'
        pois_raw = (self.var_pois.get() or '').strip()
        minutes_s = (self.var_minutes.get() or '').strip()

        if not pois_raw:
            raise ValueError('Please enter three POI names separated by commas.')
        pois = [p.strip() for p in pois_raw.split(',') if p.strip()]
        if len(pois) < 3:
            raise ValueError('Exactly three POIs are required. Provide 3 names separated by commas.')
        pois = pois[:3]

        try:
            minutes = int(minutes_s)
            if minutes <= 0:
                raise ValueError
        except Exception:
            raise ValueError('Departure time must be a positive integer (minutes).')

        picked = []
        if self.var_walk.get(): picked.append('walk')
        if self.var_bike.get(): picked.append('bike')
        if self.var_drive.get(): picked.append('drive')
        if not picked:
            raise ValueError('Select at least one mode.')

        modes = [m for m in LAYER_ORDER if m in picked]  # enforce bottom -> top order
        return project, pois, modes, minutes

    def set_status(self, msg):
        self.var_status.set(msg)
        self.update_idletasks()

    def set_progress(self, pct: int, msg: str = ''):
        def _do():
            self.prog['value'] = max(0, min(100, int(pct)))
            if msg:
                self.var_pmsg.set(msg)
        self.after(0, _do)

    def _progress_cb(self, pct: int, msg: str = ''):
        self.set_progress(pct, msg)

    # ------------- run & gallery -------------
    def on_run_clicked(self):
        try:
            project, pois, modes, minutes = self.validate_inputs()
        except Exception as e:
            messagebox.showerror('Invalid input', str(e))
            return

        self.btn_run.config(state='disabled')
        self.disable_gallery_controls()
        self.set_status(f'Running: {project} | modes={modes} | {minutes} min')
        self.prog['value'] = 0
        self.var_pmsg.set('Starting...')

        threading.Thread(
            target=self._run_pipeline_and_load_gallery,
            args=(project, pois, modes, minutes),
            daemon=True
        ).start()

    def _run_pipeline_and_load_gallery(self, project, pois, modes, minutes):
        try:
            TTA2.run_pipeline(
                location_name=project,
                poi_inputs=pois,
                modes=modes,
                durations_min=[minutes],
                save_figs=True,
                export_vectors=False,
                progress=self._progress_cb,
            )
            self.set_progress(100, 'Done')
            self.set_status('Finished generating maps. Loading gallery...')
            self.load_gallery(project, minutes)
            if len(self.gallery) > 1:
                self.autoplay = True
                self.btn_play.config(text='⏸ Pause')
                self.schedule_autonext()
            else:
                self.autoplay = False
                self.btn_play.config(text='▶ Play')
        except Exception as e:
            tb = traceback.format_exc()
            self.set_status('Error occurred.')
            self.set_progress(0, 'Error')
            messagebox.showerror('Error during run', f'{e}\n\n{tb}')
        finally:
            self.btn_run.config(state='normal')

    def load_gallery(self, project, minutes):
        maps_dir = MAPS_DIR
        maps_dir.mkdir(exist_ok=True)
        safe_project = TTA2.SAFE(project)
        pattern = f'{safe_project}_*_{int(minutes)}min*.png'
        files = sorted(maps_dir.glob(pattern), key=os.path.getmtime)
        self.gallery = files
        self.idx = 0 if self.gallery else -1
        self.update_counter()
        if self.gallery:
            self.show_image(self.gallery[self.idx])
            self.enable_gallery_controls()
            self.set_status(f'Loaded {len(self.gallery)} figure(s).')
        else:
            self.preview.configure(image='', text='No figures found.', anchor='center')
            self.disable_gallery_controls()
            self.set_status('No figures found for this project/minutes.')

    def show_image(self, path: Path):
        if Image is None or ImageTk is None:
            return
        img = Image.open(path).convert('RGB')
        max_w = max(400, self.preview.winfo_width() or 900)
        max_h = max(300, self.preview.winfo_height() or 600)
        img.thumbnail((max_w, max_h))
        self._img_ref = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._img_ref, text='')
        self.update_counter()

    def show_prev(self):
        if not self.gallery: return
        self.idx = (self.idx - 1) % len(self.gallery)
        self.show_image(self.gallery[self.idx])

    def show_next(self):
        if not self.gallery: return
        self.idx = (self.idx + 1) % len(self.gallery)
        self.show_image(self.gallery[self.idx])

    def update_counter(self):
        if self.gallery and self.idx >= 0:
            self.lbl_counter.config(text=f'{self.idx+1}/{len(self.gallery)}  –  {self.gallery[self.idx].name}')
        else:
            self.lbl_counter.config(text='')

    def toggle_play(self):
        if not self.gallery: return
        self.autoplay = not self.autoplay
        self.btn_play.config(text='⏸ Pause' if self.autoplay else '▶ Play')
        if self.autoplay:
            self.schedule_autonext()
        else:
            self.cancel_autonext()

    def schedule_autonext(self):
        self.cancel_autonext()
        try:
            interval_ms = max(1, int(self.var_interval.get())) * 1000
        except Exception:
            interval_ms = 2000
            self.var_interval.set('2')
        self.after_id = self.after(interval_ms, self._autonext_step)

    def _autonext_step(self):
        if not self.autoplay: return
        self.show_next()
        self.schedule_autonext()

    def cancel_autonext(self):
        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def enable_gallery_controls(self):
        self.btn_prev.config(state='normal')
        self.btn_next.config(state='normal')
        self.btn_play.config(state='normal')

    def disable_gallery_controls(self):
        self.btn_prev.config(state='disabled')
        self.btn_next.config(state='disabled')
        self.btn_play.config(state='disabled')
        self.cancel_autonext()

    def open_maps_folder(self):
        maps_dir = MAPS_DIR
        maps_dir.mkdir(exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(str(maps_dir))
            elif sys.platform == 'darwin':
                os.system(f'open "{maps_dir}"')
            else:
                os.system(f'xdg-open "{maps_dir}"')
        except Exception:
            messagebox.showinfo('Open Folder', f'Maps folder: {maps_dir}')

if __name__ == '__main__':
    App().mainloop()