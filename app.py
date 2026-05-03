"""
app.py — GUI launcher for Panoramic Image Stitching
-----------------------------------------------------
Run with:
    python app.py

A simple Tkinter window lets you:
  1. Add images via a file browser (drag to reorder not needed — just add in order)
  2. Remove or reorder images in the list
  3. Click "Stitch Panorama" to run the algorithm
  4. View the result inline and save it

Requirements: same as the project (numpy, opencv-contrib-python, imutils)
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import imutils
import numpy as np
from PIL import Image, ImageTk   # pip install Pillow

from panorama import Panaroma


# ------------------------------------------------------------------
# Backend helpers
# ------------------------------------------------------------------

TARGET_WIDTH = 400
TARGET_HEIGHT = 400
OUTPUT_DIR = "output"


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def resize_image(img):
    img = imutils.resize(img, width=TARGET_WIDTH)
    img = imutils.resize(img, height=TARGET_HEIGHT)
    return img


def cv2_to_tk(cv_img, max_w=900, max_h=400):
    """Convert an OpenCV BGR image to a Tkinter-compatible PhotoImage, scaled to fit."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pil_img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil_img)


def run_stitching(paths, progress_cb, done_cb, error_cb):
    """
    Run the stitching pipeline in a background thread.
    Callbacks are called from the thread — callers must use .after() to update the UI.
    """
    try:
        progress_cb("Loading images…")
        images = [resize_image(load_image(p)) for p in paths]

        panorama = Panaroma()
        n = len(images)
        progress_cb(f"Stitching {n} images…")

        if n == 2:
            output = panorama.image_stitch([images[0], images[1]], match_status=True)
        else:
            output = panorama.image_stitch([images[n - 2], images[n - 1]], match_status=True)
            if output is None:
                raise RuntimeError("First pair failed. Not enough overlap?")
            result, matched = output
            for i in range(n - 2):
                idx = n - i - 3
                progress_cb(f"Adding image {idx + 1} of {n}…")
                output = panorama.image_stitch([images[idx], result], match_status=True)
                if output is None:
                    raise RuntimeError(f"Failed adding image {idx + 1}. Check order/overlap.")
                result, matched = output

        if output is None:
            raise RuntimeError("Stitching produced no result.")

        result, matched = output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "panorama_image.jpg"), result)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "matched_points.jpg"), matched)

        done_cb(result, matched)

    except Exception as e:
        error_cb(str(e))


# ------------------------------------------------------------------
# GUI
# ------------------------------------------------------------------

class PanoramaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Panoramic Image Stitching")
        self.resizable(True, True)
        self.configure(bg="#1a1a2e")
        self.geometry("960x720")

        self._image_paths = []
        self._tk_images = []   # keep references alive

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        DARK  = "#1a1a2e"
        PANEL = "#16213e"
        ACCENT = "#0f3460"
        HIGHLIGHT = "#e94560"
        TEXT  = "#eaeaea"
        MUTED = "#888"

        self.configure(bg=DARK)

        # ── Header ──────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=ACCENT, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🖼  Panoramic Image Stitcher",
                 font=("Courier New", 18, "bold"),
                 bg=ACCENT, fg=TEXT).pack()
        tk.Label(hdr, text="SIFT · RANSAC · Homography · Warp Perspective",
                 font=("Courier New", 9),
                 bg=ACCENT, fg=MUTED).pack()

        # ── Main body ───────────────────────────────────────────────
        body = tk.Frame(self, bg=DARK)
        body.pack(fill="both", expand=True, padx=18, pady=12)

        # Left panel: image list controls
        left = tk.Frame(body, bg=PANEL, bd=0, relief="flat", width=280)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(left, text="Images  (left → right order)",
                 font=("Courier New", 10, "bold"),
                 bg=PANEL, fg=HIGHLIGHT).pack(pady=(12, 4))

        list_frame = tk.Frame(left, bg=PANEL)
        list_frame.pack(fill="both", expand=True, padx=8)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(
            list_frame, bg="#0d1b2a", fg=TEXT,
            selectbackground=HIGHLIGHT, selectforeground=TEXT,
            font=("Courier New", 9), relief="flat", bd=0,
            yscrollcommand=scrollbar.set, activestyle="none"
        )
        self.listbox.pack(fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        # Buttons row
        btn_row = tk.Frame(left, bg=PANEL)
        btn_row.pack(fill="x", padx=8, pady=8)

        for text, cmd in [
            ("+ Add",    self._add_images),
            ("↑ Up",     self._move_up),
            ("↓ Down",   self._move_down),
            ("✕ Remove", self._remove_selected),
        ]:
            tk.Button(
                btn_row, text=text, command=cmd,
                bg=ACCENT, fg=TEXT, activebackground=HIGHLIGHT,
                font=("Courier New", 9), relief="flat", padx=6, pady=4, cursor="hand2"
            ).pack(side="left", padx=2)

        # Stitch button
        self.stitch_btn = tk.Button(
            left, text="▶  Stitch Panorama", command=self._start_stitching,
            bg=HIGHLIGHT, fg="white", activebackground="#c73652",
            font=("Courier New", 11, "bold"), relief="flat",
            pady=10, cursor="hand2"
        )
        self.stitch_btn.pack(fill="x", padx=8, pady=(4, 12))

        # Right panel: result display
        right = tk.Frame(body, bg=DARK)
        right.pack(side="left", fill="both", expand=True)

        tab_bar = ttk.Notebook(right)
        tab_bar.pack(fill="both", expand=True)

        self.panorama_tab = tk.Frame(tab_bar, bg="#0d1b2a")
        self.matches_tab  = tk.Frame(tab_bar, bg="#0d1b2a")
        tab_bar.add(self.panorama_tab, text="  Panorama  ")
        tab_bar.add(self.matches_tab,  text="  Keypoint Matches  ")

        self.panorama_label = tk.Label(
            self.panorama_tab,
            text="Add images on the left, then click\n▶  Stitch Panorama",
            font=("Courier New", 12), bg="#0d1b2a", fg=MUTED
        )
        self.panorama_label.pack(expand=True)

        self.matches_label = tk.Label(
            self.matches_tab,
            text="Keypoint matches will appear here after stitching.",
            font=("Courier New", 12), bg="#0d1b2a", fg=MUTED
        )
        self.matches_label.pack(expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Add at least 2 images to begin.")
        status_bar = tk.Label(
            self, textvariable=self.status_var,
            font=("Courier New", 9), bg=ACCENT, fg=TEXT,
            anchor="w", padx=12, pady=5
        )
        status_bar.pack(fill="x", side="bottom")

    # ------------------------------------------------------------------
    # List management
    # ------------------------------------------------------------------

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select images (in order)",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        for p in paths:
            self._image_paths.append(p)
            self.listbox.insert("end", os.path.basename(p))
        self._set_status(f"{len(self._image_paths)} image(s) loaded.")

    def _move_up(self):
        sel = self.listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self._image_paths[i], self._image_paths[i - 1] = self._image_paths[i - 1], self._image_paths[i]
        text = self.listbox.get(i)
        self.listbox.delete(i)
        self.listbox.insert(i - 1, text)
        self.listbox.select_set(i - 1)

    def _move_down(self):
        sel = self.listbox.curselection()
        if not sel or sel[0] == len(self._image_paths) - 1:
            return
        i = sel[0]
        self._image_paths[i], self._image_paths[i + 1] = self._image_paths[i + 1], self._image_paths[i]
        text = self.listbox.get(i)
        self.listbox.delete(i)
        self.listbox.insert(i + 1, text)
        self.listbox.select_set(i + 1)

    def _remove_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self.listbox.delete(i)
        del self._image_paths[i]
        self._set_status(f"{len(self._image_paths)} image(s) loaded.")

    # ------------------------------------------------------------------
    # Stitching
    # ------------------------------------------------------------------

    def _start_stitching(self):
        if len(self._image_paths) < 2:
            messagebox.showwarning("Not enough images", "Please add at least 2 images.")
            return

        self.stitch_btn.config(state="disabled", text="⏳  Stitching…")
        self._set_status("Stitching in progress…")

        def progress_cb(msg):
            self.after(0, lambda: self._set_status(msg))

        def done_cb(result, matched):
            self.after(0, lambda: self._on_done(result, matched))

        def error_cb(msg):
            self.after(0, lambda: self._on_error(msg))

        thread = threading.Thread(
            target=run_stitching,
            args=(self._image_paths, progress_cb, done_cb, error_cb),
            daemon=True
        )
        thread.start()

    def _on_done(self, result, matched):
        self.stitch_btn.config(state="normal", text="▶  Stitch Panorama")
        self._set_status(
            f"Done! Saved to output/panorama_image.jpg  &  output/matched_points.jpg"
        )

        # Show panorama
        tk_panorama = cv2_to_tk(result, max_w=860, max_h=360)
        self._tk_images.append(tk_panorama)
        self.panorama_label.config(image=tk_panorama, text="")

        # Show matches
        tk_matches = cv2_to_tk(matched, max_w=860, max_h=360)
        self._tk_images.append(tk_matches)
        self.matches_label.config(image=tk_matches, text="")

        messagebox.showinfo(
            "Stitching complete",
            "Panorama saved to:\n  output/panorama_image.jpg\n\nMatch visualization:\n  output/matched_points.jpg"
        )

    def _on_error(self, msg):
        self.stitch_btn.config(state="normal", text="▶  Stitch Panorama")
        self._set_status(f"Error: {msg}")
        messagebox.showerror("Stitching failed", msg)

    def _set_status(self, msg):
        self.status_var.set(f"  {msg}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Check Pillow is installed (needed for image display in the GUI)
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("[ERROR] Pillow is required for the GUI. Install it with:\n  pip install Pillow")
        sys.exit(1)

    app = PanoramaApp()
    app.mainloop()
