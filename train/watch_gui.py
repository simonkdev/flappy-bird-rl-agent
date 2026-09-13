"""Tkinter checkpoint player for observing a PPO policy in real time."""

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from src import config


FRAME_SCALE = 10
STACK_SCALE = 2
POLL_MILLISECONDS = 30


def ppm_image_data(pixels, width, height):
    grayscale = bytes(pixels)
    rgb = b"".join(bytes((value, value, value)) for value in grayscale)
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return header + rgb


def photo_from_pixels(root, pixels, width, height, scale):
    image = tk.PhotoImage(
        master=root,
        data=ppm_image_data(pixels, width, height),
        format="PPM",
    )
    return image.zoom(scale, scale)


class PlaybackWorker(threading.Thread):
    def __init__(self, checkpoint_path, seed, stochastic, native_window, events):
        super().__init__(daemon=True)
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.stochastic = stochastic
        self.native_window = native_window
        self.events = events
        self.stop_requested = threading.Event()

    def emit(self, kind, **payload):
        self.events.put((kind, payload))

    def stop(self):
        self.stop_requested.set()

    def run(self):
        ppo = None
        try:
            import tensorflow as tf

            from src.ppo import PPO
            from train.training import create_checkpoint_managers, restore_checkpoint

            self.emit("loading", text="Creating environment and restoring checkpoint...")
            ppo = PPO(
                env_backend="subprocess",
                num_envs=1,
                show_game_window=self.native_window,
            )
            checkpoint, _ = create_checkpoint_managers(
                ppo,
                "/tmp/flappyrl-watch-gui",
                1,
            )
            epoch = restore_checkpoint(ppo, checkpoint, self.checkpoint_path)
            self.emit("ready", epoch=epoch)

            episode = 1
            seed = self.seed
            env = ppo.envs[0]
            while not self.stop_requested.is_set():
                result = env.reset_result(seed=seed)
                ppo.framestack_buffers[0].clear()
                self.emit("log", text=f"episode={episode} seed={seed} reset")

                for step in range(1, 10_001):
                    if self.stop_requested.is_set():
                        break
                    state = ppo.framestack(result.observation, step - 1, 0)
                    logits = ppo.actor(state[None, ...])
                    probabilities = tf.nn.softmax(logits)[0].numpy()
                    value = float(ppo.critic(state[None, ...])[0, 0].numpy())
                    action = ppo.decide(state)[0] if self.stochastic else int(
                        tf.argmax(logits[0], axis=0).numpy()
                    )
                    started = time.monotonic()
                    result = env.step_result(action)
                    self.emit(
                        "frame",
                        pixels=result.observation,
                        stack=[state[..., index].tobytes() for index in range(config.FRAME_STACK)],
                        score=result.score,
                    )
                    self.emit(
                        "log",
                        text=(
                            f"episode={episode} step={step} action={action} "
                            f"p(no_flap)={probabilities[0]:.3f} "
                            f"p(flap)={probabilities[1]:.3f} value={value:.2f} "
                            f"reward={result.reward:.2f} score={result.score} "
                            f"alive={int(result.alive)}"
                        ),
                    )
                    remaining = (env.ticks_per_step * env.dt) - (time.monotonic() - started)
                    if remaining > 0:
                        self.stop_requested.wait(remaining)
                    if result.terminated:
                        self.emit(
                            "log",
                            text=f"episode={episode} finished score={result.score} steps={step}",
                        )
                        break
                episode += 1
                seed += 1
        except Exception as error:
            self.emit("error", message=str(error))
        finally:
            if ppo is not None:
                ppo.close()
            self.emit("stopped")


class TensorFlowLoader(threading.Thread):
    def __init__(self, events):
        super().__init__(daemon=True)
        self.events = events

    def run(self):
        try:
            import tensorflow  # noqa: F401
        except Exception as error:
            self.events.put(("startup_error", {"message": str(error)}))
        else:
            self.events.put(("runtime_ready", {}))


class WatchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flappy RL Checkpoint Watcher")
        self.root.minsize(1180, 800)
        self.events = queue.Queue()
        self.worker = None
        self.startup_loading = True
        self.loading_angle = 0
        self.game_image = None
        self.stack_images = [None for _ in range(config.FRAME_STACK)]

        self.checkpoint = tk.StringVar()
        self.seed = tk.StringVar(value=str(config.VALIDATION_SEED_START))
        self.mode = tk.StringVar(value="Deterministic")
        self.native_window = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="TensorFlow is starting...")
        self.score = tk.StringVar(value="Score: --")

        self.build_loading_screen()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(POLL_MILLISECONDS, self.drain_events)
        TensorFlowLoader(self.events).start()

    def build_loading_screen(self):
        self.loading_frame = ttk.Frame(self.root)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center")
        self.loading_canvas = tk.Canvas(
            self.loading_frame,
            width=64,
            height=64,
            highlightthickness=0,
        )
        self.loading_canvas.grid(row=0, column=0, pady=(0, 12))
        ttk.Label(self.loading_frame, textvariable=self.status).grid(row=1, column=0)
        self.animate_loading_circle()

    def animate_loading_circle(self):
        if not self.startup_loading:
            return
        self.loading_canvas.delete("all")
        self.loading_canvas.create_arc(
            8,
            8,
            56,
            56,
            start=self.loading_angle,
            extent=280,
            style="arc",
            outline="#278a5b",
            width=5,
        )
        self.loading_angle = (self.loading_angle + 18) % 360
        self.root.after(40, self.animate_loading_circle)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        outer.rowconfigure(2, weight=1)

        controls = ttk.LabelFrame(outer, text="Playback Controls", padding=10)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Checkpoint").grid(row=0, column=0, sticky="w")
        self.checkpoint_entry = ttk.Entry(controls, textvariable=self.checkpoint)
        self.checkpoint_entry.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Browse", command=self.browse_checkpoint).grid(row=0, column=2)
        ttk.Label(controls, text="Seed").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.seed, width=16).grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )
        ttk.Combobox(
            controls,
            textvariable=self.mode,
            values=("Deterministic", "Stochastic"),
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky="w", padx=(145, 0), pady=(8, 0))
        ttk.Checkbutton(
            controls,
            text="Open native C++ game window",
            variable=self.native_window,
        ).grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.play_button = ttk.Button(controls, text="Play", command=self.play)
        self.play_button.grid(row=0, column=3, rowspan=2, padx=(10, 0), sticky="ns")
        self.stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self.stop_button.grid(row=0, column=4, rowspan=2, padx=(6, 0), sticky="ns")
        ttk.Label(controls, textvariable=self.score, font=("TkDefaultFont", 12, "bold")).grid(
            row=0,
            column=5,
            rowspan=2,
            padx=(14, 0),
        )

        views = ttk.Frame(outer)
        views.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(0, weight=1)

        game = ttk.LabelFrame(views, text="Game View", padding=8)
        game.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        game.grid_propagate(False)
        game.configure(width=500, height=450)
        self.game_label = tk.Label(game, bg="#111111")

        debug = ttk.LabelFrame(views, text="Debug Frame Stack", padding=8)
        debug.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        debug.grid_propagate(False)
        debug.configure(width=500, height=450)
        self.debug_frame = debug
        self.stack_labels = []

        telemetry = ttk.LabelFrame(outer, text="Telemetry", padding=8)
        telemetry.grid(row=2, column=0, sticky="nsew")
        telemetry.columnconfigure(0, weight=1)
        telemetry.rowconfigure(0, weight=1)
        self.terminal = tk.Text(
            telemetry,
            height=15,
            bg="#101510",
            fg="#d7f5d0",
            insertbackground="#d7f5d0",
            font=("TkFixedFont", 10),
            state="disabled",
            wrap="none",
        )
        scrollbar = ttk.Scrollbar(telemetry, command=self.terminal.yview)
        self.terminal.configure(yscrollcommand=scrollbar.set)
        self.terminal.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        ttk.Label(outer, textvariable=self.status).grid(row=3, column=0, sticky="w", pady=(8, 0))

    def browse_checkpoint(self):
        selected = filedialog.askopenfilename(
            title="Select checkpoint prefix or index file",
            filetypes=(("TensorFlow checkpoint index", "*.index"), ("All files", "*")),
        )
        if selected:
            self.checkpoint.set(selected[:-6] if selected.endswith(".index") else selected)

    def play(self):
        path = self.checkpoint.get().strip()
        if not path:
            self.status.set("A checkpoint path is required.")
            return
        if not Path(f"{path}.index").exists() and not Path(path).is_dir():
            self.status.set("Checkpoint prefix or directory was not found.")
            return
        try:
            seed = int(self.seed.get())
            if seed < 0:
                raise ValueError
        except ValueError:
            self.status.set("Seed must be a non-negative integer.")
            return

        self.append_log(f"loading checkpoint={path}")
        self.status.set("Loading checkpoint...")
        self.play_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker = PlaybackWorker(
            path,
            seed,
            self.mode.get() == "Stochastic",
            self.native_window.get(),
            self.events,
        )
        self.worker.start()

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.status.set("Stopping playback...")
            self.stop_button.configure(state="disabled")

    def append_log(self, text):
        self.terminal.configure(state="normal")
        self.terminal.insert("end", f"{text}\n")
        self.terminal.see("end")
        self.terminal.configure(state="disabled")

    def update_frame(self, payload):
        if not self.stack_labels:
            self.create_stack_labels()
        self.game_image = photo_from_pixels(
            self.root,
            payload["pixels"],
            config.OBS_WIDTH,
            config.OBS_HEIGHT,
            FRAME_SCALE,
        )
        self.game_label.configure(image=self.game_image)
        self.game_label.place(relx=0.5, rely=0.5, anchor="center")
        self.score.set(f"Score: {payload['score']}")
        for index, pixels in enumerate(payload["stack"]):
            image = photo_from_pixels(
                self.root,
                pixels,
                config.OBS_WIDTH,
                config.OBS_HEIGHT,
                STACK_SCALE,
            )
            self.stack_images[index] = image
            self.stack_labels[index].configure(image=image)

    def create_stack_labels(self):
        for index in range(config.FRAME_STACK):
            label = tk.Label(self.debug_frame, bg="#111111")
            label.grid(row=0, column=index, padx=3, pady=16)
            ttk.Label(
                self.debug_frame,
                text=f"t-{config.FRAME_STACK - index - 1}",
            ).grid(row=1, column=index)
            self.stack_labels.append(label)

    def drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "runtime_ready":
                    self.startup_loading = False
                    self.loading_frame.destroy()
                    self.build_ui()
                    self.status.set("TensorFlow ready. Select a checkpoint and press Play.")
                elif kind == "startup_error":
                    self.startup_loading = False
                    self.status.set("TensorFlow startup failed.")
                    self.loading_canvas.delete("all")
                    self.loading_canvas.create_text(
                        32,
                        32,
                        text="!",
                        fill="#aa2222",
                        font=("TkDefaultFont", 28),
                    )
                    self.append_loading_error(payload["message"])
                elif kind == "ready":
                    self.status.set(f"Playing checkpoint epoch {payload['epoch']}.")
                    self.append_log(f"checkpoint restored at epoch={payload['epoch']}")
                elif kind == "loading":
                    self.status.set(payload["text"])
                    self.append_log(payload["text"])
                elif kind == "frame":
                    self.update_frame(payload)
                elif kind == "log":
                    self.append_log(payload["text"])
                elif kind == "error":
                    self.status.set("Playback failed.")
                    self.append_log(f"ERROR: {payload['message']}")
                elif kind == "stopped":
                    self.worker = None
                    self.play_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if self.status.get() != "Playback failed.":
                        self.status.set("Playback stopped.")
        except queue.Empty:
            pass
        self.root.after(POLL_MILLISECONDS, self.drain_events)

    def append_loading_error(self, message):
        ttk.Label(
            self.loading_frame,
            text=message,
            foreground="#aa2222",
            justify="center",
            wraplength=700,
        ).grid(row=2, column=0, pady=(10, 0))

    def on_close(self):
        self.stop()
        self.root.after(50, self.wait_for_worker)

    def wait_for_worker(self):
        if self.worker is not None and self.worker.is_alive():
            self.root.after(50, self.wait_for_worker)
            return
        self.root.destroy()


def main():
    root = tk.Tk()
    WatchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
