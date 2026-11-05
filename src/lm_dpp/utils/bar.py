from typing import Optional
from datetime import timedelta
from operator import length_hint
from rich.progress import BarColumn, Progress, ProgressColumn, Task, Text, Column, filesize, get_console

__all__ = ["bar"]

# TODO: nested bars, multiple succesive bars should not alter the normal print


class FractionColumn(ProgressColumn):
    """Renders completed/total, e.g. '0.5/2.3 G'."""

    def __init__(self, unit_scale=False, unit_divisor=1000):
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Calculate common unit for completed and total."""
        completed = int(task.completed)
        total = int(task.total)
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                total,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(total, [""], 1)
        precision = 0 if unit == 1 else 1
        return Text(
            f"{completed / unit:,.{precision}f}/{total / unit:,.{precision}f} {suffix}", style="progress.download"
        )


class RateColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, unit="", unit_scale=False, unit_divisor=1000):
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        super().__init__()

    def render(self, task):
        """Show data transfer speed."""
        speed = task.speed
        if speed is None:
            bar_instance = getattr(task, "_bar_instance", None)
            if bar_instance and bar_instance.manual_speed is not None:
                manual_speed = bar_instance.manual_speed
                # debug_info = f"manual:{manual_speed:.3f}"
                if self.unit_scale:
                    unit, suffix = filesize.pick_unit_and_suffix(
                        manual_speed,
                        ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                        self.unit_divisor,
                    )
                else:
                    unit, suffix = filesize.pick_unit_and_suffix(manual_speed, [""], 1)
                precision = 2
                return Text(
                    f"{manual_speed / unit:,.{precision}f} {suffix}{self.unit}/s", style="progress.data.speed"
                ) + Text("]", style="default")
            else:
                # debug_info = f"elapsed:{task.elapsed:.2f}s,completed:{task.completed}" if task.elapsed else "no_elapsed"
                return Text(f"? {self.unit}/s", style="progress.data.speed") + Text("]", style="default")
        if self.unit_scale:
            unit, suffix = filesize.pick_unit_and_suffix(
                speed,
                ["", "K", "M", "G", "T", "P", "E", "Z", "Y"],
                self.unit_divisor,
            )
        else:
            unit, suffix = filesize.pick_unit_and_suffix(speed, [""], 1)
        precision = 2
        return Text(f"{speed / unit:,.{precision}f} {suffix}{self.unit}/s", style="progress.data.speed") + Text(
            "]", style="default"
        )


class TimeElapsedRemainingColumn(ProgressColumn):
    """Renders elapsed and estimated remaining time.

    Args:
        compact (bool, optional): Render MM:SS when time remaining is less than an hour. Defaults to False.
        elapsed_when_finished (bool, optional): Render time elapsed when the task is finished. Defaults to False.
    """

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(
        self,
        compact: bool = True,
        elapsed_when_finished: bool = False,
        table_column: Optional[Column] = None,
    ):
        self.compact = compact
        self.elapsed_when_finished = elapsed_when_finished
        super().__init__(table_column=table_column)

    def render(self, task: "Task") -> Text:
        """Show time elapsed and remaining."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            elapsed_txt = Text("[") + Text("-:--:--", style="progress.elapsed")
        else:
            delta = timedelta(seconds=max(0, int(elapsed)))
            elapsed_txt = Text("[") + Text(str(delta), style="progress.elapsed")

        if self.elapsed_when_finished and task.finished:
            task_time = task.finished_time
            style = "progress.elapsed"
        else:
            task_time = task.time_remaining
            style = "progress.remaining"

        if task.total is None:
            return elapsed_txt + Text("", style=style)

        elapsed_txt += Text(" < ")

        if task_time is None:
            bar_instance = getattr(task, "_bar_instance", None)
            if bar_instance and bar_instance.manual_time_remaining is not None:
                manual_time_remaining = bar_instance.manual_time_remaining

                minutes, seconds = divmod(int(manual_time_remaining), 60)
                hours, minutes = divmod(minutes, 60)

                if self.compact and not hours:
                    formatted = f"{minutes:02d}:{seconds:02d}"
                else:
                    formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

                return elapsed_txt + Text(f"{formatted}", style=style) + Text(",")
            else:
                # debug_info = f"speed:{task.speed}" if hasattr(task, 'speed') else "no_speed"
                return elapsed_txt + Text(f"--:--" if self.compact else f"-:--:--", style=style) + Text(",")

        # Based on https://github.com/tqdm/tqdm/blob/master/tqdm/std.py
        minutes, seconds = divmod(int(task_time), 60)
        hours, minutes = divmod(minutes, 60)

        if self.compact and not hours:
            formatted = f"{minutes:02d}:{seconds:02d}"
        else:
            formatted = f"{hours:d}:{minutes:02d}:{seconds:02d}"

        return elapsed_txt + Text(formatted, style=style) + Text(",")


class bar:  # pragma: no cover
    def __init__(
        self, iterable, unit="it", unit_scale=False, leave=True, disable=False, desc=None, total=None, widget=None
    ):
        if type(iterable) is int:
            iterable = range(iterable)
        self.iterable = iterable
        self.unit = unit
        self.unit_scale = unit_scale
        self.leave = leave
        self.disable = disable
        self.desc = desc
        self.total = total or length_hint(iterable)
        self.manual_speed = None
        self.manual_time_remaining = None

        self.last_update_time = None
        self.smoothed_speed = None
        self.alpha = 0.3  # Smoothing factor: higher = more weight to recent updates

        if self.disable:
            return

        progress = (
            "[progress.description]{task.description}[progress.percentage]{task.percentage:>4.0f}%",
            BarColumn(bar_width=25),
            FractionColumn(),
            TimeElapsedRemainingColumn(),
            RateColumn(unit=unit, unit_scale=unit_scale),
        )
        self.redirect = not get_console().is_jupyter
        self._prog = Progress(
            *progress,
            transient=not self.leave,
            auto_refresh=True,
            redirect_stderr=self.redirect,
            redirect_stdout=self.redirect,
            speed_estimate_period=3,
        )
        if widget:
            self._prog.live.ipy_widget = widget
        elif not self.redirect:  # Jupyter
            try:
                from ipywidgets import Output, VBox
                from IPython.display import display

                live_w = Output()
                self._prog.live.ipy_widget = live_w
                # self.console_w = Output()
                # cont = VBox([live_w, self.console_w])
                # display(cont)
                display(live_w)
                # self.console_w.__enter__()
            except ImportError:
                pass

        self._prog.live.start(refresh=True)
        self._task_id = self._prog.add_task(self.desc or "", total=self.total)
        self._prog.tasks[self._task_id]._bar_instance = self

    def __iter__(self):
        if self.disable:
            yield from self.iterable
            return
        try:
            for obj in self.iterable:
                yield obj
                self._prog.advance(self._task_id, advance=1)
                self._update_manual_calculations()
        finally:
            self.close()

    def _update_manual_calculations(self):
        """Update manual speed and time remaining with exponential smoothing"""
        import time

        task = self._prog.tasks[self._task_id]
        current_time = time.time()

        if task.elapsed and task.elapsed > 0 and task.completed > 0:
            # Calculate instantaneous speed if we have a previous update
            if self.last_update_time is not None and task.completed > 1:
                time_delta = current_time - self.last_update_time
                if time_delta > 0:
                    instant_speed = 1.0 / time_delta  # 1 item per time_delta

                    # Exponential smoothing: new_speed = alpha * instant + (1-alpha) * old_speed
                    if self.smoothed_speed is None:
                        self.smoothed_speed = instant_speed
                    else:
                        self.smoothed_speed = self.alpha * instant_speed + (1 - self.alpha) * self.smoothed_speed

                    self.manual_speed = self.smoothed_speed
                else:
                    # Fallback to average if time_delta is too small
                    self.manual_speed = task.completed / task.elapsed
            else:
                # First few items: use average
                self.manual_speed = task.completed / task.elapsed
                self.smoothed_speed = self.manual_speed

            # Calculate time remaining
            if task.total and task.total > task.completed and self.manual_speed > 0:
                remaining_items = task.total - task.completed
                self.manual_time_remaining = remaining_items / self.manual_speed
            else:
                self.manual_time_remaining = None
        else:
            self.manual_speed = None
            self.manual_time_remaining = None

        self.last_update_time = current_time

    def close(self):
        if self.disable:
            return
        if not self.redirect:  # Jupyter
            # pass
            self._prog.console.quiet = True  # This will suppress an empty cell when stopping
        self._prog.live.stop()
        self._prog.console.quiet = False
        # self.console_w.__exit__(None, None, None)

    def reset(self, total=None):
        self._prog.reset(total=total)
