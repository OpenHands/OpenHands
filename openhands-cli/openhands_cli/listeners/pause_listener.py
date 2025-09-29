import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from openhands.sdk import Conversation
from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.input import Input, create_input
from prompt_toolkit.keys import Keys


class PauseListener(threading.Thread):
    """Background key listener that triggers pause on Ctrl-P and immediate termination on double Ctrl-C.

    Starts and stops around agent run() loops to avoid interfering with user prompts.
    """

    def __init__(
        self,
        on_pause: Callable,
        on_terminate: Callable | None = None,  # called on double Ctrl+C
        input_source: Input | None = None,  # used to pipe inputs for unit tests
        double_ctrl_c_timeout: float = 2.0,  # seconds to wait for second Ctrl+C
    ):
        super().__init__(daemon=True)
        self.on_pause = on_pause
        self.on_terminate = on_terminate
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._terminate_event = threading.Event()
        self._input = input_source or create_input()
        self._double_ctrl_c_timeout = double_ctrl_c_timeout
        self._last_ctrl_c_time = None

    def _detect_pause_key_presses(self) -> tuple[bool, bool]:
        """Detect pause key presses and double Ctrl+C.
        
        Returns:
            tuple: (pause_detected, terminate_detected)
        """
        pause_detected = False
        terminate_detected = False

        for key_press in self._input.read_keys():
            if key_press.key == Keys.ControlP or key_press.key == Keys.ControlD:
                pause_detected = True
            elif key_press.key == Keys.ControlC:
                current_time = time.time()
                
                # Check if this is a double Ctrl+C
                if (self._last_ctrl_c_time is not None and 
                    current_time - self._last_ctrl_c_time <= self._double_ctrl_c_timeout):
                    terminate_detected = True
                    self._last_ctrl_c_time = None  # Reset
                else:
                    # First Ctrl+C or too much time has passed
                    self._last_ctrl_c_time = current_time
                    pause_detected = True

        return pause_detected, terminate_detected

    def _execute_pause(self) -> None:
        self._pause_event.set()  # Mark pause event occurred
        print_formatted_text(HTML(""))
        print_formatted_text(
            HTML("<gold>Pausing agent once step is completed... (Press Ctrl+C again to terminate immediately)</gold>")
        )
        try:
            self.on_pause()
        except Exception:
            pass

    def _execute_terminate(self) -> None:
        self._terminate_event.set()  # Mark terminate event occurred
        print_formatted_text(HTML(""))
        print_formatted_text(
            HTML("<red>Terminating agent immediately...</red>")
        )
        try:
            if self.on_terminate:
                self.on_terminate()
        except Exception:
            pass

    def run(self) -> None:
        try:
            with self._input.raw_mode():
                # User hasn't paused/terminated and pause listener hasn't been shut down
                while not (self.is_paused() or self.is_terminated() or self.is_stopped()):
                    pause_detected, terminate_detected = self._detect_pause_key_presses()
                    
                    if terminate_detected:
                        self._execute_terminate()
                        break
                    elif pause_detected:
                        self._execute_pause()
        finally:
            try:
                self._input.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def is_terminated(self) -> bool:
        return self._terminate_event.is_set()


@contextmanager
def pause_listener(
    conversation: Conversation, 
    on_terminate: Callable | None = None,
    input_source: Input | None = None
) -> Iterator[PauseListener]:
    """Ensure PauseListener always starts/stops cleanly."""
    listener = PauseListener(
        on_pause=conversation.pause, 
        on_terminate=on_terminate,
        input_source=input_source
    )
    listener.start()
    try:
        yield listener
    finally:
        listener.stop()
