from tkinter import Tk
from tkinter.ttk import Button, Label
from tkinter import Frame

from GUI import Label
from GUI import Button
from GUI import Settings
from GUI import Images

SIZE_WINDOW = "600x350"
ICON = "Resources/Icon/icon.ico"
TITLE = "Defect tiles"


class App(Tk):
    r"""
    Class App GUI
    """
    def __init__(self):
        super().__init__()
        frame = Frame(self)
        frame.grid(row=0, column=0)

        self.title(TITLE)
        self.iconbitmap(ICON)
        self.geometry(SIZE_WINDOW)
        self.resizable(width=True, height=True)

        Label.TitleEntry(frame)                                     # Title
        buttons = Button.ButtonEntry(frame)                         # Button: upload, exit

        Label.FilterEntry(frame)                                    # Label filter
        settingFilter = Settings.SettingsFilterEntry(frame)         # Checkbox filter

        Label.EdgeDetectionEntry(frame)                             # Label edge detection
        settingDetection = Settings.SettingsDetectionEntry(frame)   # Checkbox detection

        buttons.set_state_checkbox_detection(settingDetection)
        buttons.set_state_checkbox_filter(settingFilter)


if __name__ == "__main__":
    App().mainloop()
