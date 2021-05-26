from tkinter import Tk
from tkinter.ttk import Button, Label
from tkinter import Frame

from Features.GUI import Label
from Features.GUI import Button
from Features.GUI import Settings

SIZE_WINDOW = "400x300"
ICON = "Resources/Icon/Icon.ico"
TITLE_WINDOW = "Defect detection of tiles"


class App(Tk):
    r"""
    Class App GUI
    """

    def __init__(self):
        super().__init__()
        frame = Frame(self)
        frame.grid(row=0, column=0)

        self.title(TITLE_WINDOW)
        self.iconbitmap(ICON)
        self.geometry(SIZE_WINDOW)
        self.resizable(width=True, height=True)

        Label.TitleEntry(frame)  # Title
        buttons = Button.ButtonEntry(frame)  # Button: upload, exit

        Label.FilterEntry(frame)  # Label filter
        setting_Filter = Settings.SettingsFilterEntry(frame)  # Checkbox filter

        time_label = Label.TimeEntry(frame)  # Time information

        msg_label = Label.ErrorEntry(frame)  # Label information

        buttons.set_time_label(time_label)
        buttons.set_message_label(msg_label)
        buttons.set_state_checkbox_filter(setting_Filter)


if __name__ == "__main__":
    App().mainloop()
