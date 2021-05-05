from tkinter import Tk
from tkinter.ttk import Button, Label
from tkinter import Frame

from GUI import Label
from GUI import Button
from GUI import Settings

SIZE_WINDOW = "400x300"
ICON = "Resources/Icon/icon.ico"
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

        Label.TitleEntry(frame)                                     # Title
        buttons = Button.ButtonEntry(frame)                         # Button: upload, exit

        Label.FilterEntry(frame)                                    # Label filter
        settingFilter = Settings.SettingsFilterEntry(frame)         # Checkbox filter

        msgLabel = Label.ErrorEntry(frame)                          # Label information

        buttons.set_message_label(msgLabel)
        buttons.set_state_checkbox_filter(settingFilter)


if __name__ == "__main__":
    App().mainloop()
