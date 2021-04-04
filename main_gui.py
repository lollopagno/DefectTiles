from tkinter import Tk
from tkinter.ttk import Button, Label
from tkinter import Frame

from GUI import Label
from GUI import Button
from GUI import Settings

SIZE_WINDOW = "1000x700"
ICON = "Resources/Icon/icon.ico"
TITLE = "Defect tiles"

class App(Tk):
    def __init__(self):
        super().__init__()
        frame = Frame(self)
        frame.grid(row=0, column=0)

        self.title(TITLE)
        self.iconbitmap(ICON)
        self.geometry(SIZE_WINDOW)
        self.resizable(width=True, height=True)

        Label.TitleEntry(frame)
        btn = Button.ButtonEntry(frame)
        Label.EdgeDetectionEntry(frame)
        settings = Settings.SettingsEntry(frame)

        btn.setStateCheckbox(settings)


if __name__ == "__main__":
    App().mainloop()
