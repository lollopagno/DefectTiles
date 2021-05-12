from tkinter import Frame, Label, TOP, LEFT, X


class TitleEntry(Frame):
    r"""
    Class Title Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Defect detection of tiles", width=20, font=('Arial', 17, 'bold'))
        label.pack(side=TOP, padx=50, pady=10)


class FilterEntry(Frame):
    r"""
    Class Filter Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Filter:", width=6, font=('Arial', 13, "underline"))
        label.pack(side=LEFT, padx=14)


class TimeEntry(Frame):
    r"""
    Class Time Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.label = Label(self, text="Time for detection: 0 sec", width=20, font=('Arial', 10))
        self.label.pack(side=LEFT, padx=12)

    def update_time(self, time):
        r"""
        Updated text label
        :param time: time taken for detection
        """
        self.label.config(text=f"Time for detection: {time} sec", padx=12)


class ErrorEntry(Frame):
    r"""
    Class Error Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.label = Label(self, text="", width=0, fg="red", font=('Arial', 11))

    def enabled(self, text, color):
        r"""
        Show the error label
        :param text: text to show
        :param width: with of the label
        :param color: color of the text
        """
        self.label = Label(self, text=text, width=40, fg=color, font=('Arial', 11))
        self.label.pack(side=LEFT, padx=20, pady=10)

    def disabled(self):
        r"""
        Hide the error label
        """
        self.label.pack_forget()
