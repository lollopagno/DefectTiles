from tkinter import Frame, Label, TOP, LEFT, X


class TitleEntry(Frame):
    r"""
    Class Title Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Defect Tiles", width=14, font=('Arial', 17, 'bold'))
        label.pack(side=TOP, padx=150, pady=10)


class FilterEntry(Frame):
    r"""
    Class Filter Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Filter:", width=6, font=('Arial', 13, "underline"))
        label.pack(side=LEFT, padx=12)


class EdgeDetectionEntry(Frame):
    r"""
    Class Edge detection Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Edge detection:", width=14, font=('Arial', 13, "underline"))
        label.pack(side=LEFT, padx=12)


class ErrorEntry(Frame):
    r"""
    Class Error Label
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.label = Label(self, text="", width=0, fg="red", font=('Arial', 11))

    def enabled(self, text, width, color):
        r"""
        Show the label
        :param text: text to show
        :param width: with of the label
        """
        self.label = Label(self, text=text, width=width, fg=color, font=('Arial', 11))
        self.label.pack(side=LEFT, padx=10, pady=30)

    def disabled(self):
        r"""
        Hide the label
        """
        self.label.pack_forget()
