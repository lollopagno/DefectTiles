from tkinter import Frame, Label, TOP, LEFT, X


class TitleEntry(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Defect Tiles", width=14, font=('Arial', 17, 'bold'))
        label.pack(side=TOP, padx=400, pady=10)


class EdgeDetectionEntry(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        label = Label(self, text="Edge detection:", width=14, font=('Arial', 13, "underline"))
        label.pack(side=LEFT, padx=12)


class ErrorEdgeDetectionEntry(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.label = Label(self, text="Specify the edge detection method: Canny or Sobel!", width=38, fg="red",
                           font=('Arial', 11))

    def enabled(self):
        self.label.pack(side=LEFT, padx=10, pady=30)

    def disabled(self):
        self.label.pack_forget()
