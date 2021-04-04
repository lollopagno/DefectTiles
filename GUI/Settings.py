from tkinter import Frame, Checkbutton, IntVar, LEFT, X


class SettingsEntry(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.varSobel = IntVar()
        self.varCanny = IntVar()

        sobel = Checkbutton(self, text="Sobel", variable=self.varSobel)
        sobel.pack(side=LEFT, padx=20, pady=20)

        canny = Checkbutton(self, text="Canny", variable=self.varCanny)
        canny.pack(side=LEFT, fill=X, padx=10)

    def getState(self):
        return self.varSobel.get(), self.varCanny.get()
