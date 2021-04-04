from tkinter import Frame, Label, Scrollbar, Listbox, RIGHT, X, Y

class ImageEntry(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)

        self.scrollbar = Scrollbar(self)
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.list = Listbox(self, yscrollcommand=self.scrollbar.set)

    def addImage(self, img):
        panel = Label(self, image=img)
        panel.image = img
        panel.pack(side=RIGHT, padx=30)
        self.list.insert(panel)
        self.scrollbar.config(command=self.list.yview)

