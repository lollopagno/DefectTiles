from tkinter import Frame, Checkbutton, IntVar, LEFT, X

class SettingsDetectionEntry(Frame):
    r"""
    Class Setting Detection Method
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.varSobel = IntVar()
        self.varCanny = IntVar()

        sobel = Checkbutton(self, text="Sobel", variable=self.varSobel)
        sobel.pack(side=LEFT, padx=20, pady=20)

        canny = Checkbutton(self, text="Canny", variable=self.varCanny)
        canny.pack(side=LEFT, fill=X, padx=10)

    def get_state(self):
        r"""
        Return the state of this checkbox
        """
        return self.varSobel.get(), self.varCanny.get()

class SettingsFilterEntry(Frame):
    r"""
    Class Setting Filters
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.varMedian = IntVar()
        self.varGaussian = IntVar()
        self.varBilateral = IntVar()

        median = Checkbutton(self, text="Median Blur", variable=self.varMedian)
        median.pack(side=LEFT, padx=19, pady=20)

        gaussian = Checkbutton(self, text="Gaussian", variable=self.varGaussian)
        gaussian.pack(side=LEFT, fill=X, padx=10)

        var = Checkbutton(self, text="Bilateral", variable=self.varBilateral)
        var.pack(side=LEFT, fill=X, padx=10)

    def get_state(self):
        r"""
        Return the state of this checkbox
        """
        return self.varMedian.get(), self.varGaussian.get(), self.varBilateral.get()

