from tkinter import Frame, Checkbutton, IntVar, LEFT, X

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

