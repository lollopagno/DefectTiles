from tkinter import Frame, Scrollbar, Canvas

class ScrollableImage(Frame):
    r"""
    Class Scrollable image
    """
    def __init__(self, master=None, **kw):
        self.image = kw.pop('image', None)
        sw = kw.pop('scrollbarwidth', 10)
        super(ScrollableImage, self).__init__(master=master, **kw)

        self.canvas = Canvas(self, highlightthickness=0, **kw)
        self.canvas.create_image(0, 0, anchor='nw', image=self.image)

        # Vertical and Horizontal scrollbars
        self.v_scroll = Scrollbar(self, orient='vertical', width=sw)
        self.h_scroll = Scrollbar(self, orient='horizontal', width=sw)

        # Grid and configure weight.
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.h_scroll.grid(row=1, column=0, sticky='ew')
        self.v_scroll.grid(row=0, column=1, sticky='ns')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Set the scrollbars to the canvas
        self.canvas.config(xscrollcommand=self.h_scroll.set,
                           yscrollcommand=self.v_scroll.set)

        # Set canvas view to the scrollbars
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        # Assign the region to be scrolled
        self.canvas.config(scrollregion=self.canvas.bbox('all'))
        self.canvas.bind_class(self.canvas, "<MouseWheel>", self.mouse_scroll)

    def mouse_scroll(self, evt):
        if evt.state == 0:
            self.canvas.yview_scroll(-1 * evt.delta, 'units')  # For MacOS
            self.canvas.yview_scroll(int(-1 * (evt.delta / 120)), 'units')  # For windows
        if evt.state == 1:
            self.canvas.xview_scroll(-1 * evt.delta, 'units')  # For MacOS
            self.canvas.xview_scroll(int(-1 * (evt.delta / 120)), 'units')  # For windows
