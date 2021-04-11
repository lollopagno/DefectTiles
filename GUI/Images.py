from tkinter import Frame, Scrollbar, RIGHT, X
from GUI.ScrollableImage import ScrollableImage

class ImageEntry(Frame):
    r"""
    Class Images
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.scrollbar = Scrollbar(self, orient='vertical')

    def add_image(self, img):
        r"""
        Showed the image in the GUi
        :param img: image to show
        """
        image_window = ScrollableImage(self, image=img, scrollbarwidth=6, width=400, height=400)
        image_window.pack(side=RIGHT, padx=10)

