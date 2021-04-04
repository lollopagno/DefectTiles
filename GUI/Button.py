from tkinter import filedialog, LEFT, RIGHT, X, Frame
from tkinter.ttk import Button, Label
from PIL import ImageTk, Image
from GUI import Images, Label
from Preprocessing import pre_processing as preprocess
from Defect import crack_defect as crack


def openfilename():
    r"""
    Select the image from a folder
    :return: selected image
    """
    filename = filedialog.askopenfilename(title='Upload image')
    return filename


class ButtonEntry(Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.stateCheckBox = None
        self.imageClass = Images.ImageEntry(self)
        self.labelErrEdgeDetect = Label.ErrorEdgeDetectionEntry(self)

        btnUpload = Button(self, text="Upload", command=self.open_img)
        btnUpload.pack(side=LEFT, padx=20, pady=30)

        btnExit = Button(self, text="Exit", command=self.quit)
        btnExit.pack(side=LEFT, fill=X, padx=10)

    def open_img(self):
        try:
            sobel, canny = self.stateCheckBox.getState()
            if (sobel and canny) or (not sobel and not canny):
                self.labelErrEdgeDetect.enabled()
                raise Exception("Error edge detection")
            elif sobel:
                method = "Sobel"
            else:
                method = "Canny"

            self.labelErrEdgeDetect.disabled()
            x = openfilename()
            img = Image.open(x)
            img = ImageTk.PhotoImage(img)
            self.imageClass.addImage(img)
            print("init pre-processing")
            img_pre_processing = preprocess.start(img, method=method)
            crack_detect = crack.detect(img_pre_processing, method=method)
            self.imageClass.addImage(crack_detect)

        except Exception as e:
            print(e)

    def setStateCheckbox(self, state):
        self.stateCheckBox = state
