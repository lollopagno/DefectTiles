from tkinter import filedialog, LEFT, RIGHT, X, Frame
from tkinter.ttk import Button, Label
from PIL import ImageTk, Image
from GUI import Images, Label
from Preprocessing import pre_processing as preprocess
from Defect import crack_defect as crack
import cv2 as cv


def openfilename():
    r"""
    Select the image from a folder
    :return: selected image
    """
    filename = filedialog.askopenfilename(title='Upload image')
    return filename


class ButtonEntry(Frame):
    r"""
    Class Button entry
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(fill=X)
        self.stateCheckBoxFilter = None
        self.stateCheckBoxDetection = None
        self.objImage = None

        btnUpload = Button(self, text="Upload", command=self.open_img)
        btnUpload.pack(side=LEFT, padx=20, pady=30)

        btnExit = Button(self, text="Exit", command=self.quit)
        btnExit.pack(side=LEFT, fill=X, padx=5)

        self.labelErr = Label.ErrorEntry(self)

    def open_img(self):
        r"""
        Check the current state of the checkbox filters and edge detection methods.
        Open the image after press buttton upload
        """
        try:
            self.labelErr.disabled()

            # Check state check box filter
            median, gaussian, bilateral = self.stateCheckBoxFilter.getState()
            if (median and gaussian and bilateral) or (median and gaussian) or (median and bilateral) or (
                    gaussian and bilateral) or (not median and not gaussian and not bilateral):
                self.labelErr.enabled("Specify the filter to apply: median, gaussian or bilateral!", 40)
                raise Exception("Error filter!")
            elif median:
                filter = "Median"
            elif gaussian:
                filter = "Gaussian"
            else:
                filter = "Bilateral"

            self.labelErr.disabled()

            # Check state check box edge detection method
            sobel, canny = self.stateCheckBoxDetection.getState()
            if (sobel and canny) or (not sobel and not canny):
                self.labelErr.enabled("Specify the edge detection method: Canny or Sobel!", 38)
                raise Exception("Error edge detection!")
            elif sobel:
                method_edge_detection = "Sobel"
            else:
                method_edge_detection = "Canny"

            self.labelErr.disabled()

            path = openfilename()
            if path != "":
                img = cv.imread(path)
                imgOriginal = img.copy()
                img = resizeImage(img)

                # self.objImage.enabledScrool()
                self.objImage.addImage(img)

                img_pre_processing = preprocess.start(imgOriginal, filter=filter, method_edge_detection=filter)
                img_crack = crack.detect(img_pre_processing, method=method_edge_detection)
                img_crack = convertCVToPIL(img_crack)
                self.objImage.addImage(img_crack)

        except Exception as e:
            print(e)

    def setStateCheckboxDetection(self, state):
        r"""
        Set the state of the checkbox detection methods
        :param state: state of the checkbox detection methods
        """
        self.stateCheckBoxDetection = state

    def setStateCheckboxFilter(self, state):
        r"""
        Set the state of the checkbox filter
        :param state: state of the checkbox filter
        """
        self.stateCheckBoxFilter = state

    def setObjImages(self, objImage):
        r"""
        Saves the instance of the object
        :param objImage: instance of the object of images
        """
        self.objImage = objImage


def resizeImage(img):
    r"""
    Resizesthe image
    :param img: image to resize
    :return: image PIL to be resized
    """
    height, width, _ = img.shape
    if height > 400:
        height = 400
    if width > 400:
        width = 400

    imgResized = cv.resize(img, (height, width))
    return convertCVToPIL(imgResized)


def convertCVToPIL(img):
    r"""
    Convert the image from open-cv to PIL
    :param img: image in open-cv to convert
    :return: image PIL to be converted
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return ImageTk.PhotoImage(img)
