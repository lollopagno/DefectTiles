from tkinter import filedialog, LEFT, X, Frame
from tkinter.ttk import Button, Label
from PIL import ImageTk, Image
from GUI import Label
from Preprocessing import pre_processing as preprocess
from Defect import crack_defect as crack
import cv2 as cv
import numpy as np

SCALE = 1
RESIZE_IMAGE = 400


def open_file_name():
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

        btnUpload = Button(self, text="Upload", command=self.open_img)
        btnUpload.pack(side=LEFT, padx=20, pady=30)

        btnExit = Button(self, text="Exit", command=self.quit)
        btnExit.pack(side=LEFT, fill=X, padx=5)

        self.labelErr = Label.ErrorEntry(self)

    def check_state_checkbox(self):
        r"""
         Check the current state of the checkbox filters and edge detection methods.
        :return: filter and edge detection method chosen by the user
        """
        self.labelErr.disabled()

        # Check state check box filter
        median, gaussian, bilateral = self.stateCheckBoxFilter.get_state()
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
        sobel, canny = self.stateCheckBoxDetection.get_state()
        if (sobel and canny) or (not sobel and not canny):
            self.labelErr.enabled("Specify the edge detection method: Canny or Sobel!", 38)
            raise Exception("Error edge detection!")
        elif sobel:
            method_edge_detection = "Sobel"
        else:
            method_edge_detection = "Canny"

        self.labelErr.disabled()
        return filter, method_edge_detection

    def open_img(self):
        r"""
        Open the image after press buttton upload
        """
        try:
            filter, edge_detection = self.check_state_checkbox()
            path = open_file_name()
            if path != "":
                img = cv.imread(path)
                img_original = img.copy()
                img = resize_image(img)

                # Pre processing
                img_pre_processing = preprocess.start(img_original, filter=filter, edge_detection=edge_detection)

                # Crack Detect
                img_crack = crack.detect(img_pre_processing, method=edge_detection)

                imgStack = stackImages(SCALE, ([img, img_pre_processing], [img_crack, img_crack]))
                cv.imshow("Result", imgStack)
        except Exception as e:
            print(e)

    def set_state_checkbox_detection(self, state):
        r"""
        Set the state of the checkbox detection methods
        :param state: state of the checkbox detection methods
        """
        self.stateCheckBoxDetection = state

    def set_state_checkbox_filter(self, state):
        r"""
        Set the state of the checkbox filter
        :param state: state of the checkbox filter
        """
        self.stateCheckBoxFilter = state


def resize_image(img):
    r"""
    Resizes the image
    :param img: image to resize
    :return: image PIL to be resized
    """
    height, width, _ = img.shape
    if height > RESIZE_IMAGE:
        height = RESIZE_IMAGE
    if width > RESIZE_IMAGE:
        width = RESIZE_IMAGE

    imgResized = cv.resize(img, (height, width))
    return imgResized


def stackImages(scale, imgArray):
    r"""
    Stack the images based on the number of them by rows and columns.
    Resize the images.
    :param scale: scale factor
    :param imgArray: array of images
    :return: array of images to show
    """
    rows = len(imgArray)
    cols = len(imgArray[0])

    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):

                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                               None, scale, scale)

                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)

        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

    else:
        for x in range(0, rows):

            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)

            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)

        hor = np.hstack(imgArray)
        ver = hor

    return ver

# def convert_cv_to_pil(img):
#     r"""
#     Convert the image from open-cv to PIL, resize image
#     :param img: image in open-cv to convert
#     :return: image PIL to be converted
#     """
#     img = cv.resize(img, (200, 200))
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img = Image.fromarray(img)
#     return ImageTk.PhotoImage(img)
