from tkinter import filedialog, LEFT, X, Frame
from tkinter.ttk import Button
from Features.Preprocessing import pre_processing as preprocess
from Features.Defect import common as utility, crack_defect as crack, blob_defect as blob
import cv2 as cv
import numpy as np
import os
import time

SCALE = 1
RESIZE_HEIGHT_IMAGE = 400
RESIZE_WIDTH_IMAGE = 450
CRACKS = "Cracks"
BLOBS = "Blobs"
PATH_HISTOGRAM = "Resources/Histogram/Hist"


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
        self.messageLabel = None
        self.timeLabel = None

        self.path = None

        btnUpload = Button(self, text="Upload", command=self.open_img)
        btnUpload.pack(side=LEFT, padx=20, pady=30)

        btnStart = Button(self, text="Start", command=self.start_detect)
        btnStart.pack(side=LEFT, padx=20)

        btnExit = Button(self, text="Exit", command=self.quit)
        btnExit.pack(side=LEFT, padx=20)

    def check_state_checkbox(self):
        r"""
        Check the current state of the checkbox filters.
        :return: filter chosen by the user
        """

        # Check state check box filter
        median, gaussian, bilateral = self.stateCheckBoxFilter.get_state()
        if (median and gaussian and bilateral) or (median and gaussian) or (median and bilateral) or (
                gaussian and bilateral) or (not median and not gaussian and not bilateral):
            self.messageLabel.update_params_label("Specify the filter to apply: median, gaussian or bilateral!", "red")
            raise Exception("Error filter!")

        elif median:
            filter = "Median"

        elif gaussian:
            filter = "Gaussian"

        else:
            filter = "Bilateral"

        # Check if image is loaded
        if self.path is None:
            self.messageLabel.update_params_label("Upload an image!", "red")
            raise Exception("Error upload image!")

        return filter

    def open_img(self):
        r"""
        Open the image after press buttton upload
        """

        path_img = open_file_name()

        if path_img != "":
            self.path = path_img
            self.messageLabel.update_params_label("Image loaded successfully!", "blue")

    def start_detect(self):
        r"""
        Start detect
        """

        try:

            if not os.path.isdir("Resources/Histogram"):
                # Create directory histogram
                os.mkdir("Resources/Histogram")

            filter = self.check_state_checkbox()
            if self.path is not None:

                # *** START TIME ***
                start_time = time.time()

                img = cv.imread(self.path)

                # Compute histograms and save them.
                file_name = self.path.split("/")[-1]
                utility.histogram(img, file_name)
                histogram = cv.imread(PATH_HISTOGRAM + file_name)

                # Pre processing
                binary_edge_cracks = preprocess.start(img.copy(), filter=filter)

                # Crack Detect
                img_crack_original, img_detected_cracks = crack.detect(img_original=img.copy(),
                                                                       img_edge=binary_edge_cracks)

                # Subtraction of detected defects
                binary_edge_blob = cv.subtract(binary_edge_cracks, img_detected_cracks, cv.CV_8U)

                # Blob Detect
                img_blob = blob.detect(img.copy(), binary_edge_blob)

                # *** END TIME ***
                end_time = time.time()
                self.timeLabel.update_time(round(end_time - start_time, 3))

                cv.destroyWindow('Original')
                cv.destroyWindow('Histogram')

                # Sow result
                if img.shape[1] >= 300:
                    imgStack = stackImages(SCALE, (
                        [draw_description(resize_image(img), "Original image"),
                         draw_description(img_crack_original, "Crack detect"),
                         draw_description(img_blob, "Blob detect")],
                        [histogram, draw_description(binary_edge_cracks, "Binary cracks"),
                         draw_description(binary_edge_blob, "Binary blobs")]))

                    cv.imshow("Result detect", imgStack)

                else:
                    imgStackDetect = stackImages(SCALE, (
                        [draw_description(img_crack_original, "Crack detect"),
                         draw_description(img_blob, "Blob detect")],
                        [draw_description(binary_edge_cracks, "Binary cracks"),
                         draw_description(binary_edge_blob, "Binary blobs")]))

                    cv.imshow("Original", img)
                    cv.imshow("Histogram", histogram)
                    cv.imshow("Result detect", imgStackDetect)

                os.remove(PATH_HISTOGRAM + file_name)
        except Exception as e:
            print(e)

    def set_state_checkbox_filter(self, state):
        r"""
        Set the state of the checkbox filter
        :param state: state of the checkbox filter
        """
        self.stateCheckBoxFilter = state

    def set_message_label(self, obj):
        r"""
        Set object message label
        :param obj: message label object
        """
        self.messageLabel = obj

    def set_time_label(self, obj):
        r"""
        Set object time label
        :param obj: time label object
        """
        self.timeLabel = obj


def resize_image(img):
    r"""
    Resizes the image
    :param img: image to resize
    :return: image to be resized
    """

    height, width = utility.get_shape(img)

    if height > RESIZE_HEIGHT_IMAGE:
        height = RESIZE_HEIGHT_IMAGE
    if width > RESIZE_HEIGHT_IMAGE:
        width = RESIZE_WIDTH_IMAGE

    return cv.resize(img, (width, height))


def draw_description(img, text):
    r"""
    Draw description image
    :param text: image description
    :param img: image in which to insert the descrition
    :return: image with description
    """

    bottom = int(0.08 * img.shape[0])
    img = cv.copyMakeBorder(img, 0, bottom, 0, 0, cv.BORDER_CONSTANT, None, (255, 255, 255))

    height, _ = utility.get_shape(img)
    cv.putText(img, text, (0, height - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


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
