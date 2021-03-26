from tkinter import Tk, filedialog
from tkinter.ttk import Button, Label
from PIL import ImageTk, Image


def openfilename():
    filename = filedialog.askopenfilename(title='"pen')
    return filename


def open_img():
    # Select the Imagename  from a folder
    x = openfilename()

    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.grid(row=2)


root = Tk()
root.title("Defect tiles")
root.geometry("550x300")
root.resizable(width=True, height=True)

btn = Button(root, text='open image', command=open_img).grid(row=1, columnspan=4)

root.mainloop()
