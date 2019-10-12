from tkinter import *
import tkinter.filedialog
from fastai.vision import *
import torch
from fastai.metrics import error_rate
from PIL import ImageTk, Image

bs = 64
path = "./"
learn = load_learner(path, 'resnet152transform.pkl')

class Redir(object):
    # This is what we're using for the redirect, it needs a text box
    def __init__(self, textbox):
        self.textbox = textbox
        self.textbox.config(state=NORMAL)
        self.fileno = sys.stdout.fileno

    def write(self, message):
        # When you set this up as redirect it needs a write method as the
        # stdin/out will be looking to write to somewhere!
        self.textbox.insert(END, str(message))

def askopenfilename():
    """ Prints the selected files name """
    # get filename, this is the bit that opens up the dialog box this will
    # return a string of the file name you have clicked on.
    filename = tkinter.filedialog.askopenfilename()
    if filename:
        # Will print the file name to the text box
        path1 = filename
        mia1 = open_image(filename)
        img = ImageTk.PhotoImage(Image.open(filename).resize((800,600)))
        pred = learn.predict(mia1)

        label1.configure(image = img)
        label1.img1 = img
        label["text"] = "The selected image is of a " + str(pred[0])


if __name__ == '__main__':

    # Make the root window
    root = Tk()
    root.title("Prediction")

    # Make a button to get the file name
    # The method the button executes is the askopenfilename from above
    # You don't use askopenfilename() because you only want to bind the button
    # to the function, then the button calls the function.
    button = Button(root, text='Select an image', command=askopenfilename)
    # this puts the button at the top in the middle
    button.grid(row=1, column=1,pady=4,padx=100)

    img1 = ImageTk.PhotoImage(Image.open("./a.jpg").resize((800,600)))
    label1 = Label(root, image = img1 )
    label1.grid(row=2,column=1)

    label = Label(root, text = "")
    label.grid(row=3,column=1)

    root.mainloop()
