import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk


def display_single_image_results(image_in, image_filename, res_list):

    # Create a window
    window = tk.Tk()
    window.title(Path(image_filename).stem)
    
    # Get the image dimensions
    height, width, no_channels = image_in.shape
    image_in = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    height_sm = int(height / 8)
    width_sm = int(width / 8)
    image_sm = cv2.resize(image_in, (width_sm, height_sm))

    res_string = f'This image contains {res_list[0]} true positives in green, {res_list[1]} false positives in yellow and {res_list[2]} false negatives in red'
    img_string = f'This image is saved at {res_list[4]}'
    csv_string = f'The box locations are saved at {res_list[3]}'

    # add text to canvas
    label1 = tk.Label(window, text=res_string)
    label1.pack()
    label2 = tk.Label(window, text=img_string)
    label2.pack()
    label3 = tk.Label(window, text=csv_string)
    label3.pack()

    # Create a canvas that can fit the above image
    canvas = tk.Canvas(window, width = width_sm, height = height_sm)
    canvas.pack()

    # Convert to a PhotoImage
    photo = ImageTk.PhotoImage(image = Image.fromarray(image_sm))

    # Add a PhotoImage to the Canvas
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)

    # Run the window loop
    window.mainloop()


def manual_check_single_image(results_loc, image_output_loc):
    root = tk.Tk()
    # size of the window
    root.geometry("400x300")
    app = CheckDetectionWindowsSingleImage(results_loc, image_output_loc, root)
    root.mainloop()


class CheckDetectionWindows(tk.Frame):
    def __init__(self, results_path, imagedir, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.results_path = results_path
        gfrc_results = pd.read_csv(self.results_path)
        self.gfrc_results_sort = gfrc_results.sort_values(by='filename')
        if 'checked' not in self.gfrc_results_sort.columns:
            self.gfrc_results_sort['checked'] = np.zeros(self.gfrc_results_sort.shape[0])
        if 'manual_result' not in self.gfrc_results_sort.columns:
            self.gfrc_results_sort['manual_result'] = np.repeat("", self.gfrc_results_sort.shape[0])
        self.pos_folder = imagedir + 'pos/'
        self.neg_folder = imagedir + 'neg/'
        self.init_window()


    # Creation of init_window
    def init_window(self):
        self.imagefile = ""
        self.rowimage = None
        self.idx = int(np.sum(self.gfrc_results_sort['checked']))

        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 25)
        rowxmx = min(7360, row.xmx + 25)
        rowymn = max(0, row.ymn - 25)
        rowymx = min(4912, row.ymx + 25)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        # creating a button instance
        AnimalButton = tk.Button(self, text="Animal", command=self.onClickAnimal)
        NotAnimalButton = tk.Button(self, text="Not Animal", command=self.onClickNot)
        SaveButton = tk.Button(self, text="Save", command=self.onClickSave)
        BackButton = tk.Button(self, text="Back", command=self.onClickBack)

        # placing the button on my window
        AnimalButton.place(x=0, y=250)
        NotAnimalButton.place(x=75, y=250)
        SaveButton.place(x=150, y=250)
        BackButton.place(x=225, y=250)

    def onClickAnimal(self):
        # update details
        self.gfrc_results_sort['checked'].iloc[self.idx] = 1
        self.gfrc_results_sort['manual_result'].iloc[self.idx] = "Animal"
        self.idx = self.idx + 1

        self.imgwindow.config(image="")
        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 25)
        rowxmx = min(7360, row.xmx + 25)
        rowymn = max(0, row.ymn - 25)
        rowymx = min(4912, row.ymx + 25)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

    def onClickNot(self):
        # update details
        self.gfrc_results_sort['checked'].iloc[self.idx] = 1
        self.gfrc_results_sort['manual_result'].iloc[self.idx] = "Not_Animal"
        self.idx = self.idx + 1

        self.imgwindow.config(image="")
        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 25)
        rowxmx = min(7360, row.xmx + 25)
        rowymn = max(0, row.ymn - 25)
        rowymx = min(4912, row.ymx + 25)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)

    def onClickSave(self):
        self.gfrc_results_sort.to_csv(self.results_path, index=False)

    def onClickBack(self):
        # update details
        self.idx = self.idx - 1
        self.gfrc_results_sort['checked'].iloc[self.idx] = 0
        self.gfrc_results_sort['manual_result'].iloc[self.idx] = ""

        self.imgwindow.config(image="")
        row = self.gfrc_results_sort.iloc[self.idx, :]
        rowimagefile = row.filename
        if rowimagefile != self.imagefile:
            rowpath = self.pos_folder + rowimagefile
            if not os.path.isfile(rowpath):
                rowpath = self.neg_folder + rowimagefile
            rowimage = cv2.imread(rowpath)
            self.rowimage = cv2.cvtColor(rowimage, cv2.COLOR_BGR2RGB)
            self.imagefile = rowimagefile
        rowxmn = max(0, row.xmn - 25)
        rowxmx = min(7360, row.xmx + 25)
        rowymn = max(0, row.ymn - 25)
        rowymx = min(4912, row.ymx + 25)
        rowwindow = self.rowimage[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)

        # changing the title of our master widget
        self.master.title(self.imagefile)


class CheckDetectionWindowsSingleImage(tk.Frame):

    def __init__(self, results_path, image_path, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.results_path = results_path
        self.gfrc_results = pd.read_csv(self.results_path)
        if 'checked' not in self.gfrc_results.columns:
            self.gfrc_results['checked'] = np.zeros(self.gfrc_results.shape[0])
        if 'manual_result' not in self.gfrc_results.columns:
            self.gfrc_results['manual_result'] = np.repeat("", self.gfrc_results.shape[0])
        image = cv2.imread(image_path)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.init_window()


    # Creation of init_window
    def init_window(self):
        self.idx = int(np.sum(self.gfrc_results['checked']))

        if self.idx < self.gfrc_results.shape[0]:
            row = self.gfrc_results.iloc[self.idx, :]
            if row.confmat == 'FN':
                print("no detections to check")
            else:
                rowxmn = max(0, row.xmn - 25)
                rowxmx = min(7360, row.xmx + 25)
                rowymn = max(0, row.ymn - 25)
                rowymx = min(4912, row.ymx + 25)
                rowwindow = self.image[rowymn:rowymx, rowxmn:rowxmx, :]

                load = Image.fromarray(rowwindow)
                render = ImageTk.PhotoImage(load)

                # labels can be text or images
                self.imgwindow = tk.Label(self, image=render)
                self.imgwindow.image = render
                self.imgwindow.place(x=0, y=0)
        else:
            print("no detections to check")

        # changing the title of our master widget
        self.master.title("Check Detections")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)

        # creating a button instance
        AnimalButton = tk.Button(self, text="Animal", command=self.onClickAnimal)
        NotAnimalButton = tk.Button(self, text="Not Animal", command=self.onClickNot)
        SaveButton = tk.Button(self, text="Save", command=self.onClickSave)
        BackButton = tk.Button(self, text="Back", command=self.onClickBack)

        # placing the button on my window
        AnimalButton.place(x=0, y=250)
        NotAnimalButton.place(x=75, y=250)
        SaveButton.place(x=150, y=250)
        BackButton.place(x=225, y=250)

    def onClickAnimal(self):
        # update details
        self.gfrc_results['checked'].iloc[self.idx] = 1
        self.gfrc_results['manual_result'].iloc[self.idx] = "Animal"
        self.idx = self.idx + 1

        self.imgwindow.config(image="")
        if self.idx < self.gfrc_results.shape[0]:
            row = self.gfrc_results.iloc[self.idx, :]
            if row.confmat == 'FN':
                print("no more detections to check")
            else:
                rowxmn = max(0, row.xmn - 25)
                rowxmx = min(7360, row.xmx + 25)
                rowymn = max(0, row.ymn - 25)
                rowymx = min(4912, row.ymx + 25)
                rowwindow = self.image[rowymn:rowymx, rowxmn:rowxmx, :]

                load = Image.fromarray(rowwindow)
                render = ImageTk.PhotoImage(load)

                # labels can be text or images
                self.imgwindow = tk.Label(self, image=render)
                self.imgwindow.image = render
                self.imgwindow.place(x=0, y=0)
        else:
            print("no more detections to check")

    def onClickNot(self):
        # update details
        self.gfrc_results['checked'].iloc[self.idx] = 1
        self.gfrc_results['manual_result'].iloc[self.idx] = "Not_Animal"
        self.idx = self.idx + 1

        self.imgwindow.config(image="")
        if self.idx < self.gfrc_results.shape[0]:
            row = self.gfrc_results.iloc[self.idx, :]
            if row.confmat == 'FN':
                print("no more detections to check")
            else:
                rowxmn = max(0, row.xmn - 25)
                rowxmx = min(7360, row.xmx + 25)
                rowymn = max(0, row.ymn - 25)
                rowymx = min(4912, row.ymx + 25)
                rowwindow = self.image[rowymn:rowymx, rowxmn:rowxmx, :]

                load = Image.fromarray(rowwindow)
                render = ImageTk.PhotoImage(load)

                # labels can be text or images
                self.imgwindow = tk.Label(self, image=render)
                self.imgwindow.image = render
                self.imgwindow.place(x=0, y=0)
        else:
            print("no more detections to check")

    def onClickSave(self):
        self.gfrc_results.to_csv(self.results_path, index=False)

    def onClickBack(self):
        # update details
        self.idx = self.idx - 1
        self.gfrc_results['checked'].iloc[self.idx] = 0
        self.gfrc_results['manual_result'].iloc[self.idx] = ""

        self.imgwindow.config(image="")
        row = self.gfrc_results.iloc[self.idx, :]
        rowxmn = max(0, row.xmn - 25)
        rowxmx = min(7360, row.xmx + 25)
        rowymn = max(0, row.ymn - 25)
        rowymx = min(4912, row.ymx + 25)
        rowwindow = self.image[rowymn:rowymx, rowxmn:rowxmx, :]

        load = Image.fromarray(rowwindow)
        render = ImageTk.PhotoImage(load)

        # labels can be text or images
        self.imgwindow = tk.Label(self, image=render)
        self.imgwindow.image = render
        self.imgwindow.place(x=0, y=0)
