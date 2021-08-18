import cv2
import os
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
import tkinter as tk

from postprocess.display_output import CheckDetectionWindows

root = tk.Tk()

# size of the window
root.geometry("400x300")

results_path = 'C:/Users/kryzi/Documents/gfrc_rgb_baseline2_results.csv'
imagedir = 'C:/Users/kryzi/OneDrive - University of St Andrews/PhD/Data_to_Save/test_images/'
app = CheckDetectionWindows(results_path, imagedir, root)
root.mainloop()
