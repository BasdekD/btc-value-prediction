import pathlib
import os


# In this file necessary global variables for the application are declared

CURR_DIR = os.getcwd() + "\\resources\\"

TRAINED_MODEL = pathlib.Path(CURR_DIR + "finalized_model.sav")
