# -*- coding: utf-8 -*-
from preprocess import clean_sentence
from classifier import get_dict
#from classifier import get_test_result
from classifier import predict_genre
from tkinter import Tk, Label, Button, Entry
from tkinter import *
#movie_plot = """ 
#Professional rock climber Alex Honnold attempts to conquer the first free solo climb of famed El Capitan's 900-metre vertical rock face at Yosemite National Park.
#"""
pipe_dict=get_dict()

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Genre Predection")

        self.lbl1 = Label(master, text="Enter Movie Name:")
        
        self.lbl2 = Label(master, text="Enter The Plot:")
        
        self.t1=Entry(bd=3, width=25)
        
        self.t2 = Entry(bd=3, width = 40)
        
        self.predict_button = Button(master, text="Predict Genre", command=self.predict)
        
        #self.close_button = Button(master, text="Close", command=master.quit)
        
        self.lbl1.grid(row = 0, column = 0, sticky = W, padx = 5, pady = 5)
        self.t1.grid(row=0, column=1, sticky = W, padx = 5, pady = 5)
        self.lbl2.grid(row = 1, column = 0, sticky = W, padx = 5, pady = 5)
        self.t2.grid(row=1, column=1, sticky = W, padx = 5, pady = 5)
        self.predict_button.grid(row=2, column=0,padx = 5, pady = 5, columnspan = 2)
       # self.close_button.grid(row=2, column=1, padx = 5, pady = 5)


    def predict(self):
        movie_plot=str(self.t2.get())
        predict_genre(movie_plot, pipe_dict)

root = Tk()
my_gui = GUI(root)
root.geometry("400x200+10+10")
root.mainloop()

