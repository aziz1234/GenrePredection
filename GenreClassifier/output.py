# -*- coding: utf-8 -*-
from preprocess import clean_sentence
from classifier import get_dict
#from classifier import get_test_result
from classifier import predict_genre
from tkinter import  Label, Button, Entry
import tkinter as tk
from tkinter import *
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from tkmagicgrid import MagicGrid
import io
import csv

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
        
        self.t1=Entry(bd=3, width=40)
        
        self.t2 = Entry(bd=3, width=40)
        
        self.predict_button = Button(master, text="Predict Genre", command=self.predict, width = 13, foreground = "blue")
        
        self.display_DB_button = Button(master, text="Display Data Base", command=self.display_db, width = 13, foreground = "green")
        
        self.clear_button = Button(master, text="Clear", command=self.clear, width = 13, foreground = "red")
        
        self.lbl1.grid(row = 0, column = 0, sticky = W, padx = 5, pady = 5)
        self.t1.grid(row=0, column=1, sticky = NW, padx = 5, pady = 5)
        self.lbl2.grid(row = 1, column = 0, sticky = W, padx = 5, pady = 5)
        self.t2.grid(row=1, column=1, sticky = W, padx = 5, pady = 5)
        self.predict_button.grid(row=2, column=0,padx = 5, pady = 5)
        self.display_DB_button.grid(row=2, column=1, padx = 5, pady = 5)
        self.clear_button.grid(row=3, column=0,  padx = 5, pady = 5)


    def predict(self):
        self.newWindow = tk.Toplevel(self.master)
        movie_plot=str(self.t2.get())
       # predict_genre(movie_plot, pipe_dict)
        self.app = Graph(self.newWindow, movie_plot, pipe_dict)
        
    def display_db(self):
        self.newWindow = tk.Toplevel(self.master)
        movie_name = str(self.t1.get())
        global temp_df
        movie_genre = temp_df['genre'].iloc[6]
        self.app = Display_DB(self.newWindow, movie_name, movie_genre)
        
    def clear(self):
        self.t1.delete(0, 'end')
        self.t2.delete(0, 'end')

temp_df= pd.DataFrame()
class Graph:
     def __init__(self, master, s, pipe_dict):
         self.master = master
         master.title("Probablity Graph")
         s_new = clean_sentence(s)
         genre_analyzed = []
         proba = []
         for genre, pipe in pipe_dict.items():
             res = pipe.predict_proba([s_new])
             genre_analyzed.append(genre)
             proba.append(res[0][1])
         data = pd.DataFrame({'genre': genre_analyzed, 'proba': proba})
         data = data.sort_values(by='proba', ascending=True)
         global temp_df
         temp_df = data.copy() 
         figure1 = plt.Figure(figsize=(9,5), dpi=100)
         ax1 = figure1.add_subplot(111)
         bar1 = FigureCanvasTkAgg(figure1, self.master)
         bar1.get_tk_widget().grid(column=0, row=0)
         data.plot(x='genre', y='proba', kind='barh', legend=True, ax=ax1)

class Display_DB:
    def __init__(self,master,movie_name,movie_genre):
        self.master = master
        master.title("DataBase")
        MovieName = movie_name
        Genre = movie_genre
        with open('db.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([MovieName, Genre])
            file.close()
        grid = MagicGrid(self.master)
        grid.pack(side="top", expand=1, fill="both")
        with io.open("db.csv", "r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            parsed_rows = 0
            for row in reader:
                if parsed_rows == 0:
    	    # Display the first row as a header 
                    grid.add_header(*row)
                else:
                    grid.add_row(*row)
                parsed_rows += 1
         
def main(): 
    root = tk.Tk()
    app = GUI(root)
    root.geometry("390x195+10+10")
    root.mainloop()

if __name__ == '__main__':
    main()


