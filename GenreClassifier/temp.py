#import tkinter as tk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
#from tkmagicgrid import MagicGrid
import io
import csv
#
#class graph:
#    def __init__(self, master):
#        data1 = {'Country': ['US','CA','GER','UK','FR'],
#         'GDP_Per_Capita': [45000,42000,52000,49000,47000]
#        }
#        df1 = DataFrame(data1,columns=['Country','GDP_Per_Capita'])
#
#
#        figure1 = plt.Figure(figsize=(6,5), dpi=100)
#        ax1 = figure1.add_subplot(111)
#        bar1 = FigureCanvasTkAgg(figure1, root)
#        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
#        df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
#        df1.plot(kind='barh', legend=True, ax=ax1)
#        ax1.set_title('Country Vs. GDP Per Capita')
#        plt.show()
#class temp:
#    def __init__(self, master):
#        self.master = master
#        self.frame = tk.Frame(self.master)
#        self.button1 = tk.Button(self.frame, text = 'New Window', width = 25, command = self.new_window)
#        self.button1.pack()
#        self.frame.pack()
#
#    def new_window(self):
#        self.newWindow = tk.Tk()
#        self.app = graph(self.newWindow)
#        
#
#root= tk.Tk()
#GUI = temp(root) 
#root.mainloop()
import tkinter as tk
from tkinter import  Label, Button, Entry
import wikipediaapi

class Demo1:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text = 'New Window', width = 25, command = self.new_window)
        self.button1.pack()
        self.t1=Entry(bd=3, width=40)
        self.t1.pack()
        self.t2=Entry(bd=3, width=40)
        self.t2.pack()
        self.lbl1 = Label(master, text="Enter Movie Name:")
        self.lbl1.pack()
        self.display = Button(master, text="Search plot", command=self.display, width = 13, foreground = "blue")
        self.display.pack()
        self.frame.pack()

    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)
    def display(self):
         movie_name=str(self.t1.get())
         wiki_wiki = wikipediaapi.Wikipedia('en')

         page = wiki_wiki.page(movie_name)
         def print_sections(sections):
             for i in sections:
                 if (i.title=="Plot"):
                     self.t2.insert(0,i.text[0:])
                     #print(i.text[0:])

         print_sections(page.sections)

class Demo2:
    def __init__(self, master):
        self.master = master
#        data1 = {'Country': ['US','CA','GER','UK','FR'],
#         'GDP_Per_Capita': [45000,42000,52000,49000,47000]
#        }
#        df1 = DataFrame(data1,columns=['Country','GDP_Per_Capita'])
        resultset = pd.read_csv('D:/GenreClassifier/results.csv' ,low_memory=False, dtype = str)
#
#        figure1 = plt.Figure(figsize=(6,5), dpi=100)
#        ax1 = figure1.add_subplot(111)
#        bar1 = FigureCanvasTkAgg(figure1, self.master)
#        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
#        df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
#        df1.plot(kind='barh', legend=True, ax=ax1)
#        ax1.set_title('Country Vs. GDP Per Capita')
        x = resultset['genre']
        
        y = [resultset['multi_nb'][13],resultset['sgdc'][13],resultset['logi_regre'][13],resultset['random_f'][13]]
        print(y)
        z = resultset['logi_regre']
        data = pd.DataFrame({'x': x, 'y1': y,'z1':z})
        data.y1= pd.to_numeric(data.y1)
        data.z1= pd.to_numeric(data.z1)
        
       
        figure1 = plt.Figure(figsize=(8,4), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, self.master)
        bar1.get_tk_widget().grid(column=0, row=0)
        data.plot(x='x', y='y',kind='bar',color='green',legend=True, ax=ax1)
        ax2= ax1.twinx()
        data.plot(x='x', y='z',kind='bar', legend=True, ax=ax2)



def main(): 
    root = tk.Tk()
    app = Demo1(root)
    root.geometry("450x300+10+10")
    root.mainloop()

if __name__ == '__main__':
    main()