#import tkinter as tk
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

class Demo1:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.button1 = tk.Button(self.frame, text = 'New Window', width = 25, command = self.new_window)
        self.button1.pack()
        self.frame.pack()

    def new_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = Demo2(self.newWindow)

class Demo2:
    def __init__(self, master):
        self.master = master
        data1 = {'Country': ['US','CA','GER','UK','FR'],
         'GDP_Per_Capita': [45000,42000,52000,49000,47000]
        }
        df1 = DataFrame(data1,columns=['Country','GDP_Per_Capita'])


        figure1 = plt.Figure(figsize=(6,5), dpi=100)
        ax1 = figure1.add_subplot(111)
        bar1 = FigureCanvasTkAgg(figure1, self.master)
        bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        df1 = df1[['Country','GDP_Per_Capita']].groupby('Country').sum()
        df1.plot(kind='barh', legend=True, ax=ax1)
        ax1.set_title('Country Vs. GDP Per Capita')


    def close_windows(self):
        self.master.destroy()

def main(): 
    root = tk.Tk()
    app = Demo1(root)
    root.geometry("400x200+10+10")
    root.mainloop()

if __name__ == '__main__':
    main()