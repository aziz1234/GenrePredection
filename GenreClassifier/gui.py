# -*- coding: utf-8 -*-
from tkinter import Tk, Label, Button

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Genre Predection")

        self.lbl1 = Label(master, text="Enter Movie Name!")
        self.lbl1.pack()

        self.greet_button = Button(master, text="predict", command=self.greet)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

root = Tk()
my_gui = GUI(root)
root.mainloop()
root.geometry("400x300+10+10")
root.mainloop()

