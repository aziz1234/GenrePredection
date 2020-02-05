from tkinter import Tk
from tkmagicgrid import MagicGrid
import io
import csv

# Create a root window
root = Tk()

# Create a MagicGrid widget
grid = MagicGrid(root)
grid.pack(side="top", expand=1, fill="both")

# Display the contents of some CSV file
# (note this is not a particularly efficient viewer)
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

# Start Tk's event loop
root.mainloop()


##fields = ['MovieName','Plot', 'Genre']
MovieName = "Terminator: Dark Fate"
Plot = "In Mexico City, a newly modified liquid Terminator -- the Rev-9 model -- arrives from the future to kill a young factory worker named Dani Ramos. Also sent back in time is Grace, a hybrid cyborg human who must protect Ramos from the seemingly indestructible robotic assassin. But the two women soon find some much-needed help from a pair of unexpected allies -- seasoned warrior Sarah Connor and the T-800 Terminator."
Genre = "Action" 
with open('db.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["MovieName","Plot", "Genre"])
    writer.writerow([MovieName, Plot, Genre])
#with open('db.csv', 'r') as csvfile:
#    rows =[]
#    # creating a csv reader object 
#    csvreader = csv.reader(csvfile)
#    for row in csvreader: 
#        rows.append(row)
#for row in rows: 
#    # parsing each column of a row 
#    for col in row: 
#        print (col, end="    ")  
#    print("") 
#    
