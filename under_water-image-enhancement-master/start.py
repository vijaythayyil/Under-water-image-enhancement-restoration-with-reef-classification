import os
import tkinter as tk
import webbrowser



root= tk.Tk()

canvas1 = tk.Canvas(root, width = 300, height = 200, bg = 'gray90', relief = 'raised')
canvas1.pack()

def myCmd ():
    webbrowser.open_new_tab('http://127.0.0.1:8000/home')
    os.system('cmd /k "py manage.py runserver"')
    
    

def myCmd2 ():
    os.system('cmd /k "kill -SIGINT processPIDHere"')
     
button1 = tk.Button(text='      START SERVER     ', command=myCmd, bg='green', fg='white', font=('helvetica', 12, 'bold'))
button2 = tk.Button(text='      STOP SERVER     ', command=myCmd2, bg='green', fg='white', font=('helvetica', 12, 'bold'))
canvas1.create_window(150, 50, window=button1)
canvas1.create_window(150, 150, window=button2)


root.mainloop()

#155658001298