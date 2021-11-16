from tkinter import *
from tkinter import messagebox as ms
import tkinter as tk
import sqlite3
from login import Log
from register import Register
from PIL import ImageTk, Image

    
def create():
    def Quit():
        response = ms.askokcancel('Exit!', 'Do you really want to exit ?')
        if response == 1:
            system.destroy()
        else:
            pass
        
    def destroy_login():
        system.destroy()
        Log()
        
    def destroy_register():
        Register()
        
    system = tk.Tk()
    system.geometry('700x700+0+0')
    system.configure(background='#3B2C35')
    system.resizable(width=False, height=False)
    system.title('Login and Registration System')
    
    top_frame = Label(system, text='WELCOME TO THE SYSTEM',font = ('Cosmic', 25, 'bold'), bg='#7268A6',relief='groove',padx=500, pady=30)
    top_frame.pack(side='top')
    
    canvas = Canvas(system, width=700, height=700)
    # image = ImageTk.PhotoImage(Image.open('img/aa.jpeg').resize((700, 700), Image.ANTIALIAS))
    # canvas.create_image(0,0, anchor=NW, image=image)
    canvas.pack()
    
    frame = LabelFrame(system,text='SERVICES', padx=30, pady=40, bg='white', bd='5', relief='groove')
    frame.place(relx = 0.5, rely = 0.5, anchor = CENTER)
    
    login = tk.Button(frame, text = "Login", width="10", bd = '3', command = destroy_login, font = ('Times', 12, 'bold'), bg='#7268A6',relief='groove', justify = 'center', pady='5')
    login.pack()
    
    label = Label(frame, bg='white').pack()
    
    register = tk.Button(frame, text = "Register", width="10", bd = '3',  command = destroy_register, font = ('Times', 12, 'bold'), bg='#2A1F2D',fg='white', relief='groove', justify = 'center', pady='5')
    register.pack()
            
    Quit = tk.Button(system, text = "Quit", width="10",bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Quit.place(anchor ='sw',rely=1,relx=0.775)

    footer=Label(system, bg="white",text="Document Classification version 1.0",fg="black",font="Leelawadee 14")
    footer.place(height=36,width=480,x=60, y=657)

    system.mainloop()
        
create()
