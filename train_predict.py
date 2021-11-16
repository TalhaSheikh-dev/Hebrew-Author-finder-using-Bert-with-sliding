# Importing tkinter and login, register libraries
from tkinter import *
from tkinter import messagebox as ms
import tkinter as tk
from PIL import ImageTk, Image
from predict import Predict
from train import Train



def functionality(username):
        system1 = tk.Tk()
        system1.geometry('700x700+0+0')
        system1.resizable(width=False, height=False)
        system1.title('Predict or Train')

        top_frame = Label(system1, text='WELCOME TO THE SYSTEM',font = ('Cosmic', 25, 'bold'), bg='#7268A6',relief='groove',padx=500, pady=30)
        top_frame.pack(side='top')

        canvas = Canvas(system1, width=700, height=700)

        #       image = ImageTk.PhotoImage(Image.open('img/aa.jpeg').resize((700, 700), Image.ANTIALIAS))

        #       canvas.create_image(0,0, anchor=NW, image=image)
        canvas.pack()

        frame = LabelFrame(system1,text='SERVICES', padx=30, pady=40, bg='white', bd='5', relief='groove')
        frame.place(relx = 0.5, rely = 0.5, anchor = CENTER)
        
        def pred(username):
                system1.destroy()
                Predict(username)
        login = tk.Button(frame, text = "Predict", width="10", bd = '3', command = lambda: pred(username) , font = ('Times', 12, 'bold'), bg='#7268A6',relief='groove', justify = 'center', pady='5')
        login.pack()

        label = Label(frame, bg='white').pack()

        def trains(username):
                system1.destroy()
                Train(username)
        register = tk.Button(frame, text = "Train", width="10", bd = '3',  command = lambda: trains(username), font = ('Times', 12, 'bold'), bg='#2A1F2D',fg='white', relief='groove', justify = 'center', pady='5')
        register.pack()
        
        Quit1 = tk.Button(system1, text = "Quit", width="10", command = system1.destroy, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
        Quit1.place(anchor ='sw',rely=1,relx=0.775)
        def backed():
                system1.destroy()
                from login import Log
                Log()
        Back = tk.Button(system1, text = "Back", width="10", command = backed, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
        Back.place(anchor ='se',rely=1,relx=0.775)
        footer=Label(system1, bg="white",text="Document Classification version 1.0",fg="black",font="Leelawadee 14")
        footer.place(height=36,width=350,x=20, y=657)
    
