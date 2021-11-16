
# Importing Libraries
from tkinter import *
from tkinter import messagebox as ms
import tkinter as tk
import sqlite3 
from PIL import ImageTk, Image
import tkinter.tix as tix
from train_predict import functionality
# Creating Input




def Log():

    Login = tk.Tk()
    Login.geometry('700x700+0+0')
    Login.title('Login to System !')
    Login.resizable(width=False, height=False)
    

    top_frame = Label(Login, text='Login To System',font = ('Cosmic', 25, 'bold'), bg='#7268A6',relief='groove',padx=500, pady=30)
    top_frame.pack(side='top')
    
    frame = LabelFrame(Login, padx=40, pady=30, bg='white')
    frame.place(relx = 0.5, rely = 0.55, anchor = CENTER)
    
    def Search(arg = None):
        if username_entry.get() == '': 
            ms.showerror('Oops', 'Enter Username !!')
            
        elif password_entry.get() == '':
            ms.showerror('Oops', 'Enter Password !!')
            
        else:
            global username
            username = username_entry.get()
            global password
            password = password_entry.get()

            # Making connection
            conn = sqlite3.connect('Database.db')

            # Creating cursor
            with conn:
                cursor = conn.cursor()

            # Searching for users
            find_user = ('SELECT * FROM users WHERE Username = ? AND Password = ?')
            cursor.execute(find_user,(username, password))
            results = cursor.fetchall()

            # if user then new window
            if results:
                Login.destroy()
                functionality(username)

            # if user do not exist
            else:
                ms.showerror('Oops','User Not Found !! Check Username and Password Again !!')

    # creating a label for username and password using Label 
    

    img = ImageTk.PhotoImage(Image.open("img/icon.png"))
    panel = tk.Label(frame, image = img)
    panel.config(width=10,height=10)
  



    username = tk.Label(frame, text = 'Username',font=('Arial',12, 'bold'),bg='white', fg='green')
    panel1 = tk.Label(frame, image = img)
    panel1.config(width=10,height=10)
    password = tk.Label(frame, text = 'Password', font = ('Arial',12,'bold'),bg='white', fg='green')

    def user_help():
        messagebox.showinfo("User Name","Enter your username",parent=frame)
        
    # -------------------  ADD IMAGE
    # opens the image
    img = Image.open("icon.png")
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((25,25), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel2 = Button(frame, image = img,relief = "flat",command=user_help)
    # set the image as img
    panel2.image = img
    panel2.place(height=25,width=25,x=0, y=10)

    def pass_help():
        messagebox.showinfo("Password","Enter your Password",parent=frame)
        
    # -------------------  ADD IMAGE
    # opens the image
    img = Image.open("icon.png")
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((25,25), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel3 = Button(frame, image = img,relief = "flat",command=pass_help)
    # set the image as img
    panel3.image = img
    panel3.place(height=25,width=25,x=0, y=90)

    

    # creating a entry for username 
    username_entry = tk.Entry(frame, font=('calibre',10,'normal'), justify = 'center', bg='#FBB13C')
    username_entry.bind('<Return>', Search)
    password_entry=tk.Entry(frame, font = ('calibre',10,'normal'), show = '*', justify = 'center', bg='#FBB13C') 
    password_entry.bind('<Return>', Search)
    
    # Button that will call the submit function  
    submit=tk.Button(frame,text = 'Login', command = Search, width="10",bd = '3',  font = ('Times', 12, 'bold'), bg='#581845', fg='white',relief='groove', justify = 'center', pady='5') 

    # Placing the label and entry 
    panel.pack(anchor="nw")

    username.pack()
    username_entry.focus_set()
    username_entry.pack()
    
    # Label for seperating Buttons
    label = Label(frame, bg='white').pack()
    panel1.pack(anchor="nw") 
    password.pack() 
    password_entry.pack()

    # Label for seperating Buttons
    label = Label(frame, bg='white').pack()
    
    submit.pack()

    # Quit Button

    Quit = tk.Button(Login, text = "Quit", width="10", command = Login.destroy, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Quit.place(anchor ='sw',rely=1,relx=0.775)

    def backk():
        Login.destroy()
        from Code import create
        create()

    Back = tk.Button(Login, text = "Back", width="10", command = backk, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Back.place(anchor ='se',rely=1,relx=0.775)

    footer=Label(Login, bg="white",text="Document Classification version 1.0",fg="black",font="Leelawadee 14")
    footer.place(height=36,width=350,x=50, y=657)
    
