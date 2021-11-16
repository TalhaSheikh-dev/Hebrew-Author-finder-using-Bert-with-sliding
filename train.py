from tkinter import *
from tkinter import messagebox as ms
import tkinter as tk
import sqlite3
from train_new import train_new_model
from PIL import ImageTk, Image
import os



def check_directory(directory):
    if os.path.exists(directory):
        for a in os.listdir(directory):
            if not os.path.isdir(os.path.join(directory,a)):
                return "The folder should only contain folders and not other file"
            else:
                for x in os.listdir(os.path.join(directory,a)):
                    if not x.endswith(".txt"):
                        return "all file should be .txt file"
    else:
        return "The Directory is wrong or doesn't exists"

    return True

def Train(username):
    
    # Creating a new window
    Reg = tk.Tk()
    Reg.title('Train using system')
    Reg.geometry('700x700+0+0')

    Reg.resizable(width=False, height=False)

    top_frame = Label(Reg, text='Train',font = ('Cosmic', 25, 'bold'), bg='#7268A6',relief='groove',padx=500, pady=30)
    top_frame.pack(side='top')
    
    frame = LabelFrame(Reg, padx=30, bg='white')
    frame.place(relx = 0.5, rely = 0.25, anchor = "s")
    
    def directory_getter(arg = None):
        txt = name_entry.get()
        if txt != "":
            directory_check_msg = check_directory(txt)
            if directory_check_msg:
                systemprogress=Label(Reg, text = 'System is in Progress......\n  Please wait', font=('Arial',12, 'bold'), bg='white', fg='green')
                systemprogress.place(height=70,width=350,x=250,y=320)
                systemprogress.update_idletasks()
                out = train_new_model(username,txt)
                set1()
                systemprogress.destroy()
                if out ==1:
                    messagebox.showinfo(title="Training", message="successfully done the training",parent=Reg)

                else:
                    ms.showerror('Oops','Unable to train the model for technical issues')
            else:
                ms.showerror('Oops',directory_check_msg)
        else:
            ms.showerror('Oops',"Unable to train the model for technical issues")
            
        
    name = tk.Label(frame, text = 'Directory', font=('Arial',12, 'bold'), bg='white', fg='green')
    
    def train_help():
        messagebox.showinfo("Directory","Enter the directory name under which you have the data. like data_our",parent=frame)
        
    # -------------------  ADD IMAGE
    # opens the image
    img = Image.open("icon.png")
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((25,25), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel3 = Button(frame, image = img,relief = "flat",command=train_help)
    # set the image as img
    panel3.image = img
    panel3.place(height=25,width=25,x=0, y=50)
    
    name_entry = tk.Entry(frame ,font=('Arial',12,'normal'), bg='#FBB13C')
    name_entry.bind('<Return>', directory_getter)
    # Button that will call the submit function  
    submit=tk.Button(frame,text = 'Train', command = directory_getter, width="10",bd = '3',  font = ('Times', 12, 'bold'),bg='#581845', 	 fg='white',relief='groove', justify = 'center', pady='5'  ) 
    Quit = tk.Button(Reg, text = "Quit", width="10", command = Reg.destroy, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Quit.place(anchor ='sw',rely=1,relx=0.84)

    def back():
        Reg.destroy()
        from train_predict import functionality
        functionality(username)

    Back = tk.Button(Reg, text = "Back", width="10", command =back, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Back.place(anchor ='se',rely=1,relx=0.84) 

    name.pack()
    name_entry.focus_set()
    name_entry.pack()


    # -------------------  ADD IMAGE
    # opens the image
    def set1():
        path1 = os.path.join("saved_models",username,"accuracy.png")
        img = Image.open(path1)
    # resize the image and apply a high-quality down sampling filter
        img = img.resize((300,300), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
        img = ImageTk.PhotoImage(img)
    # create a label
        panel = Label(Reg, image = img)
    # set the image as img
        panel.image = img
        panel.place(height=300,width=300,x=30, y=250)

    # -------------------  ADD IMAGE
    # opens the image
        path2 = os.path.join("saved_models",username,"loss.png")
        img = Image.open(path2)
    # resize the image and apply a high-quality down sampling filter
        img = img.resize((300,300), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
        img = ImageTk.PhotoImage(img)
    # create a label
        panel2 = Label(Reg, image = img)
    # set the image as img
        panel2.image = img
        panel2.place(height=300,width=300,x=350, y=250)

    # graph 1
        path3 = os.path.join("saved_models",username,"data.txt")

                
    
        accurary_label1=Label(Reg, bg="orange",text="Training Accuracy : ",fg="black",font="Leelawadee 10")
        accurary_label1.place(height=25,width=70,x=30, y=560)
        loss_label1=Label(Reg, bg="orange",text="Training  Loss : ",fg="black",font="Leelawadee 10")
        loss_label1.place(height=25,width=70,x=30, y=590)
#        time_label1=Label(Reg, bg="orange",text="Time : ",fg="black",font="Leelawadee 10")
#        time_label1.place(height=25,width=70,x=30, y=620)

        accurary_entry1=Entry(Reg, bg="black",fg="white",font="Leelawadee 14")
        accurary_entry1.place(height=25,width=210,x=110, y=560)
        loss_entry1=Entry(Reg, bg="black",fg="white",font="Leelawadee 14")
        loss_entry1.place(height=25,width=210,x=110, y=590)
#    time_entry1=Entry(Reg, bg="black",fg="white",font="Leelawadee 14")
#    time_entry1.place(height=25,width=210,x=110, y=620)

    # graph 2
        accurary_label2=Label(Reg, bg="orange",text="Validation Accuracy : ",fg="black",font="Leelawadee 10")
        accurary_label2.place(height=25,width=70,x=350, y=560)
        loss_label2=Label(Reg, bg="orange",text="Validation Loss : ",fg="black",font="Leelawadee 10")
        loss_label2.place(height=25,width=70,x=350, y=590)
#        time_label2=Label(Reg, bg="orange",text="Time : ",fg="black",font="Leelawadee 10")
#        time_label2.place(height=25,width=70,x=350, y=620)
    
        accurary_entry2=Entry(Reg, bg="black",fg="white",font="Leelawadee 14")
        accurary_entry2.place(height=25,width=210,x=440, y=560)
        loss_entry2=Entry(Reg, bg="black",fg="white",font="Leelawadee 14")
        loss_entry2.place(height=25,width=210,x=440, y=590)
#        time_entry2=Entry(Reg, bg="black",fg="white",font="Leelawadee 14")
#        time_entry2.place(height=25,width=210,x=440, y=620)
        with open(path3) as fp:
            Lines = fp.readlines()
    
        accurary_entry1.insert(0,Lines[0][:-1])
        loss_entry1.insert(0,Lines[1][:-1])

        accurary_entry2.insert(0,Lines[2][:-1])
        loss_entry2.insert(0,Lines[3][:-1])
    footer=Label(Reg, bg="white",text="Document Classification version 1.0",fg="black",font="Leelawadee 14")
    footer.place(height=36,width=420,x=10, y=657)

       
    submit.pack()
