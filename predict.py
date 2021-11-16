# Importing Libraries
from tkinter import *
from tkinter import messagebox as ms
import tkinter as tk
import sqlite3
from PIL import ImageTk, Image
from run_predict import predict_on_text
from tkinter import filedialog
import time

filename = "aa"

def Predict(username):
        
    predict = tk.Tk()
    predict.title('Predict using system')
    predict.geometry('700x700+0+0')

    predict.resizable(width=False, height=False)
    global filename
    filename = "empty"
    def UploadAction(event=None):
        global filename
        filename = filedialog.askopenfilename()
    
    
    def Predict_pretrained(arg = None):

        
        
        if clicked.get() == username:
            name = username
            if os.path.exists(os.path.join(os.getcwd(),"saved_models",model_name,"pytorch_model.bin")):
                pass
            else:
                name = "Default"
                messagebox.showinfo(title="Model Not found", message="No custom model was found so using default model",parent=predict)
        else:
            name="Default"
            
        if filename.endswith(".txt"):
            systemprogress=Label(predict, text = 'System is in Progress......\n  Please wait', font=('Arial',12, 'bold'), bg='white', fg='green')
            systemprogress.place(height=70,width=350,x=250,y=320)
            systemprogress.update_idletasks()
            txt = open(filename,encoding="utf-8")
            
            ans = predict_on_text(name,txt.read())
            
            systemprogress.destroy()
            messagebox.showinfo(title="Author Name", message=ans,parent=predict)
        elif name_entry_text.get() != "":
            systemprogress=Label(predict, text = 'System is in Progress......\n  Please wait', font=('Arial',12, 'bold'), bg='white', fg='green')
            systemprogress.place(height=70,width=350,x=250,y=320)
            systemprogress.update_idletasks()
            txt = name_entry_text.get()
            
            ans =  predict_on_text(name,txt)
            
            systemprogress.destroy()
            messagebox.showinfo(title="Author Name", message=ans,parent=predict)
        else:
            
            ms.showerror('Oops',"please enter correct text or upload correct file")
            
    top_frame = Label(predict, text='Predict',font = ('Cosmic', 25, 'bold'), bg='#7268A6',relief='groove',padx=500, pady=30)
    top_frame.pack(side='top')
    
    frame = LabelFrame(predict, padx=30, pady=30, bg='white')
    frame.place(relx = 0.5, rely = 0.55, anchor = CENTER)
    
    clicked= StringVar(predict,"Select Model")
    clicked.set("Select Model")
    main_menu = OptionMenu(frame, clicked, "Default", username)
    main_menu.pack(pady=10)
    
    def predict_help():
        messagebox.showinfo("Enter text","Enter plain Hebrew text to classify. make sure its big enough to get accurate results",parent=frame)
        
    # -------------------  ADD IMAGE
    # opens the image
    img = Image.open("icon.png")
    # resize the image and apply a high-quality down sampling filter
    img = img.resize((25,25), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)
    # create a label
    panel3 = Button(frame, image = img,relief = "flat",command=predict_help)
    # set the image as img
    panel3.image = img
    panel3.place(height=25,width=25,x=0, y=10)

    def open_help():
        messagebox.showinfo("Input File","Input the file you want to classify in .txt format",parent=frame)
    # -------------------  ADD IMAGE
    # opens the image
    img1 = Image.open("icon.png")
    # resize the image and apply a high-quality down sampling filter
    img1 = img1.resize((25,25), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img1 = ImageTk.PhotoImage(img1)
    # create a label
    panel3 = Button(frame, image = img1,relief = "flat",command=open_help)
    # set the image as img
    panel3.image = img1
    panel3.place(height=25,width=25,x=0, y=40)

    def select_help():
        messagebox.showinfo("Model Selection","Please select which model to use, Default or your custom trained",parent=frame)
    # -------------------  ADD IMAGE
    # opens the image
    img2 = Image.open("icon.png")
    # resize the image and apply a high-quality down sampling filter
    img2 = img2.resize((25,25), Image.ANTIALIAS)
    # PhotoImage class is used to add image to widgets, icons etc
    img2 = ImageTk.PhotoImage(img2)
    # create a label
    panel3 = Button(frame, image = img2,relief = "flat",command=select_help)
    # set the image as img
    panel3.image = img2
    panel3.place(height=25,width=25,x=0, y=70)
    
    name = tk.Label(frame, text = 'Text', font=('Arial',12, 'bold'), bg='white', fg='green')

    name_entry_text = tk.Entry(frame ,font=('Arial',12,'normal'), bg='#FBB13C',justify = 'center')
    name_entry_text.bind('<Return>', Predict_pretrained)
    button = tk.Button(frame, text='Open', command=UploadAction,justify = 'center')
    button.pack()   
    
    submit=tk.Button(frame,text = 'Predict', command = Predict_pretrained, width="10",bd = '3',  font = ('Times', 12, 'bold'),bg='#581845', 	 fg='white',relief='groove', justify = 'center', pady='5'  ) 
    Quit = tk.Button(predict, text = "Quit", width="10", command = predict.destroy, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Quit.place(anchor ='sw',rely=1,relx=0.775)

    def back():
        predict.destroy()
        from train_predict import functionality
        functionality(username)

    Back = tk.Button(predict, text = "Back", width="10", command = back, bd = '3',  font = ('Times', 12, 'bold'), bg='black', fg='white',relief='groove', justify = 'center', pady='5')
    Back.place(anchor ='se',rely=1,relx=0.775)

    """footer=Label(predict, bg="white",text="Document Classification version 1.0",fg="black",font="Leelawadee 14")
    footer.place(height=36,width=480,x=120, y=657)"""
    
    name.pack()
    name_entry_text.focus_set()
    name_entry_text.pack()    
    submit.pack()
    

