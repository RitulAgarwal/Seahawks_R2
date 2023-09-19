import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("plant_disease_model.h5")

def diagnose_image(file_path):
    if file_path:
        
        image = Image.open(file_path)
        image = image.resize((256, 256))
        image = np.array(image) / 255.0  
        image = image.reshape(1, 256, 256, 3)  

        
        img = ImageTk.PhotoImage(Image.open(file_path))
        image_label.config(image=img)
        image_label.image = img

        
        prediction = model.predict(image)
        class_index = np.argmax(prediction)
   
        class_labels = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy","Blueberry___healthy","Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy"]
        disease = class_labels[class_index]

       
        result_label.config(text=f"Diagnosis: {disease}")

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    diagnose_image(file_path)


app = tk.Tk()
app.title("Plant Disease Diagnosis")


screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()
app.geometry(f"{screen_width}x{screen_height}+0+0")


app.configure(bg="white")  


choose_file_button = tk.Button(app, text="Choose File", command=open_file_dialog, padx=20, pady=10)
choose_file_button.pack(pady=20)


image_label = tk.Label(app, bg="white")
image_label.pack()

result_label = tk.Label(app, text="", font=("Helvetica", 24), fg="green", bg="white")
result_label.pack()

app.mainloop()
