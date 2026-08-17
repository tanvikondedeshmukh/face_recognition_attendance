import os
import tkinter as tk
from tkinter import messagebox
import face_recognition

def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
        window,
        text=text,
        activebackground="black",
        activeforeground="white",
        fg=fg,
        bg=color,
        command=command,
        height=2,
        width=20,
        font=('Helvetica bold', 16)
    )
    return button

def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label

def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 18), justify="left")
    return label

def get_entry_text(window):
    inputtxt = tk.Text(window, height=2, width=15, font=("Arial", 20))
    return inputtxt

def msg_box(title, description):
    messagebox.showinfo(title, description)

def recognize(img, db_path):
    """Recognize a face from an image array against stored database photos."""
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    
    unknown_encoding = embeddings_unknown[0]
    
    if not os.path.exists(db_path):
        return 'unknown_person'

    db_files = [f for f in sorted(os.listdir(db_path)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for file_name in db_files:
        path_ = os.path.join(db_path, file_name)
        known_img = face_recognition.load_image_file(path_)
        known_encodings = face_recognition.face_encodings(known_img)
        
        if len(known_encodings) > 0:
            match = face_recognition.compare_faces([known_encodings[0]], unknown_encoding, tolerance=0.5)[0]
            if match:
                return os.path.splitext(file_name)[0]

    return 'unknown_person'
