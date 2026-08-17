import os
from datetime import datetime
import tkinter as tk
import cv2
import pandas as pd
from PIL import Image, ImageTk
import util

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("Face Recognition Attendance System")
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(
            self.main_window, "Login", "green", self.login
        )
        self.login_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, "Register New User", "gray", self.register_new_user, fg='black'
        )
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)

        self.add_webcam(self.webcam_label)

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def login(self):
        user_name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if user_name == 'no_persons_found':
            util.msg_box('Error', 'No face detected in webcam feed.')
        elif user_name == 'unknown_person':
            util.msg_box('Error', 'User not recognized. Please register first.')
        else:
            self.mark_attendance(user_name)
            util.msg_box('Success', f'Welcome back, {user_name}! Attendance marked.')

    def mark_attendance(self, name):
        attendance_file = "Attendance.csv"
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        if not os.path.exists(attendance_file):
            df = pd.DataFrame(columns=["Name", "Date", "Time"])
        else:
            df = pd.read_csv(attendance_file)

        already_marked = ((df['Name'] == name) & (df['Date'] == current_date)).any()
        if not already_marked:
            new_entry = pd.DataFrame([[name, current_date, current_time]], columns=["Name", "Date", "Time"])
            df = pd.concat([df, new_entry], ignore_index=True)
            df.to_csv(attendance_file, index=False)

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, "Accept", "green", self.accept_register_new_user
        )
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, "Try again", "red", self.try_again_register_new_user
        )
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, "Please input username:"
        )
        self.text_label_register_new_user.place(x=750, y=70)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        if not name:
            util.msg_box('Error', 'Please enter a username.')
            return

        cv2.imwrite(os.path.join(self.db_dir, f'{name}.jpg'), self.register_new_user_capture)
        util.msg_box('Success!', 'User was registered successfully!')
        self.register_new_user_window.destroy()

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()
