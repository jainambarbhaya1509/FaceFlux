import os
from tkinter import simpledialog
import cv2
import numpy as np
import insightface
import sqlite3
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import io

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class FaceVerificationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Verification System")
        self.root.geometry("1200x700")
        
        # Initialize InsightFace model
        print("🚀 Initializing InsightFace...")
        self.model = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)
        
        # Initialize database
        self.init_database()
        
        # Camera variables
        self.camera = None
        self.is_camera_running = False
        self.verification_mode = False
        self.registration_mode = False
        self.current_frame = None
        
        # Create UI
        self.create_ui()
        
    def init_database(self):
        """Initialize SQLite database for storing face embeddings"""
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
        conn.close()
        print("✅ Database initialized")
        
    def create_ui(self):
        """Create the main UI components"""
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed and controls
        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Camera feed
        self.camera_label = ctk.CTkLabel(self.left_panel, text="Camera Feed")
        self.camera_label.pack(pady=5)
        
        self.camera_frame = ctk.CTkFrame(self.left_panel, width=640, height=480)
        self.camera_frame.pack(pady=10)
        
        self.video_label = ctk.CTkLabel(self.camera_frame, text="Camera Off")
        self.video_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Camera controls
        self.camera_controls = ctk.CTkFrame(self.left_panel)
        self.camera_controls.pack(fill=tk.X, pady=10)
        
        self.camera_button = ctk.CTkButton(
            self.camera_controls, 
            text="Start Camera", 
            command=self.toggle_camera
        )
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        self.register_button = ctk.CTkButton(
            self.camera_controls, 
            text="Register Face", 
            command=self.start_registration_mode,
            state="disabled"
        )
        self.register_button.pack(side=tk.LEFT, padx=5)
        
        self.verify_button = ctk.CTkButton(
            self.camera_controls, 
            text="Start Verification", 
            command=self.toggle_verification_mode,
            state="disabled"
        )
        self.verify_button.pack(side=tk.LEFT, padx=5)
        
        self.upload_button = ctk.CTkButton(
            self.camera_controls, 
            text="Upload Image", 
            command=self.upload_image
        )
        self.upload_button.pack(side=tk.LEFT, padx=5)
        
        # Registration form
        self.registration_frame = ctk.CTkFrame(self.left_panel)
        self.registration_frame.pack(fill=tk.X, pady=10)
        
        self.name_label = ctk.CTkLabel(self.registration_frame, text="Name:")
        self.name_label.pack(side=tk.LEFT, padx=5)
        
        self.name_entry = ctk.CTkEntry(self.registration_frame, width=200)
        self.name_entry.pack(side=tk.LEFT, padx=5)
        
        self.capture_button = ctk.CTkButton(
            self.registration_frame, 
            text="Capture & Save", 
            command=self.capture_and_save,
            state="disabled"
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(self.left_panel, text="Ready", height=30)
        self.status_bar.pack(fill=tk.X, pady=5)
        
        # Right panel - Database entries
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.db_label = ctk.CTkLabel(self.right_panel, text="Registered Faces")
        self.db_label.pack(pady=5)
        
        # Search box
        self.search_frame = ctk.CTkFrame(self.right_panel)
        self.search_frame.pack(fill=tk.X, pady=5)
        
        self.search_label = ctk.CTkLabel(self.search_frame, text="Search:")
        self.search_label.pack(side=tk.LEFT, padx=5)
        
        self.search_entry = ctk.CTkEntry(self.search_frame, width=150)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        
        self.search_button = ctk.CTkButton(
            self.search_frame, 
            text="Search", 
            command=self.search_database
        )
        self.search_button.pack(side=tk.LEFT, padx=5)
        
        self.refresh_button = ctk.CTkButton(
            self.search_frame, 
            text="Refresh", 
            command=self.load_database_entries
        )
        self.refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Database entries scrollable frame
        self.db_container = ctk.CTkScrollableFrame(self.right_panel, width=400, height=500)
        self.db_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Load database entries
        self.load_database_entries()
        
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_camera_running:
            self.stop_camera()
            self.camera_button.configure(text="Start Camera")
            self.register_button.configure(state="disabled")
            self.verify_button.configure(state="disabled")
            self.capture_button.configure(state="disabled")
            self.video_label.configure(text="Camera Off")
        else:
            self.start_camera()
            self.camera_button.configure(text="Stop Camera")
            self.register_button.configure(state="normal")
            self.verify_button.configure(state="normal")
            
    def start_camera(self):
        """Start the camera feed"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
            
        self.is_camera_running = True
        self.update_camera()
        self.set_status("Camera started")
        
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_camera_running = False
        self.verification_mode = False
        self.registration_mode = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.set_status("Camera stopped")
        
    def update_camera(self):
        """Update the camera feed"""
        if not self.is_camera_running:
            return
            
        ret, frame = self.camera.read()
        if not ret:
            self.set_status("Error reading from camera")
            return
            
        self.current_frame = frame.copy()
        
        # Process frame based on mode
        if self.verification_mode:
            self.process_verification(frame)
        elif self.registration_mode:
            self.process_registration(frame)
        else:
            # Just display the frame with face detection
            faces = self.model.get(frame)
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Convert to tkinter format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")
        
        # Schedule next update
        self.root.after(10, self.update_camera)
        
    def start_registration_mode(self):
        """Start face registration mode"""
        if not self.is_camera_running:
            messagebox.showerror("Error", "Camera is not running")
            return
            
        self.registration_mode = True
        self.verification_mode = False
        self.capture_button.configure(state="normal")
        self.set_status("Registration mode: Position face in frame and enter name")
        
    def process_registration(self, frame):
        """Process frame for registration"""
        faces = self.model.get(frame)
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(faces) > 1:
            cv2.putText(frame, "Multiple faces detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, "Face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    def capture_and_save(self):
        """Capture current frame and save face to database"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame to capture")
            return
            
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return
            
        frame = self.current_frame.copy()
        faces = self.model.get(frame)
        
        if len(faces) == 0:
            messagebox.showerror("Error", "No face detected")
            return
        elif len(faces) > 1:
            messagebox.showerror("Error", "Multiple faces detected. Please ensure only one face is in the frame.")
            return
            
        face = faces[0]
        embedding = face.embedding
        
        # Save face to database
        try:
            # Convert frame to JPEG for storage
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_bytes = img_encoded.tobytes()
            
            # Save to database
            conn = sqlite3.connect('face_database.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (name, embedding, image) VALUES (?, ?, ?)",
                (name, embedding.tobytes(), img_bytes)
            )
            conn.commit()
            conn.close()
            
            messagebox.showinfo("Success", f"Face for {name} registered successfully")
            self.name_entry.delete(0, tk.END)
            self.registration_mode = False
            self.capture_button.configure(state="disabled")
            self.load_database_entries()
            self.set_status("Face registered successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save face: {str(e)}")
            
    def toggle_verification_mode(self):
        """Toggle face verification mode"""
        if not self.is_camera_running:
            messagebox.showerror("Error", "Camera is not running")
            return
            
        self.verification_mode = not self.verification_mode
        self.registration_mode = False
        
        if self.verification_mode:
            self.verify_button.configure(text="Stop Verification")
            self.set_status("Verification mode: Looking for matches...")
        else:
            self.verify_button.configure(text="Start Verification")
            self.set_status("Verification mode stopped")
            
    def process_verification(self, frame):
        """Process frame for verification against database"""
        faces = self.model.get(frame)
        
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
            
        # Get all face embeddings from database
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, embedding FROM faces")
        db_faces = cursor.fetchall()
        conn.close()
        
        if not db_faces:
            cv2.putText(frame, "No faces in database", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return
            
        # Convert database embeddings to numpy arrays
        db_embeddings = []
        for face_id, name, embedding_bytes in db_faces:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            db_embeddings.append((face_id, name, embedding))
            
        # Check each detected face against database
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            
            # Find best match
            best_match = None
            best_similarity = 0
            threshold = 0.5  # Similarity threshold
            
            for face_id, name, db_embedding in db_embeddings:
                similarity = np.dot(embedding, db_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
                )
                
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = (face_id, name, similarity)
            
            # Draw rectangle around face
            if best_match:
                # Green for match
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_match[1]} ({best_match[2]:.2f})", 
                            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # Red for unknown
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
    def upload_image(self):
        """Upload an image file for face registration"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return
            
        try:
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not read the image file")
                return
                
            # Detect faces
            faces = self.model.get(image)
            if len(faces) == 0:
                messagebox.showerror("Error", "No face detected in the image")
                return
            elif len(faces) > 1:
                messagebox.showerror("Error", "Multiple faces detected. Please use an image with a single face.")
                return
                
            # Get name for registration
            name = self.name_entry.get().strip()
            if not name:
                name = simpledialog.askstring("Input", "Enter name for this face:")
                if not name or name.strip() == "":
                    return
                    
            face = faces[0]
            embedding = face.embedding
            
            # Save face to database
            _, img_encoded = cv2.imencode('.jpg', image)
            img_bytes = img_encoded.tobytes()
            
            conn = sqlite3.connect('face_database.db')
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO faces (name, embedding, image) VALUES (?, ?, ?)",
                (name, embedding.tobytes(), img_bytes)
            )
            conn.commit()
            conn.close()
            
            messagebox.showinfo("Success", f"Face for {name} registered successfully from uploaded image")
            self.load_database_entries()
            self.set_status(f"Face for {name} registered from image")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
            
    def load_database_entries(self, search_term=None):
        """Load and display database entries"""
        # Clear existing entries
        for widget in self.db_container.winfo_children():
            widget.destroy()
            
        # Get entries from database
        conn = sqlite3.connect('face_database.db')
        cursor = conn.cursor()
        
        if search_term:
            cursor.execute("SELECT id, name, image FROM faces WHERE name LIKE ?", (f"%{search_term}%",))
        else:
            cursor.execute("SELECT id, name, image FROM faces ORDER BY created_at DESC")
            
        entries = cursor.fetchall()
        conn.close()
        
        if not entries:
            no_entries_label = ctk.CTkLabel(self.db_container, text="No faces registered")
            no_entries_label.pack(pady=10)
            return
            
        # Display entries
        for face_id, name, img_bytes in entries:
            entry_frame = ctk.CTkFrame(self.db_container)
            entry_frame.pack(fill=tk.X, pady=5, padx=5)
            
            # Convert image bytes to PhotoImage
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image for display
            img = cv2.resize(img, (100, 100))
            photo = ImageTk.PhotoImage(image=Image.fromarray(img))
            
            # Keep reference to prevent garbage collection
            entry_frame.photo = photo
            
            # Image label
            img_label = ctk.CTkLabel(entry_frame, image=photo, text="")
            img_label.pack(side=tk.LEFT, padx=5)
            
            # Name and ID
            info_frame = ctk.CTkFrame(entry_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            name_label = ctk.CTkLabel(info_frame, text=f"Name: {name}")
            name_label.pack(anchor=tk.W, pady=2)
            
            id_label = ctk.CTkLabel(info_frame, text=f"ID: {face_id}")
            id_label.pack(anchor=tk.W, pady=2)
            
            # Delete button
            delete_button = ctk.CTkButton(
                entry_frame, 
                text="Delete", 
                command=lambda id=face_id: self.delete_entry(id),
                fg_color="red",
                hover_color="darkred",
                width=80
            )
            delete_button.pack(side=tk.RIGHT, padx=5)
            
    def delete_entry(self, face_id):
        """Delete a face entry from the database"""
        confirm = messagebox.askyesno("Confirm", "Are you sure you want to delete this face?")
        if not confirm:
            return
            
        try:
            conn = sqlite3.connect('face_database.db')
            cursor = conn.cursor()
            cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            conn.commit()
            conn.close()
            
            self.load_database_entries()
            self.set_status(f"Face ID {face_id} deleted")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete face: {str(e)}")
            
    def search_database(self):
        """Search database for faces by name"""
        search_term = self.search_entry.get().strip()
        self.load_database_entries(search_term)
        
    def set_status(self, message):
        """Update status bar message"""
        self.status_bar.configure(text=message)
        
    def on_closing(self):
        """Handle window closing"""
        if self.is_camera_running:
            self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    # Create the main window
    root = ctk.CTk()
    app = FaceVerificationApp(root)
    
    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start the main loop
    root.mainloop()