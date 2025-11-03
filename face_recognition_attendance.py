
import cv2
import face_recognition
import pickle
import os
import pandas as pd
import numpy as np
import datetime


# Register a new student

def register_student():
    student_name = input("Enter student name: ").strip()
    if not student_name:
        print(" Student name cannot be empty.")
        return

    # Load existing faces if available
    known_faces = {}
    if os.path.exists('known_faces.pkl'):
        with open('known_faces.pkl', 'rb') as f:
            known_faces = pickle.load(f)

    if student_name in known_faces:
        print(f" Student '{student_name}' is already registered.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print(f" Registering {student_name} — Press 'C' to capture image, 'Q' to quit.")
    images_to_capture = 5
    captured_encodings = []
    count = 0

    while count < images_to_capture:
        ret, frame = cap.read()
        if not ret:
            print(" Error reading from webcam.")
            break

        # Display the frame
        cv2.imshow("Registration - Press 'C' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                captured_encodings.extend(face_encodings)
                count += 1
                print(f" Captured {count}/{images_to_capture} face(s).")
            else:
                print(" No face detected, try again.")

        elif key == ord('q'):
            print("Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_encodings:
        known_faces[student_name] = captured_encodings
        with open('known_faces.pkl', 'wb') as f:
            pickle.dump(known_faces, f)
        print(f"Registration complete for {student_name} ({len(captured_encodings)} encodings saved).")
    else:
        print(" No face data captured. Registration failed.")



# Recognize faces and mark attendance
def start_attendance():
    if not os.path.exists('known_faces.pkl'):
        print(" No registered faces found. Please register students first.")
        return

    with open('known_faces.pkl', 'rb') as f:
        known_faces = pickle.load(f)

    known_face_encodings = []
    known_face_names = []

    for name, encodings in known_faces.items():
        known_face_encodings.extend(encodings)
        known_face_names.extend([name] * len(encodings))

    attendance_df = pd.DataFrame(columns=['Name', 'Timestamp'])
    attended_this_session = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: Could not open webcam.")
        return

    print(" Starting attendance — Press 'Q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Error reading frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]

            # Draw face box
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Mark attendance
            if name != "Unknown" and name not in attended_this_session:
                timestamp = datetime.datetime.now()
                new_record = pd.DataFrame([{'Name': name, 'Timestamp': timestamp}])
                attendance_df = pd.concat([attendance_df, new_record], ignore_index=True)
                attended_this_session.add(name)
                print(f"🕒 Attendance logged for {name} at {timestamp}")

        cv2.imshow("Attendance - Press 'Q' to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    attendance_df.to_csv('attendance.csv', index=False)
    print("Attendance saved to attendance.csv")
    print(" Session ended.")



# Main menu

def main():
    while True:
        print("\n=== FACE RECOGNITION ATTENDANCE SYSTEM ===")
        print("1. Register new student")
        print("2. Start attendance")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            register_student()
        elif choice == '2':
            start_attendance()
        elif choice == '3':
            print("👋 Exiting... Goodbye!")
            break
        else:
            print("⚠️ Invalid input. Please enter 1, 2, or 3.")

# Run the main interface
if __name__ == "__main__":
    main()
