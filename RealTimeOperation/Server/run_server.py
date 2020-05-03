import sys
import cv2
import time
import socket
import numpy as np
import tensorflow as tf

from server_utils import load_detection_model
from server_utils import get_all_face_box, find_face_box_to_track, track_face_box, get_command, visualize_box
from server_utils import recv_image, send_image, send_command_idx


def main():
    if len(sys.argv) != 2:
        print("Usage : run_server.py [PORT]")
        return

    # Client's ip address, port number
    client_ip = ""
    client_port = int(sys.argv[1])

    # Initialize server socket (TCP)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((client_ip, client_port))

    # Wait for client's connection
    sock.listen(1)

    # Accept client's connection
    conn, addr = sock.accept()

    # Load face detector
    detection_model_path = './models/face_detection_model'
    face_detector = load_detection_model(detection_model_path)

    # Load hand sign classifier
    classification_model_path = './models/hand_sign_simple_cnn.h5'
    hand_sign_classifier = tf.keras.models.load_model(classification_model_path)
    hand_sign_classes = ["0_front", "1_back", "1_front", "2_back", "2_front", "5_front", "ILU"]

    while True:
        try:
            frame = recv_image(conn)
            # Detect all faces
            face_boxes = get_all_face_box(face_detector, frame, 10)

            if len(face_boxes) != 0:
                # Find face with hand sign 5 in the right area
                target_face_box = find_face_box_to_track(hand_sign_classifier, frame, face_boxes)

                if target_face_box is not None:
                    # Start command mode for 3 seconds
                    send_command_idx(conn, 5)

                    start_wait = time.time()
                    commands = []
                    while(time.time() - start_wait < 3):
                        send_image(conn, frame)
                        frame = recv_image(conn)

                        # Face tracking
                        face_boxes = get_all_face_box(face_detector, frame, 10)

                        if len(face_boxes) == 0:
                            continue
                        tracked_face_box = track_face_box(target_face_box, face_boxes)
                        if tracked_face_box is None:
                            continue

                        # Get command sign
                        command_result = get_command(hand_sign_classifier, frame, tracked_face_box)
                        if command_result is not None:
                            hand_area = command_result[0]
                            class_idx = command_result[1]
                            
                            # Visualize command box
                            if class_idx == 5:
                                visualize_box(frame, hand_area, hand_sign_classes[class_idx], 'LightGray')
                            else:
                                commands.append(class_idx)
                                visualize_box(frame, hand_area, hand_sign_classes[class_idx], 'Green')
                    
                    # If the number of recognized command signs is greater than 5, send the most frequent command
                    if len(commands) > 5:
                        command_idx = max(set(commands), key = commands.count)
                        send_command_idx(conn, command_idx)

            send_image(conn, frame)

        except:
            # Close all socket
            print("Client socket closed")
            conn.close()
            # sock.close()
            break


if __name__ == '__main__':
    # execute only if run as a script
    main()
