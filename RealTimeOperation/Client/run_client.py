import sys
import cv2
import time
import socket
import numpy as np

from client_utils import AirConditioner
from client_utils import recv_data, recv_command_index, recv_image, send_image


def main():
    if len(sys.argv) != 3:
        print("Usage : run_client.py [IP] [PORT]")
        return    

    # Server's ip address, port number
    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
    
    # Initialize client socket (TCP)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect to server
    sock.connect((server_ip, server_port))

    # Initialize webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    cv2.namedWindow('Screen', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Screen', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Initialize AirConditioner object
    air_conditioner = AirConditioner()
    air_conditioner.update_display_panel()

    while True:
        ret, frame = cap.read()

        # Encode webcam feed image and send to server
        send_image(sock, frame)

        # If the length of data is 1, the command index and image are received sequentially
        length = int(recv_data(sock, 16))
        if length == 1:
            # Receive command index
            command_index = recv_command_index(sock, length)

            # Receive image
            length = int(recv_data(sock, 16))
            frame = recv_image(sock, length)

            # Operate airconditioner
            air_conditioner.operate(command_index)
            air_conditioner.update_display_panel()
        else:
            # If not, only the image is received
            frame = recv_image(sock, length)


        result = np.concatenate((air_conditioner.get_display_panel(), frame), axis=1)
        result = cv2.resize(result, (1024, 600), cv2.INTER_CUBIC)
        cv2.imshow("Screen", result)

        # Press ESC to exit
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    sock.close()

if __name__ == '__main__':
    # execute only if run as a script
    main()
