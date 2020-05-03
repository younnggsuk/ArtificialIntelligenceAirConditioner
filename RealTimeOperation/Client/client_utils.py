import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Font setting
icon_font_path = "./font/Font Awesome 5 Free-Solid-900.otf"
text_font_path = "./font/NanumGothic-ExtraBold.ttf"

icon_font = ImageFont.truetype(icon_font_path, 70)
fan_speed_font = ImageFont.truetype(icon_font_path, 50)
text_font = ImageFont.truetype(text_font_path, 60)

# Icon's unicode
fan_code = chr(0xf863)
temp_code = chr(0xf769)
rotation_code = chr(0xf2f1)
fan_speed_code = chr(0xf04b)
recognize_code = chr(0xf5dc)

class AirConditioner:
    # Air conditioner's status
    def __init__(self):
        self.power = False
        self.temp = 24
        self.fan_speed = 1
        self.rotation = False
        self.recognizing = False
        self.display_panel = np.zeros((480, 400, 3), np.uint8)

    
    # Get airconditioner's display panel to draw screen
    def get_display_panel(self):
        return self.display_panel


    # Operate air conditioner
    def operate(self, idx):
        if idx == 5:
            self.recognizing = True
            return
        else:
            self.recognizing = False

        if idx == 0:
            self.power = not self.power
        else:
            if self.power:
                if idx == 1:
                    self.temp = max(18, self.temp-1)
                elif idx == 2:
                    self.temp = min(28, self.temp+1)
                elif idx == 3:
                    self.fan_speed = max(1, self.fan_speed-1)
                elif idx == 4:
                    self.fan_speed = min(3, self.fan_speed+1)
                elif idx == 6:
                    self.rotation = not self.rotation


    # Display panel update
    def update_display_panel(self):
        self.display_panel = np.zeros((480, 400, 3), np.uint8)
        img_pil = Image.fromarray(self.display_panel)
        draw = ImageDraw.Draw(img_pil)

        # Recognizing icon
        if self.recognizing:
            draw.text((215, 340), recognize_code, font=icon_font, fill=(255, 255, 255, 0))
        else:
            draw.text((215, 340), recognize_code, font=icon_font, fill=(30, 30, 30, 0))

        # Power OFF        
        if not self.power:
            draw.text((85, 60), temp_code, font=icon_font, fill=(30, 30, 30, 0))
            draw.text((75, 200), fan_code, font=icon_font, fill=(30, 30, 30, 0))
            draw.text((175, 60), f"{self.temp}°C", font=text_font, fill=(30, 30, 30, 0))
            draw.text((175, 212), fan_speed_code, font=fan_speed_font, fill=(30, 30, 30, 0))
            draw.text((225, 212), fan_speed_code, font=fan_speed_font, fill=(30, 30, 30, 0))
            draw.text((275, 212), fan_speed_code, font=fan_speed_font, fill=(30, 30, 30, 0))
            draw.text((95, 340), rotation_code, font=icon_font, fill=(30, 30, 30, 0))            
        # Power ON
        else:
            # Temp, Fan icon
            draw.text((85, 60), temp_code, font=icon_font, fill=(255, 255, 255, 0))
            draw.text((75, 200), fan_code, font=icon_font, fill=(255, 255, 255, 0))
            
            # Temperature value
            draw.text((175, 60), f"{self.temp}°C", font=text_font, fill=(255, 255, 255, 0))
            
            # Fan speed icons
            if self.fan_speed == 1:
                draw.text((175, 212), fan_speed_code, font=fan_speed_font, fill=(255, 255, 255, 0))
                draw.text((225, 212), fan_speed_code, font=fan_speed_font, fill=(30, 30, 30, 0))
                draw.text((275, 212), fan_speed_code, font=fan_speed_font, fill=(30, 30, 30, 0))
            elif self.fan_speed == 2:
                draw.text((175, 212), fan_speed_code, font=fan_speed_font, fill=(255, 255, 255, 0))
                draw.text((225, 212), fan_speed_code, font=fan_speed_font, fill=(255, 255, 255, 0))
                draw.text((275, 212), fan_speed_code, font=fan_speed_font, fill=(30, 30, 30, 0))
            elif self.fan_speed == 3:
                draw.text((175, 212), fan_speed_code, font=fan_speed_font, fill=(255, 255, 255, 0))
                draw.text((225, 212), fan_speed_code, font=fan_speed_font, fill=(255, 255, 255, 0))
                draw.text((275, 212), fan_speed_code, font=fan_speed_font, fill=(255, 255, 255, 0))
            
            # Rotation icon
            if self.rotation:
                draw.text((95, 340), rotation_code, font=icon_font, fill=(255, 255, 255, 0))
            else:
                draw.text((95, 340), rotation_code, font=icon_font, fill=(30, 30, 30, 0))
        
        self.display_panel = np.array(img_pil)


'''
Receive server's command index
'''
def recv_command_index(sock, length):
    stringData = recv_data(sock, int(length))
    command_index = int(stringData.decode())

    return command_index


'''
Receive server's image
'''
def recv_image(sock, length):
    stringData = recv_data(sock, int(length))
    data = np.fromstring(stringData, dtype='uint8')
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return frame


'''
Send image to server
'''
def send_image(sock, frame):
    result, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    data = np.array(encoded_img)
    stringData = data.tostring()
    sock.sendall((str(len(stringData))).encode().ljust(16) + stringData)


'''
Read data from server's socket
'''
def recv_data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None

        buf += newbuf
        count -= len(newbuf)

    return buf