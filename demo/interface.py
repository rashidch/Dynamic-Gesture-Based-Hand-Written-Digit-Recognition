"""
Author: Sherman and Rashid Ali
"""

import thread
import Leap
import sys
import os
import time
import math
import Tkinter
import tkMessageBox
import PIL.ImageDraw
import time
import json
from Tkinter import *
from prediction import predict_multi_image

standard_pos = [0, 0]
Filename = 0
root = Tk()
root.title("Smart Elevator Interface")
root.geometry("400x400")
cv = Canvas(root, bg="white", width=400, height=400)
l3 = Tkinter.Button(cv, bd=2, bg="#A6FFA6", text="Predict", font=("Arial", 13), width=21, height=5)
image = PIL.Image.new("RGB", (400, 300), (255, 255, 255))
mouse_point = Canvas(root, bg="#D0D0D0", width=5, height=5)
draw = PIL.ImageDraw.Draw(image)
sys.path.insert(0, "../lib")
sys.path.insert(1, "../lib/x86")


def draw_canvas(x, y):
    print(str(x) + " powition " + str(y))
    if y >= 105:
        cv.create_oval(x, y, x, y, width=10)
    draw.ellipse((x, y - 100, x + 10, y - 90), fill=(0, 0, 0), width=10)


def open_canvas():
    cv.pack()
    l1 = Tkinter.Button(cv, bd=2, bg="#ACD6FF", text="Save me", font=("Arial", 13), width=10, height=5, command=save)
    l1.place(x=0, y=0)
    l2 = Tkinter.Button(cv, bd=2, bg="#FF7575", text="Clear", font=("Arial", 13), width=10, height=5, command=delete)
    l2.place(x=100, y=0)
    l3.place(x=200, y=0)
    l3.configure(command=prediction_multi)
    mouse_point.place(x=20, y=200)
    root.mainloop()


def delete():
    cv.create_rectangle(0, 0, 500, 500, fill="white")
    draw.rectangle((-100, -100, 600, 600), fill=(255, 255, 255))


def prediction_multi():
    image2 = image.copy()
    tStart = time.time()
    prediction = predict_multi_image(image2)
    tEnd = time.time()
    i, j = prediction[0][0], prediction[0][1]
    i = int(i)
    j = int(j)
    pred = i if j == 10 else i * 10 + j
    pred = "B" + str(j) if i == 0 else pred
    l3.configure(text="predict result:\n" + str(pred) + "\nTime:" + "%.3f" % (tEnd - tStart))


def save():
    print("77\n\n\n77\n7777\n77777\n7777")
    global Filename
    filename = "./data/test/" + str(Filename) + ".jpg"
    image.save(filename)
    # image.show()
    Filename += 1


def ClickEvent(x, y):
    if y <= 100 and y >= 0:
        if x >= 0 and x <= 100:
            save()
        elif x >= 100 and x <= 200:
            delete()
        elif x >= 200 and x <= 300:
            prediction_multi()


def _get_eucledian_distance(vect1, vect2):
    Distance = 0
    for i in range(3):
        Distance += (vect1[i] - vect2[i]) * (vect1[i] - vect2[i])
    dist = math.sqrt(Distance)
    return dist


frameRecord = dict()
RecordList = list()


class ElevatorListener(Leap.Listener):
    def __init__(self):
        super(ElevatorListener, self).__init__()
        self.frames = []

    def on_init(self, controller):
        self.onclick = 0
        print("Initialized")

    def on_connect(self, controller):
        print("Connected")

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("Disconnected")

    def on_exit(self, controller):
        print("Exited")

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame_data = {}
        frame = controller.frame()
        if len(frame.hands) > 0:

            # Get hands
            for hand in frame.hands:
                if hand.is_right:

                    # Get Thumb and Index fingers data
                    fingerset = []
                    for finger in hand.fingers:
                        fingerset.append(finger)
                    thumb = fingerset[0]
                    index_finger = fingerset[1]

                    # get touch distance between thumb and index finger
                    touch_distance = _get_eucledian_distance(thumb.bone(3).prev_joint, index_finger.bone(3).prev_joint)

                    # get 2D position of palm
                    hand_x = hand.palm_position[0]
                    hand_y = hand.palm_position[1]

                    mouse_point.place(x=(150 + hand_x), y=(450 - hand_y))
                    if touch_distance < ((thumb.width + index_finger.width) / 2 + 25):
                        self.onclick = 1
                        draw_canvas(hand_x + 150, 450 - hand_y)

                        # add palm postion to hand_data
                        hand_data["palm_position"]["x"] = hand_x
                        hand_data["palm_position"]["y"] = hand_y
                    else:

                        if self.onclick == 1:
                            ClickEvent(150 + hand_x, 450 - hand_y)  # click event
                            self.onclick = 0
                        hand_data["palm_position"]["x"] = None
                        hand_data["palm_position"]["y"] = None

                    # add hand_data in fram_data dict
                    frame_data["hand"] = hand_data
                    self.frames.append(frame_data)
        else:
            print("")
            print("fps", frame.current_frames_per_second)

        if not (frame.hands.is_empty and frame.gestures().is_empty):
            print("")

    def get_data(self):

        return self.frames

    def Save_json(self):
        path = os.getcwd()
        root_dir = os.path.join(path, "production/")
        with open(root_dir + "frame.json", "w") as file:
            json.dump(self.frames, file)


def main():
    # Create a Elevator listener and controller
    listener = ElevatorListener()
    controller = Leap.Controller()
    thread.start_new_thread(open_canvas, ())
    # Have the Elevator listener receive events from the controller
    controller.add_listener(listener)
    # Keep this process running until Enter is pressed
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
