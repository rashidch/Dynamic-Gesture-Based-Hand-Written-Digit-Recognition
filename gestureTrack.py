'''
Author : Rashid Ali and sherman
'''
import sys
import time
import math
import Tkinter
import tkMessageBox
import PIL.ImageDraw
from Tkinter import *
import Leap
import sys
import thread
import time
import math

# from predict import process_image, predict_single_image
from predict_multi_digit_shu import predict_multi_image
# standard_pos = [0, 0]
Filename = 0
root = Tk()
root.geometry('400x400')
cv = Canvas(root, bg='white', width=400, height=300)
image = PIL.Image.new("RGB", (400, 300), (255, 255, 255))
draw = PIL.ImageDraw.Draw(image)
sys.path.insert(0, "../lib")
sys.path.insert(1, "../lib/x86")


def draw_canvas(x, y):
    print str(x)+' position'+str(y)
    cv.create_oval(x, y, x, y, width=10)
    draw.ellipse((x, y, x+10, y+10), fill=(0, 0, 0), width=10)
    cv.pack()


def open_canvas():
    B = Tkinter.Button(root, text="Save Me", command=save)
    B.pack()
    B2 = Tkinter.Button(root, text="Clear", command=delete)
    B2.pack()
    B3 = Tkinter.Button(root, text="Predict", command=prediction_multi)
    l = Tkinter.Label(root, text='predict result:\n',
                      font=('Arial', 13), width=15, height=5)
    l.place(x=10, y=-20)
    B3.pack()
    cv.pack()
    root.mainloop()


def delete():
    cv.delete("all")
    draw.rectangle((-100, -100, 600, 600), fill=(255, 255, 255))


def prediction_multi():
    image2 = image.copy()
    prediction = predict_multi_image(image2)
    i, j = prediction[0][0], prediction[0][1]
    i = int(i)
    j = int(j)
    pred = i if j == 10 else i*10+j
    l = Tkinter.Label(root, text='predict result:\n'+str(pred),
                      font=('Arial', 13), width=15, height=5)
    l.place(x=10, y=-20)


def save():
    print "77\n\n\n77\n7777\n77777\n7777"
    global Filename
    filename = "./data/6/"+str(Filename)+'.jpg'
    image.save(filename)
    image.show()
    Filename += 1


def _get_eucledian_distance(vect1, vect2):
    Distance = 0
    for i in range(3):
        Distance += ((vect1[i] - vect2[i])*(vect1[i] - vect2[i]))
    dist = math.sqrt(Distance)
    return dist


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    threshold = 0.1

    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):
        # Get the most recent frame and previous frame
        frame = controller.frame()
        lastFrame = controller.frame(2)

        #Frame is not empty
        if len(frame.hands) > 0:
            # Get hands
            for hand in frame.hands:
                if hand.is_right:
                    # rotation = math.sqrt(math.pow(hand.rotation_angle(lastFrame, Leap.Vector.x_axis),
                    #                              2) + math.pow(hand.rotation_angle(lastFrame, Leap.Vector.y_axis), 2))
                    palm_speed = math.sqrt(
                        math.pow(hand.palm_velocity[0], 2) + math.pow(hand.palm_velocity[1], 2))
                    if palm_speed > 50:
                        print('.... Recording .... \n')
                        print('Hand Palm Velocity', palm_speed)

                        # Get fingers
                        fingerset = []
                        for finger in hand.fingers:
                            fingerset.append(finger)

                        thumb = fingerset[0]
                        index_finger = fingerset[1]
                        touch_distance = _get_eucledian_distance(
                            thumb.bone(3).prev_joint, index_finger.bone(3).prev_joint)

                        if(touch_distance < ((thumb.width+index_finger.width)/2+30)):
                            # X=index_finger.bone(3).prev_joint[0]
                            # Y=index_finger.bone(3).prev_joint[1]
                            hand_x = hand.palm_position[0]
                            hand_y = hand.palm_position[1]
                            print("Coordinates:", hand_x, hand_y)
                            try:
                                thread.start_new_thread(
                                    draw_canvas, (hand_x+150, 350-hand_y))
                            except:
                                continue
                    else:
                        print('.... Recording Stopped ....\n')
        else:
            print('.... No Hand detecting ....\n')

        if not (frame.hands.is_empty and frame.gestures().is_empty):
            print ""


def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()
    thread.start_new_thread(open_canvas, ())
    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
