
'''
Author Sherman and Rashid Ali
'''

import thread
import cv2
import Leap
import sys
import os
import time
import math
import numpy as np
import tensorflow as tf
import Tkinter
import tkMessageBox
import PIL.ImageDraw
from Tkinter import *
from PIL import ImageTk
from binary_classifier import createModel
from predict_multi_digit_color import multiModel, subtract_mean
from seperate_overlaped_multidigits import color_based_overlap_seperation
print("Tensorflow version: " + tf.__version__)

images = []
standard_pos = [0, 0]
Filename = 20
root = Tk()
root.title('Smart Elevator Interface')
root.geometry('400x400')
cv = Canvas(root, bg='white', width=400, height=400)
l3 = Tkinter.Button(cv, bd=2, bg='#A6FFA6', text='Predict',
                    font=('Arial', 13), width=21, height=5)
mouse_point = Canvas(root, bg='#D0D0D0', width=3, height=3)
image = PIL.Image.new("RGB", (400, 300), (255, 255, 255))
draw = PIL.ImageDraw.Draw(image)
sys.path.insert(0, "../lib")
sys.path.insert(1, "../lib/x86")


def draw_canvas(x, y, Drawcolor):
    print str(x)+' powition '+str(y)
    if y >= 105:
        cv.create_oval(x, y, x, y, width=10, fill=Drawcolor, outline=Drawcolor)
    draw.ellipse((x, y-100, x+10, y-90), fill=Drawcolor, width=10)


def open_canvas():
    cv.pack()
    l1 = Tkinter.Button(cv, bd=2, bg='#ACD6FF', text='Save me', font=(
        'Arial', 13), width=10, height=5, command=save)
    l1.place(x=0, y=0)
    l2 = Tkinter.Button(cv, bd=2, bg='#FF7575', text='Clear', font=(
        'Arial', 13), width=10, height=5, command=delete)
    l2.place(x=100, y=0)
    # l3 = Tkinter.Button(cv, bd=2, bg='#A6FFA6', text='Predict',
    # font=('Arial', 13), width=10, height=5, )
    l3.place(x=200, y=0)
    l3.configure(command=prediction_multi)
    # l3.configure(bg='#00A000')
    #B = Tkinter.Button(root, text ="Save me", command = save)
    # B.place(x=100,y=100)
    mouse_point.place(x=20, y=200)
    root.mainloop()


def delete():
    Drawcolor = 'red'
    cv.delete("all")
    draw.rectangle((0, 0, 600, 600), fill=(255, 255, 255))


def save():
    if np.mean(np.array(image)) < 250:
        print "77\n\n\n77\n7777\n77777\n7777"
        global Filename
        filename = "overlap_dataset/overlap2/"+str(Filename)+'.jpg'
        image.save(filename)
        # image.show()
        Filename += 1
        delete()
    else:
        tkMessageBox.showwarning(
            title="Warning", message="Please write the correct floor number before save!")


def check_overlap(test):
    test = np.array(test)
    test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    test = cv2.resize(test, dsize=(64, 64), interpolation=cv2.INTER_AREA)
    test = test.reshape(1, 64, 64, 1)
    predict = evaluateModel(test)
    return predict[0]


def prediction_multi():
    image2 = image.copy()
    if np.mean(np.array(image2)) < 250:
        tStart = time.time()
        if check_overlap(image2) == 0:
            prediction = predict_multi_image(image2)
        elif check_overlap(image2) == 1:
            separated = color_based_overlap_seperation(image2)
            prediction = predict_multi_image(separated)
        tEnd = time.time()
        i, j = prediction[0][0], prediction[0][1]
        i = int(i)
        j = int(j)
        pred = i if j == 10 else i*10+j
        pred = 'B'+str(j) if i == 0 else pred
        l3.configure(text='predict result:\n'+str(pred) +
                     '\nTime:'+'%.3f' % (tEnd-tStart))
    else:
        tkMessageBox.showwarning(
            title="Warning", message="Please write the correct floor number!")


def ClickEvent(x, y):
    if (y <= 100 and y >= 0):
        if (x >= 0 and x <= 100):
            save()
        elif (x >= 100 and x <= 200):
            delete()
        elif (x >= 200 and x <= 300):
            l3.configure(bg='#00A000')


def _get_eucledian_distance(vect1, vect2):
    Distance = 0
    for i in range(3):
        Distance += ((vect1[i] - vect2[i])*(vect1[i] - vect2[i]))
    dist = math.sqrt(Distance)
    return dist


'''
    Function for creating tensorflow graph of binary classifier  
'''


def evaluateModel(test):
    # Helper functions for creating new variables
    tf.reset_default_graph()

    x, y_pred_cls, keep_prob1, keep_prob2 = createModel()

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    save_dir = './multi-digit-leap-motion/checkpoints/'

    # Create directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'binary_model_v2')

    saver.restore(sess=session, save_path=save_path)

    # Generate predictions for the testset
    prediction = session.run(
        y_pred_cls, {x: test, keep_prob1: 1.0, keep_prob2: 1.0})

    return prediction


'''
    Function for creating tensorflow graph of multi-digit model  
'''


def predict_multi_image(image, label=None):

    # Helper functions for creating new variables
    tf.reset_default_graph()
    x, y_pred_cls, p_keep_1, p_keep_2, p_keep_3 = multiModel()
    # Launch the graph in a session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    # session.run(tf.global_variables_initializer())
    # load checkpoints for multi-digit model
    saver = tf.train.Saver()
    save_path = os.path.join(
        './multi-digit-leap-motion/checkpoints', 'mlt_leap_v4')
    try:
        #print("Restoring checkpoint ...")
        # Try and load the data in the checkpoint.
        saver.restore(session, save_path=save_path)
        #print("Restored checkpoint from:", save_path.split('/')[-1])

    # If the above failed - initialize all the variables
    except:
        print("Failed to restore checkpoint - initializing variables")

    image = cv2.resize(np.float32(image), dsize=(
        64, 64), interpolation=cv2.INTER_AREA)
    image = subtract_mean(image)
    image = np.expand_dims(
        np.dot(image, [0.2989, 0.5870, 0.1140]), axis=3).astype(np.float32)
    img = np.expand_dims(image, axis=0)
    print('Shape of Final Image',  img.shape)
    pred = session.run(y_pred_cls, feed_dict={
                       x: img, p_keep_1: 1., p_keep_2: 1., p_keep_3: 1.})

    return pred


class SampleListener(Leap.Listener):

    def on_init(self, controller):
        self.Drawcolor = 'red'
        self.onclick = 0
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        if len(frame.hands) > 0:
            # Get hands
            for hand in frame.hands:
                if hand.is_right:
                    # Get fingers
                    fingerset = []
                    for finger in hand.fingers:
                        fingerset.append(finger)

                    print(self.Drawcolor)
                    thumb = fingerset[0]
                    index_finger = fingerset[1]
                    touch_distance = _get_eucledian_distance(
                        thumb.bone(3).prev_joint, index_finger.bone(3).prev_joint)

                    hand_x = hand.palm_position[0]
                    hand_y = hand.palm_position[1]
                    mouse_point.place(x=(150+hand_x), y=(450-hand_y))
                    if(touch_distance < ((thumb.width+index_finger.width)/2+20)):
                        self.onclick = 1
                        # X=index_finger.bone(3).prev_joint[0]
                        # Y=index_finger.bone(3).prev_joint[1]

                        draw_canvas(150+hand_x, 450-hand_y, self.Drawcolor)

                    else:
                        if self.onclick == 1:
                            self.Drawcolor = 'blue' if self.Drawcolor == 'red' else 'red'
                            ClickEvent(150+hand_x, 450-hand_y)
    # click event
                            self.onclick = 0
        else:
            print ''

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
