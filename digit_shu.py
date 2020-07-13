################################################################################
# Copyright (C) 2012-2013 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################
import sys
import time
import math
import Tkinter
import tkMessageBox
import PIL.ImageDraw
from Tkinter import *
#from predict import process_image, predict_single_image
from predict_multi_digit_shu import predict_multi_image
standard_pos=[0,0]
Filename=0
root = Tk()
root.geometry('400x400')
cv = Canvas(root,bg = 'white',width=400,height=300)
image = PIL.Image.new("RGB",(400,300),(255,255,255))
draw=PIL.ImageDraw.Draw(image)
sys.path.insert(0, "../lib")
sys.path.insert(1, "../lib/x86")
import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import math
def draw_canvas(x,y):
    print str(x)+' powition '+str(y)
    cv.create_oval(x,y,x,y,width=10)
    draw.ellipse((x, y, x+10, y+10), fill=(0, 0, 0), width=10)
    cv.pack()
def open_canvas():
    B = Tkinter.Button(root, text ="Save Me", command = save)
    B.pack()
    B2 = Tkinter.Button(root, text ="Clear", command =delete)
    B2.pack()
    B3 = Tkinter.Button(root, text ="Predict", command=prediction_multi)
    l = Tkinter.Label(root, text='predict result:\n', font=('Arial', 13), width=15, height=5)
    l.place(x=10,y=-20)
    B3.pack()
    cv.pack()
    root.mainloop()
def delete():
    cv.delete("all")
    draw.rectangle((-100,-100, 600, 600), fill=(255, 255, 255))
def prediction():
    image2 = image.copy()
    image2 = process_image(image=image2)
    #image2 = thread.start_new_thread(process_image,image2)
    prediction = predict_single_image(image2)
    print("Prediction:", prediction)
    #image3 = image.copy()
    #draw2=PIL.ImageDraw.Draw(image3)
    l = Tkinter.Label(root, text='predict result:\n'+str(prediction), font=('Arial', 13), width=15, height=5)
    l.place(x=10,y=-20)
    #tkMessageBox.showinfo(title='predict result', message=str(prediction))
    #draw.text((100, 100),str(prediction), align ="right")
    #image.show()
def prediction_multi():
    image2 = image.copy()
    prediction=predict_multi_image(image2)
    i,j=prediction[0][0],prediction[0][1]
    i=int(i)
    j=int(j)
    pred=i if j==10 else i*10+j
    l = Tkinter.Label(root, text='predict result:\n'+str(pred), font=('Arial', 13), width=15, height=5)
    l.place(x=10,y=-20)
def save():
    print "77\n\n\n77\n7777\n77777\n7777"
    global Filename
    filename = "./data/6/"+str(Filename)+'.jpg'
    image.save(filename)
    image.show()
    Filename+=1


def _get_eucledian_distance(vect1, vect2):
    Distance=0
    for i in range(3):
        Distance += ((vect1[i] - vect2[i])*(vect1[i] - vect2[i]))
    dist = math.sqrt(Distance)
    return dist

def compute_amplitude(finger, finger_prev):
    amplitude = 0
    for i in range(2):
      amplitude += math.pow((finger[1].tip_position[i]-finger_prev[1].tip_position[i]),2)
    amplitude = math.sqrt(amplitude)  

    return amplitude


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    state_names = ['STATE_INVALID', 'STATE_START', 'STATE_UPDATE', 'STATE_END']
    width_of_window = 100
    aveAmplitude = 10
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
        # Get the most recent frame and report some basic information
        frame = controller.frame()
        previous = controller.frame(1)
        
        #print "Frame_latest id: %d, Frame_prev id: %d, hands: %d, fingers: %d " % (
        #        frame.id, previous.id, len(frame.hands), len(previous.fingers))
        
        if len(frame.hands)>0:
            #implement gesture spotting algorithm
            for _ in range(self.width_of_window):
                self.aveAmplitude = compute_amplitude(finger= frame.fingers, finger_prev= previous.fingers)
            self.aveAmplitude = self.aveAmplitude/self.width_of_window 
            #print 'Average Amplitude:',self.aveAmplitude 
            
            if True:
                # Get hands
                #print('##################Gesture Recording started#####################')
                for hand in frame.hands:
                    if hand.is_right:
                        #print 'Right Hand:', hand.is_right
                        # Get fingers
                        fingerset=[]
                        for finger in hand.fingers:
                            fingerset.append(finger)
                        '''
                        print " thumb tip : %s, index tip: %s " % (
                            fingerset[0].bone(3).prev_joint,
                            fingerset[1].bone(3).prev_joint,)
                        '''
                        thumb=fingerset[0]
                        index_finger=fingerset[1]
                        touch_distance=_get_eucledian_distance(thumb.bone(3).prev_joint,index_finger.bone(3).prev_joint)
                        #print 'Touch distance between Thumb and Index:',touch_distance
                        ##################################
                        
                        if(touch_distance<((thumb.width+index_finger.width)/2+15)):
                            #X=index_finger.bone(3).prev_joint[0]
                            #Y=index_finger.bone(3).prev_joint[1]
                            hand_x=hand.palm_position[0]
                            hand_y=hand.palm_position[1]
                            try:
                                thread.start_new_thread(draw_canvas,(hand_x+150,350-hand_y))
                            except:
                                continue
                            #print "draw start"
                            #print "    %s finger, id: %d, length: %fmm, width: %fmm" % (
                            #self.finger_names[fingerset[1].type],
                            #fingerset[1].id,
                            #fingerset[1].length,
                            #fingerset[1].width)

                        # Get bones
                            #bone = fingerset[1].bone(3)
                            #print "      Bone: %s, start: %s, end: %s, direction: %s" % (
                            #self.bone_names[bone.type],
                            #bone.prev_joint,
                            #bone.next_joint,
                            #bone.direction)
            else:
                #print '################Gesture Recording Stopped######################'
                print''    
        else:
            #print '################Gesture Recording did not Start######################'
            print''

        if not (frame.hands.is_empty and frame.gestures().is_empty):
            print ""

def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()
    thread.start_new_thread(open_canvas,())
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
