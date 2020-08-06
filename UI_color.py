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
from PIL import ImageTk
images = []
standard_pos=[0,0]
Filename=11
root = Tk()
root.geometry('400x400')
cv = Canvas(root,bg = 'white',width=400,height=400)
mouse_point=Canvas(root,bg = '#D0D0D0',width=10,height=10)
image=PIL.Image.new("RGB",(400,300),(255,255,255))
draw=PIL.ImageDraw.Draw(image)
sys.path.insert(0, "../lib")
sys.path.insert(1, "../lib/x86")
import Leap, sys, thread, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture
import math
def draw_canvas(x,y,Drawcolor):
    print str(x)+' powition '+str(y)
    if y>=105:
	    cv.create_oval(x,y,x,y,width=10,fill=Drawcolor,outline=Drawcolor)
    draw.ellipse((x, y-100, x+10, y-90), fill=Drawcolor, width=10)
def open_canvas():
    cv.pack()
    l1 = Tkinter.Button(cv,bd=2, bg='#ACD6FF',text='Save me', font=('Arial', 13), width=10, height=5,command = save)
    l1.place(x=0,y=0)
    l2 = Tkinter.Button(cv,bd=2, bg='#FF7575',text='Clear', font=('Arial', 13), width=10, height=5,command =delete)
    l2.place(x=100,y=0)
    l3 = Tkinter.Button(cv,bd=2, bg='#A6FFA6',text='Predict', font=('Arial', 13), width=10, height=5)
    l3.place(x=200,y=0)
    #l3.configure(bg='#00A000')
    #B = Tkinter.Button(root, text ="Save me", command = save)
    #B.place(x=100,y=100)
    mouse_point.place(x=20,y=200)
    root.mainloop()
def delete():
    Drawcolor='red'
    cv.delete("all")
    draw.rectangle((0, 0, 600, 600), fill=(255, 255, 255))
def save():
    print "77\n\n\n77\n7777\n77777\n7777"
    global Filename
    filename = "./data/overlap2/"+str(Filename)+'.jpg'
    image.save(filename)
    #image.show()
    Filename+=1
    delete()
def ClickEvent(x,y):
    if (y<=100 and y>=0):
        if (x>=0 and x<=100):
		    save()
        elif (x>=100 and x<=200):
            delete()
        elif (x>=200 and x<=300):
            l3.configure(bg='#00A000')
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
    width_of_window = 10
    aveAmplitude = 10
    threshold = 0.7
    def on_init(self, controller):
        self.Drawcolor='red'
        self.onclick=0
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
        
        print "Frame_latest id: %d, Frame_prev id: %d, hands: %d, fingers: %d " % (
                frame.id, previous.id, len(frame.hands), len(previous.fingers))
        
        if len(frame.hands)>0:
            #implement gesture spotting algorithm
            for _ in range(self.width_of_window):
                self.aveAmplitude = compute_amplitude(finger= frame.fingers, finger_prev= previous.fingers)
            self.aveAmplitude = self.aveAmplitude/self.width_of_window 
            print 'Average Amplitude:',self.aveAmplitude 

            if True:
                # Get hands
                print('##################Gesture Recording started#####################')
                for hand in frame.hands:
                    if hand.is_right:
                        print 'Right Hand:', hand.is_right
                        # Get fingers
                        fingerset=[]
                        for finger in hand.fingers:
                            fingerset.append(finger)

                        #print " thumb tip : %s, index tip: %s " % (
                            #fingerset[0].bone(3).prev_joint,
                            #fingerset[1].bone(3).prev_joint,)
                        print(self.Drawcolor)
                        thumb=fingerset[0]
                        index_finger=fingerset[1]
                        touch_distance=_get_eucledian_distance(thumb.bone(3).prev_joint,index_finger.bone(3).prev_joint)
                        print 'Touch distance between Thumb and Index:',touch_distance
                        ##################################
                        hand_x=hand.palm_position[0]
                        hand_y=hand.palm_position[1]
                        mouse_point.place(x=(150+hand_x),y=(450-hand_y))
                        if(touch_distance<((thumb.width+index_finger.width)/2+30)):
                            self.onclick=1
                            #X=index_finger.bone(3).prev_joint[0]
                            #Y=index_finger.bone(3).prev_joint[1]
                            draw_canvas(150+hand_x,450-hand_y,self.Drawcolor)
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
                            if self.onclick==1:
							    self.Drawcolor='green' if self.Drawcolor=='red' else 'red'
							    ClickEvent(150+hand_x,450-hand_y)
                                #click event
							    self.onclick=0
            else:
                print '################Gesture Recording Stopped######################'    
        else:
            print '################Gesture Recording did not Start######################'

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
