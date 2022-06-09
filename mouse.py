# Imports
import cv2 
import numpy as np 
import HandTrackingModule as htm 
from screeninfo import get_monitors
import autopy 

# Video Capture Size
width = 1280
height = 720
capture = cv2.VideoCapture(0)
capture.set(3,width)
capture.set(4,height)
# Area frame for using mouse 
mouse_frame = 100
# Get the size of the screen 
monitor = get_monitors()[0]
screen_width , screen_height =  monitor.width, monitor.height
# Constants for smoothening the mouse motion 
smooth_const = 6
prev_loc_x, prev_loc_y = 0 , 0 
curr_loc_x, curr_loc_y = 0 , 0 
# Initialize the Hand Detector 
detector = htm.HandDetector(detection_con=0.8)

while True :
	# Capture the camera feed 
	_,img = capture.read()
	# Flip the image to mirror the user 
	img = cv2.flip(img,1)
	# Get the height and width of the capture screen
	height,width,_ = img.shape
	# Find all hands in the view 
	all_hands = detector.find_hands(img,draw=False)
	# If a hand is found 
	if all_hands:
		# Get the hand values 
		hand_one = all_hands[0]
		# Get the landmark values from hand values 
		hand_lms = hand_one["lms"]

		# Draw a rectangle frame for mouse frame 
		# This represents the area screen 
		# so the corners of the frame will equivalent to the screen corners
		cv2.rectangle(img,(mouse_frame,mouse_frame),(width- mouse_frame,height-mouse_frame),(255,0,255),5)

		# Find the fingers that are up 
		fingers_up = detector.find_fingers_up(hand_one)
		# If index finger is up and middle finger is down 
		# Mouse is in moving mode
		if fingers_up[1] == 1 and fingers_up[2] == 0 :
			# Get the landmarks for the index finger 
			index_finger = hand_lms[8]
			# print(index_finger)
			# Convert the values to match the screen coordinates 
			x = np.interp(index_finger[0],[mouse_frame,width-mouse_frame],[0,screen_width])
			y = np.interp(index_finger[1],[mouse_frame,height-mouse_frame],[0,screen_height])
			# Draw a circle on the tip of the index finger 
			# To let the user know the mouse is in moving mode 
			cv2.circle(img,(index_finger[0],index_finger[1]),15,(0,255,0),cv2.FILLED)	
			try :
				# Move the mouse to the new coordinates
				curr_loc_x = (prev_loc_x) + (x-prev_loc_x) / smooth_const
				curr_loc_y = (prev_loc_y) + (y-prev_loc_y) / smooth_const
				autopy.mouse.move(int(curr_loc_x),int(curr_loc_y))
			except:
				print("Points Out of Bounds")
			prev_loc_x, prev_loc_y = curr_loc_x, curr_loc_y
		# If index finger and middle finger both are up 
		# Mouse is in clicking mode 
		elif fingers_up[1] == 1 and fingers_up[2] == 1 :
			# Get landmarks for index and middle finger 
			index_finger = hand_lms[8]
			middle_finger = hand_lms[12]
			# Get the distance between index and middle finger 
			distance = detector.find_distance(index_finger,middle_finger,img,draw=True)
			# If the distance is less than a threshold value in this case 30 
			# its a mouse click 
			if distance < 30 :
				autopy.mouse.click()

	# Display updated image to the user 
	cv2.imshow("Virtual Mouse",img)
	# Get the key pressed 
	key = cv2.waitKey(1)
	# if the key pressed is 's' 
	# then stop the program 
	if key == ord("s"):
		break
# Release the resources allocated 
capture.release()
cv2.destroyAllWindows()