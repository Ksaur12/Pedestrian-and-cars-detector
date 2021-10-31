import cv2
from random import randrange

#loading the pre-trained data
trained_cars_data = cv2.CascadeClassifier('cars_haarcascade.xml')
trained_ped_data = cv2.CascadeClassifier('human_haarcascade.xml')

#we can give any video name instead of 0 to capture live webcam
#video = cv2.VideoCapture(0)
video = cv2.VideoCapture('video.mp4')

stored_frames = []

while True:
	frame_read_successfully, frame = video.read()
	
	#if no frame is read then break out of the while loop(if video ends then while loop ends)
	if frame_read_successfully:
		#appending the frame into the list
		stored_frames.append(frame)
		
		#converting into grayscale
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#this will output the coordinates of the rectangle
		cars_coordinates = trained_cars_data.detectMultiScale(gray_frame)
		human_coordinates = trained_ped_data.detectMultiScale(gray_frame)
		
		#rect_color = (randrange(256), randrange(256), randrange(256))
		rect_color1g = (0, 255, 0)
		rect_color2r = (0, 0, 255)
		rect_width = 2
		
		#drawing the rectangle while looping through all the faces
		for (x_coor1, y_coor1, w1, h1) in cars_coordinates:
			cv2.rectangle(frame, (x_coor1, y_coor1), (x_coor1+w1, y_coor1+h1), rect_color1g, rect_width)
		
		#drawing the rectangle while looping through all the faces
		for (x_coor2, y_coor2, w2, h2) in human_coordinates:
			cv2.rectangle(frame, (x_coor2, y_coor2), (x_coor2+w2, y_coor2+h2), rect_color2r, rect_width)
		
		
		#we needed waitKey() to keep the window open otherwise it closes immediately
		cv2.imshow('Pedestrian and Cars Detector' , frame)
		key = cv2.waitKey(1) #milliseconds waiting
	else:
		break
	
	if key==81 or key==113: #ascii of q and Q
		break

#release the video capture
video.release()

choice = input('Do you want the output as a video:[y/n] ')
if choice.lower() == 'y':
	#output filename
	outputfn = input('Enter name of the new video:[extension also] ')
	print('Converting it into a file...')
	
	#frame into video part
	fps = 15.0
	
	height, width, layers = frame.shape
	size = (width,height)
	
	out = cv2.VideoWriter(outputfn,cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
	
	for i in range(len(stored_frames)):
		# writing to a image array
		out.write(stored_frames[i])
		out.release()
else:
	print('Skipped...')

print('Done...')
