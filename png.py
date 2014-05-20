import Image
import cv2
import numpy as np
import win32api
import win32con
import time
def detect(img):
	lowerb =(0,133,0)
	upperb =(255,173,255)
	
	dstTemp1 = cv2.inRange(img, (0,0,0), (0,0,0))	
	# dstTemp2 = cv2.inRange(img, (0,0,120), (256,256,127))
	# mask = cv2.bitwise_or(dstTemp1, dstTemp2)
	
	contours, hierarchy = cv2.findContours(dstTemp1,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

	return contours, hierarchy


	
def draw_line(img,p1,p2):
	cv2.line(img,p1,p2,(255,0,0), 3)
	
def draw_rects(img, rects, color):
	cv2.rectangle(img, (105, 106), (105, 200), color, 1)
	# for i in range(0,len(rects),2):
		# x1 = int(rects[i][0])
		# print x1
		# y1 = int(rects[i][1])
		# x2 = int(rects[i+1][0])
		# y2 = int(rects[i+1][1])
		
		# cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
def draw_circle(img,y,x):
	# cv2.circle(img,center,r,color)
	cv2.circle(img,(x,y),5,(0,0,255))
		
		
		
def getBinaryNCCImage(img):
	color_B=img[:,:,0]
	color_G=img[:,:,1]
	color_R=img[:,:,2]

	color_sum = color_R+color_G+color_B
	ncc_R = color_R/color_sum
	ncc_G = color_G/color_sum
	
	Q_up = -1.3767*ncc_R*ncc_R+1.0743*ncc_R+0.1452
	Q_dw = -0.776*ncc_R*ncc_R+0.5601*ncc_R+0.18
	w = (ncc_R-0.33)**2+(ncc_G-0.33)**2

	bc_pass1 = np.nonzero(ncc_G<Q_up)
	bc_nopass1 = np.nonzero(ncc_G>=Q_up)
	bc_pass2 = np.nonzero(w>0.04)
	bc_nopass2 = np.nonzero(w<=0.04)

	ncc_G[bc_pass1]=1
	ncc_G[bc_nopass1]=0
	w[bc_pass2]=1
	w[bc_nopass2]=0

	newimg = (ncc_G+w)/2.
	bc_pass3 = np.nonzero(newimg==1)
	bc_nopass3 = np.nonzero(newimg!=1)
	newimg[bc_pass3]=1
	newimg[bc_nopass3]=0

	ncc_img = img
	ncc_img[:,:,0] = newimg
	ncc_img[:,:,1] = newimg
	ncc_img[:,:,2] = newimg

	return ncc_img
		
		

def getBinaryNCCImage2(img):
	color_B=img[:,:,0]
	color_G=img[:,:,1]
	color_R=img[:,:,2]

	color_sum = color_R+color_G+color_B
	ncc_R = color_R/color_sum
	ncc_G = color_G/color_sum
	
	Q_up = -1.3767*ncc_R*ncc_R+1.0743*ncc_R+0.25
	Q_dw = -0.776*ncc_R*ncc_R+0.5601*ncc_R+0.18
	w = (ncc_R-0.33)**2+(ncc_G-0.33)**2

	bc_pass1 = np.nonzero(ncc_G<Q_up)
	bc_pass1_1 = np.nonzero(ncc_G>Q_dw)

	ncc_G[bc_pass1]=1
	ncc_G[bc_pass1_1]=1
	
	bc_nopass1 = np.nonzero(ncc_G!=1)
	ncc_G[bc_nopass1]=0
	
	bc_pass2 = np.nonzero(w>0.01)
	bc_nopass2 = np.nonzero(w<=0.01)
	w[bc_pass2]=1
	w[bc_nopass2]=0
	
	temp1= color_R-color_G
	bc=25.
	bc_pass3 = np.nonzero(temp1>bc)
	bc_nopass3 = np.nonzero(temp1<=bc)
	temp1[bc_pass3]=1
	temp1[bc_nopass3]=0
	
	
	newimg = (ncc_G+w+temp1)/3.
	bc_pass4 = np.nonzero(newimg==1)
	bc_nopass4 = np.nonzero(newimg!=1)
	newimg[bc_pass4]=1
	newimg[bc_nopass4]=0

	ncc_img = img
	ncc_img[:,:,0] = newimg
	ncc_img[:,:,1] = newimg
	ncc_img[:,:,2] = newimg

	return ncc_img,newimg


def handDetect_height(img_2Darray):
	img_height,img_width = np.shape(img_2Darray)
	temp = []
	for i in xrange(10,img_width-10):
		sum = np.sum(img_2Darray[:,i])
		temp.append(sum)
	
	x1 = temp.index(max(temp))
	y1 = img_2Darray[:,x1].argmax()
	y2 = img_2Darray[:,x1][::-1].argmax()
	return x1,y1,img_height-y2
	
def handDetect_width(img_2Darray,y1,y2):
	img_height,img_width = np.shape(img_2Darray)
	
	
	if abs(y2-y1)*100/img_height >50:
		pass
	
	mid = (y1+y2)/2
	temp_up = []
	temp_dw = []
	for i in xrange(y1,mid):
		sum = np.sum(img_2Darray[i,:])
		temp_up.append(sum)	
		
	for i in xrange(mid,y2):
		sum = np.sum(img_2Darray[i,:])
		temp_dw.append(sum)
	

		
	try:
		y_up = y1+temp_up.index(max(temp_up))
		y_dw = mid+temp_dw.index(max(temp_dw))
	except:
		return 0,0,0
		
		
	if abs(y_up-img_height)*100./img_height<20. and abs(y_dw-img_height)*100./img_height<20.:
		return 0,0,0
	elif abs(y_up-img_height)*100./img_height<20.:
		width_y = y_dw
	elif abs(y_dw-img_height)*100./img_height<20.:	
		width_y = y_up
	else:
		if max(temp_up)>max(temp_dw):
			width_y = y_up
		else:
			width_y = y_dw
		
		
	left_x = img_2Darray[width_y,:].argmax()
	right_x = img_2Darray[width_y,:][::-1].argmax()
	
	return width_y,left_x,img_width-right_x

	
	
def video_test(cam):
	
	
	ret, img = cam.read()
	# cv2.imshow('test2', img)
	floatimg = np.array(img,dtype=np.float32)
	nccimg,img_2Darray = getBinaryNCCImage2(floatimg)
	dst_img = imclose(imopen(nccimg))
	x1,y1,y2 = handDetect_height(img_2Darray)
	
	width_y,left_x,right_x = handDetect_width(img_2Darray,y1,y2)
	# print width_y,left_x,right_x
	if width_y==0 and left_x==0 and right_x==0:
		pass
	else:
		draw_line(dst_img,(left_x,width_y),(right_x,width_y))


	getWhitePoint,result_pos = finger_detect(dst_img)
	for i in result_pos[0]:
		y = getWhitePoint[0][i]
		x = getWhitePoint[1][i]
		
		draw_circle(dst_img,y,x)		
	# if temp_direction==0:
		# temp_direction = x1
		# direction=0
	# else:
		# if x1-temp_direction > 5:
			# direction+=1
		# elif x1-temp_direction < 5:
			# direction-=1
		# else:
			# pass
		
	# pixel_count = 13
	# if 	direction>pixel_count:
		# print "left",direction
		# win32api.keybd_event(37,0,0,0) 
		# temp_direction=0
		# direction=0
			
	# elif direction<pixel_count*-1:
		# print "right",direction
		# win32api.keybd_event(39,0,0,0) 
		# temp_direction=0
		# direction=0
		
	# else:
		# pass
		
	
	
	draw_line(dst_img,(x1,y1),(x1,y2))
	cv2.imshow('test', dst_img)	

def img_test():
	img = np.array(cv2.imread("DPP_1103.JPG"),dtype=np.float32)
	nccimg , img_2Darray  = getBinaryNCCImage2(img)
	contours, hierarchy = detect(nccimg)
	
	x1,y1,y2 = handDetect_height(img_2Darray)
	draw_line(img,(x1,y1),(x1,y2))
	
	# draw_line(img,(0,195),(200,195))
	width_y,left_x,right_x = handDetect_width(img_2Darray,y1,y2)

	if width_y==0 and left_x==0 and right_x==0:
		pass
	else:
		draw_line(img,(left_x,width_y),(right_x,width_y))
	
	cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
	cv2.imshow('dst', nccimg)
	

def imopen(img):
	#1.erode-> 2.dilate
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	dst = cv2.erode(img, kernel)
	result = cv2.dilate(dst, kernel)
	# result = result-dst
	return result
	
def imclose(img):
	#1.dilate -> 2.erode
	# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	kernel = b= np.array([[1,0,1],[0,1,0],[1,0,1]],dtype=np.uint8)
	dst = cv2.dilate(img, kernel)
	result = cv2.erode(dst, kernel)
	return result
	
def skeletonization(img):
	#骨架化
	imgsize = np.size(img)
	times= 1
	finish = False
	result = np.zeros(img.shape,np.uint8)
	temp=[]
	while (not finish):
		kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3*times,3*times))
		erode = cv2.erode(img, kernel)
		erode_open = imopen(erode)
		# if len(np.nonzero(erode_open==0)[0])==imgsize:
		if times==1:
			finish = True
		else:
			times+=1
		temp2 = erode-erode_open
		temp.append(erode-erode_open)

	print temp2 == temp[0]
	result = temp[0]
	# for i in temp:
		# result+=i
		
	# result = result/len(temp)
	return result
	
def finger_detect(img):
	getWhitePoint = np.nonzero(img==1)
	getPostion = np.nonzero(getWhitePoint[2]==0)

	interval = 5
	y0 = getWhitePoint[0][getPostion][:-2:interval]
	y1 = getWhitePoint[0][getPostion][1:-1:interval]
	y2 = getWhitePoint[0][getPostion][2::interval]

	x0 = getWhitePoint[1][getPostion][:-2:interval]
	x1 = getWhitePoint[1][getPostion][1:-1:interval]
	x2 = getWhitePoint[1][getPostion][2::interval]


	a_dot_b = ((x0-x1)*(x0-x2)+(y0-y1)*(y0-y2))
	a_length = ((x0-x1)**2+(y0-y1)**2)**0.5
	b_length = ((x0-x2)**2+(y0-y2)**2)**0.5

	cos_theta = a_dot_b/(a_length*b_length)
	result_angle = np.arccos(cos_theta)*180/np.pi
	print result_angle

	result_pos =  np.nonzero(np.logical_and(result_angle>120, result_angle<180))
	return getWhitePoint,result_pos


	
'''	
cam = cv2.VideoCapture(0)
direction = 0
temp_direction = 0

while True:
	video_test(cam)
	if 0xFF & cv2.waitKey(5) == 27:
		break
# img_test()

'''
rawimg = np.array(cv2.imread("DPP_1103.JPG"),dtype=np.float32)
nccimg , img_2Darray  = getBinaryNCCImage2(rawimg)
# contours, hierarchy = detect(nccimg)
dst_img = imclose(imopen(nccimg))
# dst_img = imopen(nccimg)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
dst = cv2.erode(dst_img, kernel)
result = cv2.dilate(dst, kernel)
dst_img = result-dst
getWhitePoint,result_pos = finger_detect(dst_img)
for i in result_pos[0]:
	y = getWhitePoint[0][i]
	x = getWhitePoint[1][i]
	
	draw_circle(dst_img,y,x)

cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst_img)	
cv2.waitKey(0)
cv2.destroyAllWindows()
