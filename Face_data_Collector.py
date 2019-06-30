#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libraries
import numpy as np
import cv2
import os


# In[2]:


#To recogonize the feature of faces we import a xml file, this file has collection of feature that can detect faces.
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[4]:


def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting to gray for better accuracy
    faces= face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_faces = img[y:y+h,x:x+w]
        return cropped_faces


# In[1]:
def main():
	print('*Please be in a bright place, your face should be clearly visible*')
	print('This program will take 150 photos of you, Be calm, and wait till 150 photos are captured!')
	enter = input('Press enter to Continue')
	Pass = input('Enter the password: ')
	if Pass == 'akansh': #Default Password is 'akansh'- You can Change here!
		user_name = input('Please enter your name: ')
		cap = cv2.VideoCapture(0)
		count= 0 
		os.mkdir('/home/akansh/astute/Face_Lock_system/Faces/'+user_name+'/')

		while True:
		    ret,frame = cap.read()
		    if face_extractor is not None:
		        count +=1
		        face= cv2.resize(face_extractor(frame),(200,200))
		        face= cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
		        
		        #Path where the images will be stored!
		        file_name_path = '/home/akansh/astute/Face_Lock_system/Faces/'+user_name+'/'+str(count)+'.jpg'
		        cv2.imwrite(file_name_path,face)
		        
		        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
		        cv2.imshow('Face Cropper',face)
		        
		    else:
		        print("Face not found")
		        pass
		    if cv2.waitKey(1)==13 or count == 150:
		        break
		        
		cap.release()
		cv2.destroyAllWindows()
		print("Collecting Samples complete")
	else:
		print('Password is wrong!!!')
		main()

if __name__ == "__main__":
    main()

# In[ ]:




