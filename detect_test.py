import cv2
import numpy as np
import sys
from os import listdir
from os.path import join

#PARAMETERS---------------------
directory='test'
scaleFactor=1.1
minNeighbors=3
showImages=False
#-------------------------------

left_ear_cascade = cv2.CascadeClassifier('./haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('./haarcascade_mcs_rightear.xml')

if left_ear_cascade.empty():
  raise IOError('Unable to load the left ear cascade classifier xml file')

if right_ear_cascade.empty():
  raise IOError('Unable to load the right ear cascade classifier xml file')


def tryDetect(file):
  img = cv2.imread(file)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  left_ear = left_ear_cascade.detectMultiScale(
    gray, 
    scaleFactor, 
    minNeighbors,
  )
  right_ear = right_ear_cascade.detectMultiScale(
    gray, 
    scaleFactor, 
    minNeighbors,
  )

  if (len(left_ear) == 0 and len(right_ear) == 0):
    return False

  if (showImages):
    for (x,y,w,h) in left_ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)

    for (x,y,w,h) in right_ear:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)

    cv2.imshow('Ear Detector', img)
    cv2.waitKey()
  return True

count = 0
files = listdir(directory)
for file in files:
  sys.stdout.write('Progress: {0} / {1}\r'.format(files.index(file)+1, len(files)))
  sys.stdout.flush()

  if tryDetect(join(directory,file)):
    count+=1

print('')
print('Detected: {0}'.format(count))            