from Tkinter import Tk, Frame
import sys 
import os,os.path
from PIL import Image

from face import *
import csv

filePath = os.path.abspath(__file__)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir))
parentDir = os.path.abspath(os.path.join(curDir,os.pardir))

sys.path.append(os.path.abspath(parentDir + "/Face_Recognition/lib"))



def training_on_new_data(directory):
    with open(parentDir + '/data/face_locations.csv', 'a') as csvfile:
        fieldnames=['person','fullfilename' , 'top', 'right', 'bottom', 'left']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        face_encode=[]
        for filename in os.listdir(directory):     
            img = load_image_file(directory + "/" + filename)
                 
            print 'filename:' + filename
            all_face_locations = face_locations(img)
            for face_location in all_face_locations:
                top,right,bottom,left = face_location
                face_image = img[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                Image._show(pil_image)
                person = raw_input("Name of the person (-1 if not face):")
                if person != '-1':
                    fullfilename = directory +"/"+filename
                    writer.writerow({'person':person, 'fullfilename': fullfilename, 'top':top, 'right':right, 'bottom':bottom, 'left':left})
                    print "written file: " + filename
                
                
'''            
    new_face_encode = [face_encode[i] for i in range(len(face_encode)) if face_encode[i] != [] ]
    print new_face_encode
    result = face_distance(new_face_encode[1:], new_face_encode[0])
    print result
'''

#directory = parentDir + "/unknown"
def testing_on_new_data(directory):
#if __name__ == '__main__':
    known_face_encodings={}
    with open(parentDir + '/data/face_locations.csv', 'r') as csvfile:
        fieldnames=['person', 'fullfilename' , 'top', 'right', 'bottom', 'left']
        reader = csv.DictReader(csvfile,fieldnames=fieldnames)
        for row in reader:
            face_location = [int(row['top']), int(row['right']), int(row['bottom']), int(row['left'])]
            if row['person'] in known_face_encodings:
                known_face_encodings[row['person']] = np.append(known_face_encodings[row['person']],np.array(face_encodings(load_image_file(row['fullfilename']),[face_location])),axis=0)
            else:
                known_face_encodings.update({row['person']:np.array(face_encodings(load_image_file(row['fullfilename']),[face_location]))})
    #known_face_encodings = np.array(known_face_encodings)
    #known_face_encodings = known_face_encodings.mean(axis=0)

    for filename in os.listdir(directory):
        img = load_image_file(directory + '/' + filename)
        
            
                
        all_face_locations = face_locations(img)
        for face_location in all_face_locations:
            top,right,bottom,left = face_location
            face_image = img[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            Image._show(pil_image)
            test_face_encode = face_encodings(img,[face_location])
            result={}
            name='Unknown'
            minval=1
            for person in known_face_encodings:
                r=face_distance(known_face_encodings[person], test_face_encode[0]).mean()
                if r <= 0.6:
                    result.update({person:(1-r)*100})
                    if r < minval:
                        minval=r
                        name=person
                        
                    
            print filename + ": The person is " + name
            print "All possiblities: " + str(result)
            if name == 'Megha Jain': raw_input("continue?")


#training_on_new_data(parentDir + "/hitesh-photos")
testing_on_new_data(parentDir + '/unknown')
