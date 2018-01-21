import sys
import os

curfilePath = os.path.abspath(__file__)
curDir = os.path.abspath(os.path.join(curfilePath,os.pardir))

sys.path.insert(0, curDir+'/lib')

from train_and_test import *

if __name__ == "__main__":
    while True:
        print "1. Train on some images"
        print "2. Categorize some images"
        print "3. Exit"
        choice = raw_input()
        if choice == '1':
            directory = raw_input("Enter the Directory where images are present(0 for default): ")
            if directory == '0':
                training_on_new_data()
            else:
                training_on_new_data(directory)
        elif choice == '2':
            directory = raw_input("Enter the Directory where images are present(0 for default): ")
            if directory == '0':
                testing_on_new_data()
            else:
                testing_on_new_data(directory)
            
        elif choice == '3':
            break
        else:
            print "Enter Valid choice..."
