import os

folder_path = r'..\references\papers'
files = os.listdir(folder_path)

for file in files:
    print(file)
