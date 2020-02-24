'''
input: 
1. npy images of train/test sets
2. Solution file matching images and labels

output: 
train.h5 or test.h5

'''
import csv
import pandas as pd
import os
import os.path
from os import path
import h5py
import numpy as np
from PIL import Image

clients = 4                                   # The number of clients/sites
dir_path = '/Volumes/KESU/Retina/train-all/'              # Path of train/test dataset directories
save_path = '/Volumes/KESU/Retina/'              # The path to store genrated h5 files
solution_path = '/Volumes/KESU/Retina/labels.csv'


#['train', 'test']
dtype = 'train'                                


# initialize train.h5 and test.h5 in specific path
F = h5py.File(save_path + dtype + '.h5', 'w')
grp = F.create_group('examples')


# for i in range(clients):
# grp_el = grp.create_group(str(i+1))    # Create a subgroup for each client


# turn the solution file into a dict in order to check label quickly
df=pd.read_csv(solution_path, header=0)
df=df.applymap(str).groupby('39123_right_test.npy')['0.0'].apply(list).to_dict()
df['39123_right_test.npy'] = '0'



# read all files in train set or test set into corresponding h5 files
# path = dir_path + dtype + '/'


# list = os.listdir(path)
# num = int(len(list) / clients)                    # Data amount per client


# list = os.listdir(path)
files = os.walk(dir_path)

sites_file = dict()

for i in range(clients):
	label_array = []
	pixels_array = []

	site=pd.read_csv("/Volumes/KESU/Retina/4_sites/train_v" + str(i+1) + ".csv")
	column_name = list(site)[0]
	a = site[column_name].tolist()
	a.append(column_name)

	for eachName in a:
		if path.exists(dir_path + str(eachName)):
			pixels_array.append(np.load(dir_path + str(eachName)))
			label_array.append(int(float(df[eachName][0])))

	label_array = np.array(label_array)
	pixels_array = np.array(pixels_array)
	
	grp_el = grp.create_group(str(i)) # Create a subgroup for each client
	grp_el.create_dataset('label', data=label_array)
	grp_el.create_dataset('pixels', data=pixels_array, dtype='f')




# 	if count % num == 1:
# 		client_id = int(count / num)
# 		if(client_id > 0):                        # Put label and pixels data into proper groups in h5
# 			label_array = np.array(label_array)
# 			pixels_array = np.array(pixels_array)
# 			grp_el.create_dataset('label', data=label_array)
# 			grp_el.create_dataset('pixels', data=pixels_array, dtype='f')

# 		grp_el = grp.create_group(str(client_id)) # Create a subgroup for each client
# 		label_array = []
# 		pixels_array = []

# 	image_name = str(name[:-4])                   # if the image name is "XXX.npy" 
# 	label_array.append(int(df[image_name][0]))
# 	pixels_array.append(np.load(path + str(name)))  # load npy images

# 	count = count + 1

# label_array = np.array(label_array)
# pixels_array = np.array(pixels_array)
# grp_el.create_dataset('label', data=label_array)
# grp_el.create_dataset('pixels', data=pixels_array, dtype='f')