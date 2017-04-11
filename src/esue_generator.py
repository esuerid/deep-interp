import dicom
from dicom.dataset import Dataset, FileDataset
from dicom.filebase import DicomFile
from dicom.filereader import read_preamble, _read_file_meta_info, read_dataset

#import cv2
import os, uuid, hashlib
#import pandas as pd
import numpy as np
#import glob
#import pickle
import scipy.ndimage
from operator import itemgetter
from sklearn.metrics import log_loss
# import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import random
import math

#INPUT_FOLDER = '../input/sample_images/'
# INPUT_FOLDER = '../input/val_images/'
INPUT_FOLDER = '../input/test_images/'
#NPY_FOLDER = '../input/npy/'
# NPY_FOLDER = '../input/val/'
NPY_FOLDER = '../input/test/'
PIXEL_MEAN = 0.25 #this parameter used in the zero center function
patients = os.listdir(INPUT_FOLDER)
#patients.sort()



class DicomManager():
    def __init__(self, filename):
        self.filename = filename
        fp = DicomFile(filename, 'rb')
        self.preamble = read_preamble(fp, False)  # if no header, raise exception
        self.file_meta = _read_file_meta_info(fp)

    # def get_main_file_meta(self):
    #     fp = DicomFile(self.filename, 'rb')
    #     self.preamble = read_preamble(fp, False)  # if no header, raise exception
    #     self.file_meta = _read_file_meta_info(fp)

    def hello(self):
        ss =0

    def make_uid(self, entropy_srcs=None, prefix='2.25.'):
        '''Generate a DICOM UID value.
        Follows the advice given at:
        http://www.dclunie.com/medical-image-faq/html/part2.html#UID
        Parameters
        ----------
        entropy_srcs : list of str or None
            List of strings providing the entropy used to generate the UID. If
            None these will be collected from a combination of HW address, time,
            process ID, and randomness.
        '''
        # Combine all the entropy sources with a hashing algorithm
        if entropy_srcs is None:
            entropy_srcs = [str(uuid.uuid1()), # 128-bit from MAC/time/randomness
                            str(os.getpid()), # Current process ID
                            random().hex() # 64-bit randomness
                           ]
        hash_val = hashlib.sha256(''.join(entropy_srcs))

        # Converet this to an int with the maximum available digits
        avail_digits = 64 - len(prefix)
        int_val = int(hash_val.hexdigest(), 16) % (10 ** avail_digits)

        return prefix + str(int_val)

    def get_new_file_meta(self):
        new_MediaStorageSOPInstanceUID = self.file_meta.MediaStorageSOPInstanceUID

        # prefix = new_MediaStorageSOPInstanceUID.rsplit('.', 1)[0] + '.'
        #new_MediaStorageSOPInstanceUID = make_uid(prefix=prefix)
        # file_meta.MediaStorageSOPInstanceUID = new_MediaStorageSOPInstanceUID

        # return file_meta

    def update_original_dicom(dicom):
        if "SliceThickness" in dicom:
            dicom.SliceThickness = dicom.SliceThickness/2.0

    def make_middle_dicom(filename, dicom_from01, dicom_from02):
        # remove_dataElement_list = ("PixelData", "InstanceNumber")
        new_dicom_dict = dicom_from01.copy()
        new_dicom = Dataset(new_dicom_dict)

        #delete some dataElement
        # for s in remove_dataElement_list:
        #     tag = new_dicom.data_element(s).tag
        #     del new_dicom[tag]

        #change InstanceNumber
        if "InstanceNumber" in dicom_from01:
            new_dicom.InstanceNumber = dicom_from01.InstanceNumber + 1
        # change ImagePositionPatient
        if "ImagePositionPatient" in dicom_from01:
            slice_thickness = np.abs(dicom_from01.ImagePositionPatient[2] - dicom_from02.ImagePositionPatient[2])
            new_dicom.ImagePositionPatient[2] = dicom_from01.ImagePositionPatient[2] + slice_thickness / 2
        if "SliceLocation" in dicom_from01:
            slice_thickness = np.abs(dicom_from01.SliceLocation - dicom_from02.SliceLocation)
            new_dicom.SliceLocation = dicom_from01.SliceLocation + slice_thickness / 2
        if "SOPInstanceUID" in dicom_from01:
            SOPInstanceUID = dicom_from01.SOPInstanceUID
            prefix = SOPInstanceUID.rsplit('.', 1)[0] + '.'
            newSOPInstanceUID = make_uid(prefix=prefix)
            new_dicom.SOPInstanceUID = newSOPInstanceUID

        # print ds.SOPInstanceUID
        print "==>start<=="
        print dicom_from01.is_implicit_VR
        print dicom_from01.is_little_endian
        print

        # ds = FileDataset(filename, new_dicom, file_meta=new_dicom)


        # ds.save_as(filename)

        return new_dicom



# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def load_scan_with_header(path):
    # slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # slices.sort(key = lambda x: int(x.InstanceNumber))

    new_filename = "./newdicom.dcm"
    # new_dicom = make_middle_dicom(new_filename, slices[0], slices[1])


    patient_num = 0
    path = INPUT_FOLDER + patients[patient_num]
    for idx, s in enumerate(os.listdir(path)):
        filename = path + '/' + s
        # new_filedataset = get_new_file_meta(filename)
        # get_new_file_meta(filename)




        if idx == 1:
            break




    # new_dicom.save_as("./newdicom.dcm")

    # for s in slices:
    #     #print s.StudyInstanceUID
    #     #print s.SeriesInstanceUID
    #     print s.SOPInstanceUID
    #     #print make_uid(prefix="1.3.6.1.4.1.14519.5.2.1.7009.9004.")



    # try:
    #     slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    # except:
    #     slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    # for s in slices:
    #     s.SliceThickness = slice_thickness
    #
    #
    #
    # return slices

def get_pixels_hu_test(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
    image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def get_inv_pixels_hu(scans, image):
    # image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    # image[image == -2000] = 0
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = image.astype(np.float64) / slope
        image = image.astype(np.int16)
    image -= np.int16(intercept)
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image, new_spacing


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def inv_normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    # image = np.multiply(image,(MAX_BOUND - MIN_BOUND))
    image = np.multiply(image, (MAX_BOUND - MIN_BOUND)) + MIN_BOUND
    # image[image > 1] = 1.
    # image[image < 0] = 0.

    return np.round(image)

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

#esuerid added
#this is not working
class DataManager(object):
    PIXEL_MEAN = 0.25
    def __init__(self):
        self.count = 0
        self.x_total = []
        self.y_total = []
    def append(self, x, y):
	if not self.x_total:
	    self.x_total = x.tolist()
	    self.y_total = y.tolist()
	else:
            self.x_total.append(x.tolist())
            self.y_total.append(y.tolist())
	self.count = self.count + 1
	return self 
    def preprocess(self):
	return np.array(self.x_total), np.array(self.y_total)

class NumpyDataManager(object):
    PIXEL_MEAN = 0.25
    def __init__(self):
        self.count = 0
        self.x_total = np.array([])
        self.y_total = np.array([])
    def append(self, x, y):
        if self.x_total.size ==0:
            self.x_total = x
            self.y_total = y
        else:
            self.x_total = np.concatenate((self.x_total, x), axis=0)
            self.y_total = np.concatenate((self.y_total, y), axis=0)
        self.count = self.count + 1
        return self
    def preprocess(self):
        return self.x_total, self.y_total

def batch_convertor():

    path, dirs, files = os.walk(INPUT_FOLDER).next()
    dirs_count = len(dirs)
   
    for i in range(dirs_count):
        x, y = batch_convertor_by_patient(i)

	#save each file (ex. 000001.npy) in x, y folder
        x_dir = os.path.join(NPY_FOLDER, "x/")
        y_dir = os.path.join(NPY_FOLDER, "y/")
        if not os.path.exists(x_dir):
            os.makedirs(x_dir)
        if not os.path.exists(y_dir):
            os.makedirs(y_dir)
    	x_path = os.path.join(x_dir, '{0:05}'.format(i)  +".npy")
    	y_path = os.path.join(y_dir, '{0:05}'.format(i)  +".npy")
    	np.save(x_path, x)
    	np.save(y_path, y)
    	print "[" + str(i+1) + "/" + str(dirs_count)  + "]" + " npy data saved at " + x_path
    	print "[" + str(i+1) + "/" + str(dirs_count)  + "]" + " npy data saved at " + y_path    

def randomize(file_select_num):

    x_path = os.path.join(NPY_FOLDER, "x/")
    y_path = os.path.join(NPY_FOLDER, "y/")
    path, dirs, files = os.walk(x_path).next()
    files_count = len(files)

    random.seed()
    chosen_list = random.sample(files, file_select_num)

    #concat and shuffle data 
    data = NumpyDataManager() 
    for f in chosen_list:
	x = np.load(os.path.join(x_path, f))
	y = np.load(os.path.join(y_path, f))
	print x.shape
	data.append(x, y)
    x, y = data.preprocess()
    x, y= shuffle(x, y)

    split_x = np.array_split(x, file_select_num)
    split_y = np.array_split(y, file_select_num)
    
    #save split data
    for idx, f in enumerate(chosen_list):
	#print os.path.join(x_path, f)
	#print split_x[idx].shape
        x = np.save(os.path.join(x_path, f), split_x[idx])
        y = np.save(os.path.join(y_path, f), split_y[idx])   
	print "count : " , split_x[idx].shape[0]
 
    print "chosen list : ", chosen_list

def saveSplitData():
    split_x, split_y = randomize()
    
    x0 = split_x[0]
    y0 = split_y[0]
    
    x1 = split_x[1]
    y1 = split_y[1]

    np.save()

def plotRandomXY():

    x_path = os.path.join(NPY_FOLDER, "x/")
    y_path = os.path.join(NPY_FOLDER, "y/")
    path, dirs, files = os.walk(x_path).next()
    files_count = len(files)
    choice_count = 1

    random.seed()
    chosen_list = random.sample(files, choice_count)

    f = chosen_list[0]
    file_path = os.path.join(x_path, f)
    x = np.load(file_path)
    y = np.load(file_path)

    print "file : ", file_path

    plotFromXY(x,y)

def plotFromXY(x, y):
    row_num = 10
    total_count = x.shape[0]
    rand_num = random.randrange(0, total_count - (row_num - 1))
    x_list = np.zeros(shape=(row_num, 2, 512, 512))
    y_list = np.zeros(shape=(row_num, 1, 512, 512))
    for i in range(row_num):
	x_list[i, :, :] = x[rand_num+i]
	y_list[i, :, :] = y[rand_num+i]
    
    for i in range(row_num):
	temp_x = x_list[i]
	temp_y = y_list[i]
        plt.subplot(row_num,3,i*3+1)
        plt.imshow(temp_x[0], cmap=plt.cm.gray)
    	plt.subplot(row_num,3,i*3+2)
    	plt.imshow(temp_y[0], cmap=plt.cm.gray)
    	plt.subplot(row_num,3,i*3+3)
    	plt.imshow(temp_x[1], cmap=plt.cm.gray)
    plt.show()


def getMaxInFiles():
    path, dirs, files = os.walk(INPUT_FOLDER).next()
    dirs_count = len(dirs)

    for i in range(dirs_count):
	patient = load_scan(INPUT_FOLDER + patients[i])
        patient_pixels = get_pixels_hu(patient)
	print i , np.amax(patient_pixels)
	
        patient_pixels = normalize(patient_pixels)
        patient_pixels = zero_center(patient_pixels)

	plt.hist(patient_pixels.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
	plt.ylabel("Frequency")
        plt.show()

def getMeanInFiles():
    path, dirs, files = os.walk(INPUT_FOLDER).next()
    dirs_count = len(dirs)

    pixels_total = np.array([])

    for i in range(dirs_count):
        patient = load_scan(INPUT_FOLDER + patients[i])
        patient_pixels = get_pixels_hu(patient)

        patient_pixels = normalize(patient_pixels)
        #patient_pixels = zero_center(patient_pixels)
    
	print i, patient_pixels.shape
	
	if pixels_total.size == 0:
	    pixels_total = patient_pixels
	else:
	    pixels_total = np.concatenate((pixels_total, patient_pixels), axis=0)

    print np.mean(pixels_total)

        #plt.hist(patient_pixels.flatten(), bins=80, color='c')
        #plt.xlabel("Hounsfield Units (HU)")
        #plt.ylabel("Frequency")
        #plt.show()

def get_random_list():
    x_path = os.path.join(NPY_FOLDER, "x/")
    y_path = os.path.join(NPY_FOLDER, "y/")
    path, dirs, files = os.walk(x_path).next()
    random.seed()
    f = random.choice(files)
    x = np.load(os.path.join(x_path, f))
    y = np.load(os.path.join(y_path, f))
    #this is used for just aubsampling not for separating train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=random.uniform(0,1))
    return x_train, y_train


# esuerid added
def batch_convertor_by_patient(patient_num):
    patient = load_scan(INPUT_FOLDER + patients[patient_num])
    patient_pixels = get_pixels_hu(patient)

    im_len = patient_pixels.shape[0]
    X_LEN = patient_pixels.shape[1]
    Y_LEN = patient_pixels.shape[2]

    x = np.zeros(shape=(im_len, X_LEN, Y_LEN, 2), dtype=np.int16)
    y = np.zeros(shape=(im_len, X_LEN, Y_LEN, 1), dtype=np.int16)
    for i in range(im_len - 2):
        x[i, :, :, :1] = patient_pixels[i].reshape(X_LEN, Y_LEN, 1)
        x[i, :, :, 1:] = patient_pixels[i + 2].reshape(X_LEN, Y_LEN, 1)
        y[i] = patient_pixels[i + 1].reshape(X_LEN, Y_LEN, 1)

    # return np.transpose(x, (0, 3, 1, 2)), np.transpose(y, (0, 3, 1, 2))
    return x, y

def batch_generator(batch_size):
    while 1:
	x = np.zeros(shape=(batch_size, 512, 512, 2), dtype=np.float32) 
	y = np.zeros(shape=(batch_size, 512, 512, 1), dtype=np.float32) 
	
	total_size = 0
	while total_size < batch_size:
	    s_x, s_y = get_random_list()
	    sample_size = s_x.shape[0]
	    #print "sample_size : ", sample_size
	    for batch_i in range(sample_size):
		total_i = total_size + batch_i
		x[total_i, :, :, :] = s_x[batch_i, :, : ,:]
		y[total_i, :, :, :] = s_y[batch_i, :, : ,:]
		#print "total_i : ", total_i
		if(total_i >= batch_size-1):
		    break
	    total_size = total_size + sample_size
	
	#return np.transpose(x, (0, 3, 1, 2)), np.transpose(y, (0, 3, 1, 2))
	#x = np.transpose(x, (0, 2, 3, 1))
	#y = np.transpose(y, (0, 2, 3, 1))
	yield zero_center(normalize(x)), zero_center(normalize(y))

def main():

    menu = 1
    #1. convert each dicom file to npy 
    if menu == 1:
        batch_convertor()
    #2. shuffle the data
    elif menu == 2:
        iterate = 50
    	for i in xrange(iterate):
    	    randomize(10)
    #3. plot random data
    elif menu == 3:
        plotRandomXY()

    #4. batch_generator test
    elif menu == 4:
        for i in range(2):
	    x, y = batch_generator(100).next()
            plotFromXY(x, y)
            print x.shape

     # 5. load test
    elif menu == 5:
        load_scan_with_header(INPUT_FOLDER + patients[0])




    #getMaxInFiles()
    #getMeanInFiles()

    #first_patient = load_scan(INPUT_FOLDER + patients[0])
    #first_patient_pixels = get_pixels_hu(first_patient)

    #first_patient_pixels = first_patient_pixels.astype(np.float32)   

    #patient_list = first_patient_pixels.tolist()
 

    #print first_patient_pixels.dtype
 
    ##print first_patient_pixels[0]    

    #first_patient_pixels = normalize(first_patient_pixels)
    #print first_patient_pixels.dtype

    #plt.hist(first_patient_pixels.flatten(), bins=80, color='c')
    #plt.xlabel("Hounsfield Units (HU)")
    #plt.ylabel("Frequency")
    #plt.show()

    ## Show some slice in the middle
    #plt.imshow(first_patient_pixels[80], cmap=plt.cm.gray)
    #plt.show()
    
    #pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
    #print("Shape before resampling\t", first_patient_pixels.shape)
    #print("Shape after resampling\t", pix_resampled.shape)

    #plot_3d(pix_resampled, 400)

if __name__ == "__main__":
    main()
