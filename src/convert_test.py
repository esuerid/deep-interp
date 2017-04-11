# This script takes in an [x] FPS video and outputs 2*[x] FPS video by applying results
#  of frame interpolation using FI-CNN.

import sys
import os, uuid, hashlib

# BACKEND = "theano"
BACKEND = "tensorflow"

os.environ['KERAS_BACKEND'] = BACKEND
os.environ['THEANO_FLAGS'] = "device=gpu0, lib.cnmem=0.85, optimizer=fast_run"

import sys
import os
import time
from random import random

#import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave, imshow, imresize
from esue_generator import load_scan, get_pixels_hu, zero_center, normalize, load_scan_with_header, inv_normalize, get_inv_pixels_hu
from esue_unet import get_unet_2

from dicom.filereader import read_file_meta_info

import dicom
from dicom.dataset import Dataset, FileDataset
from dicom.filebase import DicomFile
from dicom.filereader import read_preamble, _read_file_meta_info, read_dataset



class PatientDicomManager():
    def __init__(self, patient_path):

        self.patient_path = patient_path
        self.dicoms = os.listdir(patient_path)
        self.file_meta, self.preamble =  self.__set_file_meta__()
        self.original_patient_dataset = self.__load_scan__()

    #need modify : self.file_meta changed by get_new_file_meta
    # def get_common_file_meta(self):
    #     return self.file_meta

    def __set_file_meta__(self):
        main_filename = os.path.join(self.patient_path, self.dicoms[0])
        fp = DicomFile(main_filename, 'rb')
        preamble = read_preamble(fp, False)  # if no header, raise exception
        file_meta = _read_file_meta_info(fp)
        return file_meta, preamble

    def __load_scan__(self):
        path = self.patient_path
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        for s in slices:
            s.SliceThickness = slice_thickness
        return slices

    def get_common_preamble(self):
        return self.preamble

    # def get_new_file_meta(self):
    #     new_MediaStorageSOPInstanceUID = self.file_meta.MediaStorageSOPInstanceUID
    #     prefix = new_MediaStorageSOPInstanceUID.rsplit('.', 1)[0] + '.'
    #     new_file_meta = self.file_meta
    #     new_file_meta.MediaStorageSOPInstanceUID = make_uid(prefix=prefix)
    #     return new_file_meta


    # def double_dicom(output_folder, patient):
    def double_dicom(self, output_folder):
        # patient is dicom
        # slices is pixel
        patient = self.original_patient_dataset
        slices = get_pixels_hu(patient)
        slices = slices.astype(np.float32)
        slices = normalize(slices)

        model = get_unet_2((512, 512, 2))
        model.load_weights("./../model_weights/weights.hdf5")

        im_len = slices.shape[0]
        X_LEN = slices.shape[1]
        Y_LEN = slices.shape[2]

        new_slice_arr = np.zeros(shape=(im_len * 2 - 1, 512, 512, 1))
        new_slice_arr[0, :, :, :] = slices[0].reshape(X_LEN, Y_LEN, 1)
        self.save_original_dicom(output_folder, 0, slices[0], patient)

        for i in range(1, im_len):
            if i % (im_len / 10) == 0:
                print ("Slice doubling is {0}% done.".format((i / (im_len / 10) * 10)))

            # concatenate along with the layer direction
            pre_slice = np.expand_dims(slices[i - 1], axis=2)
            post_slice = np.expand_dims(slices[i], axis=2)
            concat_arr = np.concatenate((pre_slice, post_slice), axis=2)
            expand_arr = np.expand_dims(concat_arr, axis=0)

            # prediction middle image
            pred = model.predict(expand_arr)

            # slices = np.expand_dims(slices[i], axis=0)
            # append to the new_slice_arr
            new_slice_arr[2 * i - 1, :, :, :] = pred
            new_slice_arr[2 * i, :, :, :] = np.expand_dims(np.expand_dims(slices[i], axis=0), axis=3)

            self.save_double_dicom(output_folder, i, pred, patient)
            self.save_original_dicom(output_folder, i, slices[i], patient)


            # if i == 3:
            #     break

        return new_slice_arr


    def save_double_dicom(self, output_folder, idx, pred, patient):
        new_slice_path = os.path.join(output_folder, '{0:05}'.format(2 * idx - 1) + ".dcm")
        new_dataset = self.make_middle_dicom(new_slice_path, dicom_from01=patient[idx - 1], dicom_from02=patient[idx])
        pred = np.squeeze(pred)
        pred = inv_normalize(pred)
        pred = get_inv_pixels_hu(patient, pred)
        self.file_meta.MediaStorageSOPInstanceUID = new_dataset.SOPInstanceUID
        new_filedataset = FileDataset(new_slice_path, new_dataset, file_meta=self.file_meta,
                                      preamble=self.get_common_preamble())
        new_filedataset.Columns = pred.shape[0]
        new_filedataset.Rows = pred.shape[1]
        new_filedataset.PixelData = pred.tostring()
        print new_filedataset.InstanceNumber
        new_filedataset.save_as(new_slice_path)

    def save_original_dicom(self, output_folder, idx, slice, patient):
        new_slice_path = os.path.join(output_folder, '{0:05}'.format(2 * idx) + ".dcm")

        new_dataset = self.update_original_dicom(idx, patient)
        # new_dataset = self.make_middle_dicom(new_slice_path, dicom_from01=patient[idx - 1], dicom_from02=patient[idx])
        # slice = np.squeeze(slice)
        slice = inv_normalize(slice)
        slice = get_inv_pixels_hu(patient, slice)
        self.file_meta.MediaStorageSOPInstanceUID = new_dataset.SOPInstanceUID
        new_filedataset = FileDataset(new_slice_path, new_dataset, file_meta=self.file_meta,
                                      preamble=self.get_common_preamble())
        new_filedataset.Columns = slice.shape[0]
        new_filedataset.Rows = slice.shape[1]
        new_filedataset.PixelData = slice.tostring()
        print new_filedataset.InstanceNumber
        new_filedataset.save_as(new_slice_path)


    def update_original_dicom(self, idx, patient):
        ds = patient[idx]
        if "InstanceNumber" in ds:
            ds.InstanceNumber = 2 * idx + 1
        if "SliceThickness" in ds:
            ds.SliceThickness = ds.SliceThickness/2.0
        return ds

    def make_middle_dicom(self, filename, dicom_from01, dicom_from02):
        remove_dataElement_list = ("PixelData")
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
        if "SliceThickness" in dicom_from01:
            # new_dicom.SliceThickness = dicom_from01.SliceThickness / 2.0
            new_dicom.SliceThickness = dicom_from01.SliceThickness
        if "ImagePositionPatient" in dicom_from01:
            slice_thickness = np.abs(dicom_from01.ImagePositionPatient[2] - dicom_from02.ImagePositionPatient[2])
            new_dicom.ImagePositionPatient[2] = dicom_from01.ImagePositionPatient[2] + slice_thickness / 2
        if "SliceLocation" in dicom_from01:
            slice_thickness = np.abs(dicom_from01.SliceLocation - dicom_from02.SliceLocation)
            new_dicom.SliceLocation = dicom_from01.SliceLocation + slice_thickness / 2
        if "SOPInstanceUID" in dicom_from01:
            SOPInstanceUID = dicom_from01.SOPInstanceUID
            prefix = SOPInstanceUID.rsplit('.', 1)[0] + '.'
            newSOPInstanceUID = self.make_uid(prefix=prefix)
            new_dicom.SOPInstanceUID = newSOPInstanceUID

        # print dicom_from01.is_implicit_VR
        # print dicom_from01.is_little_endian
        # print

        # ds = FileDataset(filename, new_dicom, file_meta=new_dicom)


        # ds.save_as(filename)

        return new_dicom

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
            entropy_srcs = [str(uuid.uuid1()),  # 128-bit from MAC/time/randomness
                            str(os.getpid()),  # Current process ID
                            random().hex()  # 64-bit randomness
                            ]
        hash_val = hashlib.sha256(''.join(entropy_srcs))

        # Converet this to an int with the maximum available digits
        avail_digits = 64 - len(prefix)
        int_val = int(hash_val.hexdigest(), 16) % (10 ** avail_digits)

        return prefix + str(int_val)

def test(path, patient):
    patientManager = PatientDicomManager(path + patient)


    print
    print patientManager.original_patient_dataset[0]
    print



def double_image(slices):

    model = get_unet_2((512,512,2))
    model.load_weights("./../model_weights/weights.hdf5")
	
    im_len = slices.shape[0]
    X_LEN  = slices.shape[1]
    Y_LEN  = slices.shape[2]

    new_slice_arr = np.zeros(shape=(im_len*2-1, 512, 512, 1))
    new_slice_arr[0, : , : , :] = slices[0].reshape(X_LEN, Y_LEN, 1)

    print slices[0].reshape(X_LEN, Y_LEN, 1).shape
    print slices[0].shape
    print np.expand_dims(slices[0], axis=2).shape
    print new_slice_arr.shape
	
    #print len(slices)
    #for i in range(1, len(slices)):

    for i in range(1, im_len):
        if i % (im_len / 10) == 0:
            print ("Slice doubling is {0}% done.".format((i / (im_len / 10) * 10)))

        # concatenate along with the layer direction
        pre_slice = np.expand_dims(slices[i-1], axis=2)
        post_slice = np.expand_dims(slices[i], axis=2)
        concat_arr = np.concatenate((pre_slice, post_slice), axis=2)
        expand_arr = np.expand_dims(concat_arr, axis=0)

        #prediction middle image
        pred = model.predict(expand_arr)

        # slices = np.expand_dims(slices[i], axis=0)
        #append to the new_slice_arr
        new_slice_arr[2*i-1, : , : , :] = pred
        new_slice_arr[2*i, :, :, :] = np.expand_dims(np.expand_dims(slices[i], axis=0), axis=3)

        if i == 1:
            break

    # np.save("../pred.npy", new_slice_arr)

    return new_slice_arr


#
# def double_dicom(output_folder, patient):
#     #patient is dicom
#     #slices is pixel
#     slices = get_pixels_hu(patient)
#     slices = slices.astype(np.float32)
#     slices = normalize(slices)
#
#     model = get_unet_2((512, 512, 2))
#     model.load_weights("./../model_weights/weights.hdf5")
#
#     im_len = slices.shape[0]
#     X_LEN = slices.shape[1]
#     Y_LEN = slices.shape[2]
#
#     new_slice_arr = np.zeros(shape=(im_len * 2 - 1, 512, 512, 1))
#     new_slice_arr[0, :, :, :] = slices[0].reshape(X_LEN, Y_LEN, 1)
#
#     for i in range(1, im_len):
#         if i % (im_len / 10) == 0:
#             print ("Slice doubling is {0}% done.".format((i / (im_len / 10) * 10)))
#
#         # concatenate along with the layer direction
#         pre_slice = np.expand_dims(slices[i - 1], axis=2)
#         post_slice = np.expand_dims(slices[i], axis=2)
#         concat_arr = np.concatenate((pre_slice, post_slice), axis=2)
#         expand_arr = np.expand_dims(concat_arr, axis=0)
#
#         # prediction middle image
#         pred = model.predict(expand_arr)
#
#         # slices = np.expand_dims(slices[i], axis=0)
#         # append to the new_slice_arr
#         new_slice_arr[2 * i - 1, :, :, :] = pred
#         new_slice_arr[2 * i, :, :, :] = np.expand_dims(np.expand_dims(slices[i], axis=0), axis=3)
#
#         #save dicom file
#         new_slice_path = os.path.join(output_folder, "x/" + '{0:05}'.format(2 * i - 1) + ".dcm")
#         new_dicom = make_middle_dicom(new_slice_path, dicom_from01=patient[i-1], dicom_from02=patient[i])
#         new_dicom.save_as(new_slice_path)
#
#
#         if i == 1:
#             break

    # np.save("../pred.npy", new_slice_arr)


#def save_vid(vid_arr, vid_out_path, fps):
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    # fourcc = cv2.VideoWriter_fourcc(*'X264')
#    out = cv2.VideoWriter(vid_out_path, fourcc, fps, (384, 128))
#
#    for i in range(len(vid_arr)):
#        out.write(vid_arr[i])

def main():


    menu = 1

    INPUT_FOLDER = None
    OUTPUT_FOLDER = None

    if menu == 1:
        INPUT_FOLDER = '../input/val_images/'
        OUTPUT_FOLDER = '../output/val_images/'
    elif menu == 2:
        INPUT_FOLDER = '../output/half_images/'
        OUTPUT_FOLDER = '../output/half_double_images/'
    elif menu == 3:
        INPUT_FOLDER = '../output/half_double_images/'
        OUTPUT_FOLDER = '../output/half_quad_images/'
    elif menu == 4:
        INPUT_FOLDER = '../output/half_quad_images/'
        OUTPUT_FOLDER = '../output/half_octa_images/'
    elif menu == 5:
        INPUT_FOLDER = '../output/half_octa_images/'
        OUTPUT_FOLDER = '../output/half_hexa_images/'
    elif menu == 6:
        INPUT_FOLDER = '../output/half_hexa_images/'
        OUTPUT_FOLDER = '../output/half_dohexa_images/'





    path, patients, files = os.walk(INPUT_FOLDER).next()
    for patient in patients:
        # test(path, patient)
        input_patient_path = path + patient
        patientManager = PatientDicomManager(input_patient_path)

        # make output directory
        output_patient_path = OUTPUT_FOLDER + patient
        if not os.path.exists(output_patient_path):
            os.makedirs(output_patient_path)

        patientManager.double_dicom(output_patient_path)













            # print patient[0]

            # for p in patient:
            #     print slice

            # patient_pixels = get_pixels_hu(patient)
            # patient_pixels = patient_pixels.astype(np.float32)
            # patient_pixels = normalize(patient_pixels)
            #
            #
            # double_dicom(OUTPUT_FOLDER, patient)




            # make output directory
            # if not os.path.exists(OUTPUT_FOLDER + patients[patient_num]):
            #     os.makedirs(OUTPUT_FOLDER + patients[patient_num])



            # path, patients, files = os.walk(INPUT_FOLDER).next()
            # dirs_count = len(patients)
            # for i in range(dirs_count):
            #     # make output directory
            #     if not os.path.exists(OUTPUT_FOLDER + patients[i]):
            #         os.makedirs(OUTPUT_FOLDER + patients[i])



if __name__ == '__main__':
    main()
