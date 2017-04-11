import matplotlib.pyplot as plt
import numpy as np
import os
from esue_generator import load_scan, get_pixels_hu, get_pixels_hu_test, zero_center, normalize, inv_normalize
import dicom

NPY_FOLDER = '../input/val/'
p = np.load("../pred.npy")


def listPlot():
    x_path = os.path.join(NPY_FOLDER, "x/")
    y_path = os.path.join(NPY_FOLDER, "y/")
    path, x_dirs, x_files = os.walk(x_path).next()
    path, y_dirs, y_files = os.walk(y_path).next()

    # label = np.load(os.path.join(path, x_files[0])).astype("float32")
    # label = normalize(label)
    # label0 = np.squeeze(label[0])
    # subtract = np.subtract(p, label0)

    # print "subtract: ",subtract.shape
    # print subtract


    print p.shape

    #plt.imshow(np.squeeze(p[345]), cmap=plt.cm.gray)

    idx = 30

    plt.subplot(3, 4, 1)
    plt.imshow(np.squeeze(p[idx]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 2)
    plt.imshow(np.squeeze(p[idx + 1]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 3)
    plt.imshow(np.squeeze(p[idx + 2]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 4)
    plt.imshow(np.squeeze(p[idx + 3]), cmap=plt.cm.gray)

    plt.subplot(3, 4, 5)
    plt.imshow(np.squeeze(p[idx + 4]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 6)
    plt.imshow(np.squeeze(p[idx + 5]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 7)
    plt.imshow(np.squeeze(p[idx + 6]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 8)
    plt.imshow(np.squeeze(p[idx + 7]), cmap=plt.cm.gray)

    plt.subplot(3, 4, 9)
    plt.imshow(np.squeeze(p[idx + 8]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 10)
    plt.imshow(np.squeeze(p[idx + 9]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 11)
    plt.imshow(np.squeeze(p[idx + 10]), cmap=plt.cm.gray)
    plt.subplot(3, 4, 12)
    plt.imshow(np.squeeze(p[idx + 11]), cmap=plt.cm.gray)

    # plt.subplot(2, 3, 4)
    # plt.hist(p.flatten(), bins=80, color='c')
    # plt.subplot(2, 3, 5)
    # plt.hist(label0.flatten(), bins=80, color='c')
    # plt.subplot(2, 3, 6)
    # plt.hist(subtract.flatten(), bins=80, color='c')

    plt.show()




def singlePlot():
    x_path = os.path.join(NPY_FOLDER, "x/")
    y_path = os.path.join(NPY_FOLDER, "y/")
    path, x_dirs, x_files = os.walk(x_path).next()
    path, y_dirs, y_files = os.walk(y_path).next()

    label = np.load(os.path.join(path, x_files[0])).astype("float32")
    #label = zero_center(normalize(label))
    label = normalize(label)
    #label[label < 0 ] = 0
    #print label.shape
    label0 = np.squeeze(label[0])
    subtract = np.subtract(p, label0)

    #print "subtract: ",subtract.shape
    #print subtract


    print p.shape

    plt.subplot(2, 3, 1)
    plt.imshow(p, cmap=plt.cm.gray)
    plt.subplot(2, 3, 2)
    plt.imshow(label0, cmap=plt.cm.gray)
    plt.subplot(2, 3, 3)
    plt.imshow(subtract, cmap=plt.cm.gray)

    plt.subplot(2, 3, 4)
    plt.hist(p.flatten(), bins=80, color='c')
    plt.subplot(2, 3, 5)
    plt.hist(label0.flatten(), bins=80, color='c')
    plt.subplot(2, 3, 6)
    plt.hist(subtract.flatten(), bins=80, color='c')

    plt.show()


def testPlot():

    OUTPUT_FOLDER = '/home/deep-motion/output/val_images/ffe02fe7d2223743f7fb455dfaff3842/'
    p0 = np.load(os.path.join(OUTPUT_FOLDER, "pred0.npy"))
    p = np.load(os.path.join(OUTPUT_FOLDER, "pred.npy")).astype(np.int16)
    test = np.load(os.path.join(OUTPUT_FOLDER, "test1.npy"))



    # a = [[0.0, 0.2], [0.9, 1.0]]
    #
    # t = p0[100][0]
    # ti = p[100][0]
    #
    # print t
    # print
    # print ti
    # print
    # print inv_normalize(a)
    # print
    pred = inv_normalize(p0)
    pred = pred.astype(np.int16)

    print pred


    plt.subplot(2, 3, 1)
    plt.imshow(p0, cmap=plt.cm.gray)
    plt.subplot(2, 3, 2)
    plt.imshow(pred, cmap=plt.cm.gray)
    plt.subplot(2, 3, 3)
    plt.imshow(test, cmap=plt.cm.gray)

    plt.subplot(2, 3, 4)
    plt.hist(p0.flatten(), bins=80, color='c')
    plt.subplot(2, 3, 5)
    plt.hist(p.flatten(), bins=80, color='c')
    plt.subplot(2, 3, 6)
    plt.hist(test.flatten(), bins=80, color='c')

    # plt.subplot(2, 3, 4)
    # plt.hist(p.flatten(), bins=80, color='c')
    # plt.subplot(2, 3, 5)
    # plt.hist(label0.flatten(), bins=80, color='c')
    # plt.subplot(2, 3, 6)
    # plt.hist(subtract.flatten(), bins=80, color='c')

    plt.show()

def read_dcm():
    pred = dicom.read_file(os.path.join('/home/deep-motion/output/val_images/ffe02fe7d2223743f7fb455dfaff3842/','pred.dcm'))
    patient = dicom.read_file(os.path.join('/home/deep-motion/output/val_images/ffe02fe7d2223743f7fb455dfaff3842/', 'patient.dcm'))

    pred_pixel =  pred.pixel_array
    print pred_pixel
    print
    patient_pixel = patient.pixel_array
    print patient_pixel

    plt.subplot(2, 2, 1)
    plt.imshow(pred_pixel, cmap=plt.cm.gray)
    plt.subplot(2, 2, 2)
    plt.hist(pred_pixel.flatten(), bins=80, color='c')

    plt.subplot(2, 2, 3)
    plt.imshow(patient_pixel, cmap=plt.cm.gray)
    plt.subplot(2, 2, 4)
    plt.hist(patient_pixel.flatten(), bins=80, color='c')
    plt.show()

def make_half_slices():
    # INPUT_FOLDER = '../input/sample_images/'
    # INPUT_FOLDER = '../input/val_images/ffe02fe7d2223743f7fb455dfaff3842/'
    INPUT_FOLDER = '../input/val_images/'

    half_FOLDER = '../input/half_images/'

    # NPY_FOLDER = '../input/npy/'
    NPY_FOLDER = '../input/val/'
    patients = os.listdir(INPUT_FOLDER)
    # patients.sort()

    # path, dirs, files = os.walk(INPUT_FOLDER).next()
    #
    # for idx, file in enumerate(files):
    #     # print idx
    #     print os.path.join(path, file)

    path = INPUT_FOLDER
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))


def show_hist():
    INPUT_FOLDER = '/home/deep-motion/input/val_images/'
    patients = os.listdir(INPUT_FOLDER)

    path, dirs, files = os.walk(INPUT_FOLDER).next()
    dirs_count = len(dirs)

    for i in range(dirs_count):
        patient = load_scan(INPUT_FOLDER + patients[i])
        patient_pixels_test = get_pixels_hu_test(patient)
        patient_pixels = get_pixels_hu(patient)
        print i, np.amax(patient_pixels)

        normalized_patient_pixels = normalize(patient_pixels)
        # normalized_patient_pixels = zero_center(normalized_patient_pixels)

        plt.subplot(1, 3, 1)
        plt.hist(patient_pixels_test.flatten(), bins=80, color='c')
        plt.subplot(1, 3, 2)
        plt.hist(patient_pixels.flatten(), bins=80, color='c')
        plt.subplot(1, 3, 3)
        plt.hist(normalized_patient_pixels.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()


def main():
    menu = 3

    if menu == 1:
        make_half_slices()
    elif menu == 2:
        testPlot()
    elif menu == 3:
        show_hist()





if __name__ == "__main__":
    main()
