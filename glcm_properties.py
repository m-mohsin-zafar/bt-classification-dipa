import numpy as np
import matplotlib.image as mpimg
import os
import csv
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

root_addr = 'F:/University Data/PIEAS/First Semester/DIPA/Brain Tumor Dataset'
directories = ['/L1', '/L2', '/L3']
patients_filename = 'patients_data.csv'
glcm_properties_filename = 'glcm_properties.csv'

ANGLES = [0., np.pi / 4., np.pi / 2., 3. * np.pi / 4.]
DISTANCES = [1]
properties = ["correlation", "contrast", "homogeneity", "energy"]

glcm_properties_list = []

patients_df = pd.read_csv(os.path.join(root_addr, patients_filename))

for directory in directories:

    path = os.fsdecode(root_addr + directory)

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".BMP"):
            print(filename)
            file_path = os.path.join(path, filename)
            filename_without_extension = os.path.splitext(filename)[0]

            img = mpimg.imread(file_path)
            img = np.array(img)

            glcm = greycomatrix(img, distances=DISTANCES, angles=ANGLES, levels=256, symmetric=True, normed=True)
            glcm_props = [greycoprops(glcm, properties[i]) for i in range(len(properties))]

            patient_info = patients_df[patients_df.filename == int(filename_without_extension)]

            glcm_properties_list.append([int(filename_without_extension), patient_info.patient_id.item(),
                                         patient_info.tumor_label.item(), glcm_props[0], glcm_props[1],
                                         glcm_props[2], glcm_props[3]])

        else:
            continue


glcm_props_columns = ['filename', 'patient_id', 'tumor_label', 'correlation', 'contrast', 'homogeneity', 'energy']
glcm_props_df = pd.DataFrame(glcm_properties_list)
glcm_props_df.to_csv(glcm_properties_filename, header=glcm_props_columns)
