# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:31:59 2019

@author: mohsi
"""

import numpy as np
import h5py
import os
import scipy.misc
import csv

root_addr = 'F:/University Data/PIEAS/First Semester/DIPA/Brain Tumor Dataset'
directory_input = '/brainTumorDataPublic_1-766'
directory_target_1 = '//L1/'
directory_target_2 = '//L2/'
directory_target_3 = '//L3/'

pid = 'cjdata/PID'
label = 'cjdata/label'
image = 'cjdata/image'
tumor_border = 'cjdata/tumorBorder'
tumor_mask = 'cjdata/tumorMask'

directory = os.fsdecode(root_addr + directory_input)

patients_list = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mat"):
        # print(os.path.join(directory, filename))
        print(filename)
        with h5py.File((directory + '/' + filename), 'r') as f:
            f_pid = np.array(f[pid])
            f_label = np.array(f[label])
            f_image = np.array(f[image])
            # f_tumor_border = np.array(f[tumor_border])
            # f_tumor_mask = np.array(f[tumor_mask])
            f_pid = ''.join(chr(i) for i in f_pid)

            filename_without_extension = os.path.splitext(filename)[0]

            target_path = ''

            if f_label[0] == 1:
                target_path = root_addr + directory_target_1 + filename_without_extension + '.BMP'
                # scipy.misc.toimage(f_image).save(target_path)
            elif f_label[0] == 2:
                target_path = root_addr + directory_target_2 + filename_without_extension + '.BMP'
                # scipy.misc.toimage(f_image).save(target_path)
            elif f_label[0] == 3:
                target_path = root_addr + directory_target_3 + filename_without_extension + '.BMP'
                # scipy.misc.toimage(f_image).save(target_path)
            else:
                print("Something is not right")

            # scipy.misc.toimage(f_image).save(target_path)
            patient_data = [filename_without_extension, f_pid, int(f_label[0])]
            patients_list.append(patient_data)

    else:
        continue

patients_filename = '//patients_data.csv'
with open((root_addr + patients_filename), mode='w', encoding='utf-8') as outputfile:
    ofilewriter = csv.writer(outputfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

    ofilewriter.writerow(["filename", "patient_id", "tumor_label"])
    ofilewriter.writerows(patients_list)

print("Finished! ")
