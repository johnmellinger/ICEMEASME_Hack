import cv2
import numpy as np
import os,glob
from numpy import count_nonzero

experiment_folder = ['C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P52_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P54_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P55_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P01_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P02_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P03_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P04_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P05_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P07_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P08_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P10_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P11_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P14_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P15_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P16_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P17_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P18_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P20_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P21_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P37_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P38_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P39_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P40_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P42_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P43_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P44_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P45_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P46_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P47_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P48_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P49_layer0001_r8d_BMP',
'C://Users//johnn_000//Downloads//Problem2_Hackathon_IMECE2020//Problem2_Hackathon_IMECE2020//Image//DAQ_RHF_P51_layer0001_r8d_BMP',]

exp_number = ['52', '54', '55', '01', '02', '03', '04', '05', '07', '08', '10', '11', '14', '15', '16', '17', '18', '20', '21', '37', '38', '39', '40', '42', '43', '44', '45', '46', '47', '48', '49', '51']
# change pixel to area mm2 8 x 8 micron per pixel
pixel_to_area = 0.008 ** 2
exp_number_index = 0
for experiment in experiment_folder:
    out_file = open('meltpool' + 'P' + exp_number[exp_number_index] + '.txt', 'w')
    path = experiment
    for filename in glob.glob(os.path.join(path, '*.bmp')):
      with open(filename, 'r') as f:
        img = cv2.imread(filename,0)
        r, thresh1 = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
        out_file.write(str(pixel_to_area * count_nonzero(thresh1)) + '\n')
    exp_number_index += 1
    out_file.close()

##cv2.imshow('image', thresh1)
##img = cv2.imread('frame_000282.bmp',0)


