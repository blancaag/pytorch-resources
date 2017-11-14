import eos
import numpy as np
import sys
import os
import cv2
import dlib
import glob
import numpy as np
from skimage import io
%matplotlib inline
import matplotlib.pyplot as plt

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def read_pts_file(filename):
    """A helper function to read ibug .pts landmarks from a file."""
    lines = open(filename).read().splitlines()
    lines = lines[3:71]

    landmarks = []
    for l in lines:
        coords = l.split()
        landmarks.append([float(coords[0]), float(coords[1])])
    return landmarks

def save_landmarks_as_pts(landmarks, path, f_name):
    # write data in a file.
    file_w = open(os.path.join(path, '{}.pts'.format(f_name)), 'w')

    header = ['version: 1 \n', \
              'n_points:  68 \n', \
              '{ \n']
    text = landmarks
    footer = ['}']

    file_w.writelines(header)
    file_w.writelines([str(i[0]) + ' ' + str(i[1]) + '\n' for i in text])
    file_w.writelines(footer)
    file_w.close() #to change file access modes
    
def compute_and_save_landmarks(img, output_path):
    
    face_det = dlib.get_frontal_face_detector()
    shape_pred = dlib.shape_predictor(predictor_path)    
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = face_det(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    
    non_detected_face_images_path = os.path.join(output_path, 'non_detected_face_images')
    detected_face_images_path = os.path.join(output_path, 'detected_face_images')
    if not os.path.exists(non_detected_face_images_path): os.mkdir(non_detected_face_images_path)
    if not os.path.exists(detected_face_images_path): os.mkdir(detected_face_images_path)
    
    f_name = f.split('/')[-1]
                  
    if len(dets) == 0: 
        cv2.imwrite(os.path.join(non_detected_face_images_path, f_name),  img)
    
    else: # len(dets) != 0: 
        cv2.imwrite(os.path.join(detected_face_images_path, f_name),  img)
    
        for _, i in enumerate(dets):
            (x0, y0, x1, y1) = i.left(), i.top(), i.right(), i.bottom();
            # get the landmarks/parts for the face in box d.
            landmarks = shape_to_np(shape_pred(img, i))
            landmarks_path = detected_face_images_path
            save_landmarks_as_pts(landmarks, landmarks_path, f_name.split('.')[0])

# /home/blanca/Documents/project/3DMM/4_eos/eos/python
# /home/blanca/Documents/project/3DMM/4_eos/install/bin/out.png
# /home/blanca/Documents/project/3DMM/4_eos/eos/python

def compute_3D_shape_and_texture(landmarks_path, path_to_eos):
    
    model = eos.morphablemodel.load_model(os.path.join(path_to_eos, "eos/share/sfm_shape_3448.bin"))
    blendshapes = eos.morphablemodel.load_blendshapes(os.path.join(path_to_eos, "eos/share/expression_blendshapes_3448.bin"))
    landmark_mapper = eos.core.LandmarkMapper(os.path.join(path_to_eos, 'eos/share/ibug_to_sfm.txt'))
    edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(path_to_eos, 'eos/share/sfm_3448_edge_topology.json'))
    contour_landmarks = eos.fitting.ContourLandmarks.load(os.path.join(path_to_eos, 'eos/share/ibug_to_sfm.txt'))
    model_contour = eos.fitting.ModelContour.load(os.path.join(path_to_eos, 'eos/share/model_contours.json'))
   
    for i in glob.glob(os.path.join(landmarks_path, "*.pts")):
        f_name = i.split('/')[-1].split('.')[0]
        print('Processing file: ', f_name)
        im = cv2.imread(os.path.join(landmarks_path, f_name + '.jpg'))
        """Demo for running the eos fitting from Python."""
        landmarks = read_pts_file(i)
        landmark_ids = list(map(str, range(1, 69))) # generates the numbers 1 to 68, as strings
        im_width = im.shape[0] #1280 # Make sure to adjust these when using your own images!
        im_height = im.shape[1] #1024

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(model, blendshapes,
            landmarks, landmark_ids, landmark_mapper, im_width, im_height,
            edge_topology, contour_landmarks, model_contour)

        # Now you can use your favourite plotting/rendering library to display the fitted mesh, using the rendering
        # parameters in the 'pose' variable.
        # Or for example extract the texture map, like this:

        isomap = eos.render.extract_texture(mesh, pose, im)
        print(os.path.join(landmarks_path, f_name + '.jpg'))
        cv2.imwrite(os.path.join(landmarks_path, f_name + '.isomap.jpg'), isomap)
        
path_to_eos = '/home/blanca/Documents/project/3DMM/4_eos/'

global_path = '/home/blanca/Documents/project/utils'
predictor_path = os.path.join(global_path, 'shape_predictor_68_face_landmarks.dat')
faces_folder_path =  os.path.join(global_path, 'data/new_dataset')

print("Total sample image faces: ", len(glob.glob(os.path.join(faces_folder_path, "*.jpg"))))

local_path = os.getcwd()
input_path = os.path.join(local_path, 'input')
output_path = os.path.join(local_path, 'output')

if not os.path.exists(input_path): os.mkdir(input_path)
if not os.path.exists(output_path): os.mkdir(output_path)

sample_size = 100
        
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg"))[:sample_size]:
    print("Processing file: {}".format(f))
    im = io.imread(f)     
    compute_and_save_landmarks(im, output_path)
    
non_detected_face_images_path = os.path.join(output_path, 'non_detected_face_images')
detected_face_images_path = os.path.join(output_path, 'detected_face_images')
       
print("No-face-detected: ", len(glob.glob(os.path.join(non_detected_face_images_path, "*.jpg"))))
print("Face-detected: ", len(glob.glob(os.path.join(detected_face_images_path, "*.jpg"))))

landmarks_path = '/home/blanca/Documents/project/3DMM/4_eos/ba_test/output/detected_face_images'
compute_3D_shape_and_texture(landmarks_path, path_to_eos)
