import torch
from cnn.cnn import CNN
from feature_fusion.feature_vector_generation import create_feature_vectors
import os
import os

authentic_path = r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2\Au'
tampered_path = r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2\Tp'

# Get the list of files in the 'Au' directory
authentic_files = os.listdir(authentic_path)
# Get the count of files in the 'Au' directory
authentic_count = len(authentic_files)

# Get the list of files in the 'Tp' directory
tampered_files = os.listdir(tampered_path)
# Get the count of files in the 'Tp' directory
tampered_count = len(tampered_files)

print("Number of files in Au directory:", authentic_count)
print("Number of files in Tp directory:", tampered_count)



# with torch.no_grad():
#     model = CNN()
#     model.load_state_dict(torch.load('../data/output/pre_trained_cnn/CASIA2_WithRot_LR001_b128_nodrop.pt',
#                                      map_location=lambda storage, loc: storage))
#     model.eval()
#     model = model.double()
#
#     #authentic_path = '../data/CASIA2/Au/*'
#     authentic_path = r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2\Au\*'
#     # tampered_path = '../data/CASIA2/Tp/*'
#     tampered_path = r'C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2\Tp\*'
#     output_filename = 'CASIA2_WithRot_LR001_b128_nodrop.csv'
#     create_feature_vectors(model, tampered_path, authentic_path, output_filename)
