from patch_extraction.patch_extractor_casia import PatchExtractorCASIA
from patch_extraction.patch_extractor_nc import PatchExtractorNC
#from extraction_utils import check_and_reshape, extract_all_patches, create_dirs, save_patches, find_tampered_patches

#from src.patch_extraction import extraction_utils

# CASIA Dataset
#mode='no_rot' #for no rotations
pe = PatchExtractorCASIA(input_path='../data/CASIA2', output_path='patches_casia_with_rot',
                         patches_per_image=2, stride=128, rotations=4, mode='rot')
# pe = PatchExtractorCASIA(input_path=r"C:\Users\preet\Desktop\Image-Forgery-Detection-CNN-master\data\CASIA2", output_path='patches_casia_with_rot',
#                          patches_per_image=2, stride=128, rotations=4, mode='rot')
pe.extract_patches()

# Casia Dataset
# mode='no_rot' for no rotations
# pe = PatchExtractorNC(input_path='../data/NC2016/', output_path='patches_nc_with_rot',
#                       patches_per_image=2, stride=32, rotations=4, mode='rot')
# pe.extract_patches()
