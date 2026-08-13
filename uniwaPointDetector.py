# This code contains functionality that will allow the user to obtain
# a number of "interest points"* given an input color image.
#
# *(unrelated to keypoints in the conventional vision sense)
#
#
# Giorgos Sfikas
# University of West Attica, 2026

debug_mode = True
binarizations_folder = 'binarizations'

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch.load.*weights_only=False.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated, please import via timm\.layers",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.registry is deprecated, please import via timm\.models",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Overwriting tiny_vit.*in registry with.*This is because the name being registered conflicts with an existing name\. Please check if this is not expected\.",
    category=UserWarning,
    module=r"mobile_sam",
)

import logging
import argparse
import os
import urllib.request
import cv2
from tqdm import tqdm
from shutil import rmtree
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


def pointDetector(
        inputImage,
        numVerticalPoints,
        numGeneralPoints,
        randomSeed = 1):

    
    try:
        image = cv2.cvtColor(cv2.imread(inputImage), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if w != 1024:
            new_h = int(round(h * 1024 / w))
            image = cv2.resize(image, (1024, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print('\n============================================================')
        print(f"Πρόβλημα κατά την φόρτωση της εικόνας {inputImage}\nException {e}")
        print('============================================================')
        return None, None

    base_name = os.path.splitext(os.path.basename(inputImage))[0]
    binarization_filename = f'{binarizations_folder}/{base_name}_bin.png'
    if not os.path.exists(f'{binarizations_folder}'):
        os.makedirs(binarizations_folder)
    ############ Create binarization ###############
    if not os.path.exists(binarization_filename):
        checkpoint_path = "mobile_sam.pt"
        if not os.path.exists(checkpoint_path):
            url = f"https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/{checkpoint_path}"
            print(f"Downloading {checkpoint_path}...")
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=checkpoint_path) as t:
                def reporthook(blocknum, blocksize, totalsize):
                    if totalsize is not None:
                        t.total = totalsize
                    t.update(blocknum * blocksize - t.n)
                urllib.request.urlretrieve(url, checkpoint_path, reporthook=reporthook)

        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        sam.to(device="cpu")
        sam.eval()
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=8,
            points_per_batch=32
        )
        print(f'Creating binarization for input {inputImage}...')
        masks = mask_generator.generate(image)
        print(f'Done.')

        if masks:
            sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
            foreground_mask = sorted_masks[0]['segmentation']
            #background_mask = ~foreground_mask
            cv2.imwrite(binarization_filename, foreground_mask.astype('uint8') * 255)
            print(f'Saved binarization as {binarization_filename}.')
        else:
            foreground_mask, background_mask = None, None
    #### binarization already exists.
    else:
        print('Binarization already exists, using existing binarization.')
        foreground_mask = cv2.imread(binarization_filename, cv2.IMREAD_GRAYSCALE) > 0

    return None, None


if __name__=='__main__':
    logger = logging.getLogger('UNIWA::PointDetector2026')
    logger.info('------------------------')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_image', '-i', required=True, help='Όνομα αρχείου εικόνας.')
    parser.add_argument('--numVerticalPoints', type=int, default=1, help='Αριθμός σημείων επί του αντικειμένου ενδιαφέροντος, τα οποία βρίσκονται επί της "μεσοκαθέτου". Το πρώτο σημείο βρίσκεται πιο κοντά στο άνω όριο της εικόνας, και το δεύτερο (εάν ζητείται) βρίσκεται πιο κοντά στο κάτω όριο της εικόνας.')
    parser.add_argument('--numGeneralPoints', type=int, default=4, help='Αριθμός σημείων επί του αντικειμένου ενδιαφέροντος, εκτός αυτών της "μεσοκαθέτου".')
    args = parser.parse_args()

    if debug_mode:
        print('**** DEBUG MODE *****')

    verticalPoints, randomPoints = pointDetector(
        inputImage=args.input_image,
        numVerticalPoints=args.numVerticalPoints,
        numGeneralPoints=args.numGeneralPoints)

    print(f'')


    if not debug_mode and os.path.exists(binarizations_folder):
        rmtree(binarizations_folder)