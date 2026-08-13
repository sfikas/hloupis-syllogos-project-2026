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
import numpy as np
from tqdm import tqdm
from shutil import rmtree
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator


def pointDetector(
        inputImage,
        numVerticalPoints,
        numGeneralPoints,
        randomSeed = 1):

    ### Κάποιες υπερπαράμετροι
    border_pixels_threshold = 15
    
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
            foreground_mask = None
            print('Πρόβλημα με τη δημιουργία του αρχείου δυαδικοποίησης.')
            return None, None
    #### binarization already exists.
    else:
        print('Το αρχείο δυαδικοποίησης υπάρχει ήδη, το χρησιμοποιώ..')
        try:
            foreground_mask = cv2.imread(binarization_filename, cv2.IMREAD_GRAYSCALE) > 0
        except:
            print('Πρόλημα με το άνοιγμα του αρχείου δυαδικοποίησης.')
            return None, None

    ############# Clean up the foreground mask with morphomat ##############
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_uint8 = foreground_mask.astype(np.uint8) * 255
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cleaned > 0
    #cv2.imwrite('aaaa.png', foreground_mask.astype('uint8') * 255)

    ############# Decide which is the background object ##############
    border_pixels = np.concatenate([
        foreground_mask[0:border_pixels_threshold, :].flatten(),
        foreground_mask[-border_pixels_threshold:-1, :].flatten(),
        foreground_mask[:, 0:border_pixels_threshold].flatten(),
        foreground_mask[:, -border_pixels_threshold-1].flatten()
    ])
    if np.mean(border_pixels) > 0.5:
        foreground_mask = ~foreground_mask
    cv2.imwrite(f'aaaa_{base_name}.png', foreground_mask.astype('uint8') * 255)

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

    if os.path.isdir(args.input_image):
        image_files = [
            os.path.join(args.input_image, f)
            for f in sorted(os.listdir(args.input_image))
            if f.endswith('.jpg') or f.endswith('.JPG')
        ]
    else:
        image_files = [args.input_image]

    for img_path in image_files:
        verticalPoints, randomPoints = pointDetector(
            inputImage=img_path,
            numVerticalPoints=args.numVerticalPoints,
            numGeneralPoints=args.numGeneralPoints)

    print(f'')


    if not debug_mode and os.path.exists(binarizations_folder):
        rmtree(binarizations_folder)