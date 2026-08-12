# This code contains functionality that will allow the user to obtain
# a number of "interest points"* given an input color image.
#
# *(unrelated to keypoints in the conventional vision sense)
#
#
# Giorgos Sfikas
# University of West Attica, 2026

import logging
import argparse
import os
import urllib.request
import cv2
from tqdm import tqdm
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
        print(f"Πρόβλημα κατά την φόρτωση της εικόνας {inputImage}: {e}")
        return None, None

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
    masks = mask_generator.generate(image)

    if masks:
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        foreground_mask = sorted_masks[0]['segmentation']
        background_mask = ~foreground_mask
        print(f'saving "binarization.png"')
        cv2.imwrite('binarization.png', foreground_mask.astype('uint8') * 255)
    else:
        foreground_mask, background_mask = None, None


    return None, None


if __name__=='__main__':
    logger = logging.getLogger('UNIWA::PointDetector2026')
    logger.info('------------------------')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_image', '-i', required=True, help='Όνομα αρχείου εικόνας.')
    parser.add_argument('--numVerticalPoints', type=int, default=1, help='Αριθμός σημείων επί του αντικειμένου ενδιαφέροντος, τα οποία βρίσκονται επί της "μεσοκαθέτου". Το πρώτο σημείο βρίσκεται πιο κοντά στο άνω όριο της εικόνας, και το δεύτερο (εάν ζητείται) βρίσκεται πιο κοντά στο κάτω όριο της εικόνας.')
    parser.add_argument('--numGeneralPoints', type=int, default=4, help='Αριθμός σημείων επί του αντικειμένου ενδιαφέροντος, εκτός αυτών της "μεσοκαθέτου".')
    args = parser.parse_args()

    verticalPoints, randomPoints = pointDetector(
        inputImage=args.input_image,
        numVerticalPoints=args.numVerticalPoints,
        numGeneralPoints=args.numGeneralPoints)

    print(f'')