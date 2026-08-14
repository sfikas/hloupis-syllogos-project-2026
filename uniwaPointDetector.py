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

    assert(numVerticalPoints == 1 or numVerticalPoints == 2)
    assert(numGeneralPoints > 0 and numGeneralPoints <=4)
    assert(type(inputImage) == str)
    ### Κάποιες υπερπαράμετροι
    border_pixels_threshold = 15

    
    try:
        image = cv2.cvtColor(cv2.imread(inputImage), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        original_h, original_w = h, w
        scale_factor = 1.0
        if w != 1024:
            scale_factor = 1024 / w
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
    #cv2.imwrite(f'aaaa_{base_name}.png', foreground_mask.astype('uint8') * 255)

    ############# Count number of disconnected objects ###############
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground_mask.astype(np.uint8), connectivity=8
    )
    ## Remove the really small components -- those that are less than 10% of the biggest CC area
    remaining_labels = []
    if num_labels > 1:
        max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 0.1 * max_area:
                foreground_mask[labels == i] = False
            else:
                remaining_labels.append(i)

    ## Now 'remaining_labels' will contain labels, one for each detected object.
    #cv2.imwrite(f'bbbb_{base_name}.png', foreground_mask.astype('uint8') * 255)
    num_spotted_objects = len(remaining_labels)
    print(f'Εντόπισα {num_spotted_objects} αντικείμενα.')
    if debug_mode:
        print(f'(Ετικέτες {remaining_labels})')
    
    ## Get "mesokathetos" as the vertical line that roughly passes through the middle of the object
    verticalPoints_per_object = []
    generalPoints_per_object = []
    for label in remaining_labels:
        # Υποθέτω πάντα ότι μπορούν να υπάρχουν άνω του ενός αντικείμενα στη σκηνή
        ys, xs = np.where(labels == label)
        median_x = int(np.median(xs))
        vertical_ys = ys[xs == median_x]
        min_y, max_y = np.min(vertical_ys), np.max(vertical_ys)
        current_VerticalPoints = np.array([
            median_x,
            min_y
        ])
        # Αν έχουν ζητηθεί δύο σημεία της μεσοκαθέτου, επέστρεψε και το κατώτερο. 
        # Επιτρέπω μόνο δύο πιθανά σημεία
        if numVerticalPoints > 1:
            current_VerticalPoints = np.column_stack((current_VerticalPoints, np.array([
                median_x,
                max_y
            ])))
        verticalPoints_per_object.append(current_VerticalPoints)
        ## Και τώρα τα "γενικά" σημεία
        median_y = int(np.median(ys))
        horizontal_xs = xs[ys == median_y]
        lenxs = len(horizontal_xs)
        lenys = len(vertical_ys)
        current_GeneralPoints = np.array([
            [ horizontal_xs[int(.25 * lenxs)], horizontal_xs[int(.75 * lenxs)] ],
            [ median_y, median_y]
        ])
        current_GeneralPoints = np.column_stack((current_GeneralPoints, np.array([
                [ median_x, median_x],
                [ vertical_ys[int(.25 * lenys)], vertical_ys[int(.75 * lenys)] ]
            ])))
        # Κράτησε όσα μόνο ζητήθηκαν
        current_GeneralPoints = current_GeneralPoints[:, 0:numGeneralPoints]
        generalPoints_per_object.append(current_GeneralPoints)

    ## Όλη η δουλειά έχει γίνει -- το παρακάτω υπάρχει για οπτικοποίηση του αποτελέσματος.
    if debug_mode:
        overlay = image.copy()
        np.random.seed(randomSeed)
        alpha = 0.4
        for label in remaining_labels:
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            mask = labels == label
            overlay[mask] = (image[mask] * (1 - alpha) + color * alpha).astype(np.uint8)
        for vp in verticalPoints_per_object:
            pts = np.atleast_2d(vp)
            if pts.shape[0] != 2:
                pts = pts.T
            for i in range(pts.shape[1]):
                cv2.circle(overlay, (int(pts[0, i]), int(pts[1, i])), 5, (255, 0, 0), -1)
        for gp in generalPoints_per_object:
            pts = np.atleast_2d(gp)
            if pts.shape[0] != 2:
                pts = pts.T
            for i in range(pts.shape[1]):
                cv2.circle(overlay, (int(pts[0, i]), int(pts[1, i])), 5, (0, 0, 255), -1)
        cv2.imwrite(f'{base_name}_overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Rescale points back to original coordinates
    if scale_factor != 1.0:
        verticalPoints_per_object = [np.round(pts / scale_factor).astype(int) for pts in verticalPoints_per_object]
        generalPoints_per_object = [np.round(pts / scale_factor).astype(int) for pts in generalPoints_per_object]

    return verticalPoints_per_object, generalPoints_per_object


if __name__=='__main__':
    logger = logging.getLogger('UNIWA::PointDetector2026')
    logger.info('------------------------')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_image', '-i', required=True, help='Όνομα αρχείου εικόνας.')
    parser.add_argument('--numVerticalPoints', type=int, default=1, help='Αριθμός σημείων επί του αντικειμένου ενδιαφέροντος, τα οποία βρίσκονται επί της "μεσοκαθέτου". Το πρώτο σημείο βρίσκεται πιο κοντά στο άνω όριο της εικόνας, και το δεύτερο (εάν ζητείται) βρίσκεται πιο κοντά στο κάτω όριο της εικόνας.')
    parser.add_argument('--numGeneralPoints', type=int, default=4, help='Αριθμός σημείων επί του αντικειμένου ενδιαφέροντος, εκτός αυτών της "μεσοκαθέτου".')
    parser.add_argument('--debug', action='store_true', default=False, help='Αποθήκευση εικόνας με επισημείωση του αποτελέσματος.')

    args = parser.parse_args()

    debug_mode = args.debug
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
        verticalPoints, generalPoints = pointDetector(
            inputImage=img_path,
            numVerticalPoints=args.numVerticalPoints,
            numGeneralPoints=args.numGeneralPoints)

        print('Οι συντεταγμένες των προτεινόμενων σημείων είναι οι παρακάτω:')
        for i, (vp, gp) in enumerate(zip(verticalPoints, generalPoints)):
            print(f'Αντικείμενο {i+1}: μεσαία σημεία = {np.atleast_2d(vp).tolist()}, γενικά σημεία = {np.atleast_2d(gp).tolist()}')
        


    if not debug_mode and os.path.exists(binarizations_folder):
        rmtree(binarizations_folder)