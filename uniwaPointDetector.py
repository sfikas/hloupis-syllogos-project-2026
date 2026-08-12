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


def pointDetector(
        inputImage,
        numVerticalPoints,
        numGeneralPoints,
        randomSeed = 1):
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