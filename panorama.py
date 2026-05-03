import numpy as np
import cv2


class Panaroma:
    def image_stitch(self, images, lowe_ratio=0.75, max_Threshold=4.0, match_status=False):
        """
        Stitch two images together using SIFT features, RANSAC homography, and warp perspective.

        Args:
            images: list/tuple of two images [imageB (left), imageA (right)]
            lowe_ratio: Lowe's ratio test threshold for filtering matches (default 0.75)
            max_Threshold: RANSAC reprojection error threshold (default 4.0)
            match_status: if True, also return a visualization of matched keypoints

        Returns:
            If match_status=False: stitched result image
            If match_status=True: (stitched result image, matched keypoints visualization)
            None if stitching fails (not enough matches)
        """
        (imageB, imageA) = images

        # Validate inputs
        if imageA is None or imageB is None:
            raise ValueError("One or both input images are None. Check file paths.")

        # Convert to grayscale for SIFT feature detection
        imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) if len(imageA.shape) == 3 else imageA
        imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY) if len(imageB.shape) == 3 else imageB

        # Detect features and keypoints using SIFT
        (key_points_A, features_of_A) = self.detect_feature_and_keypoints(imageA_gray)
        (key_points_B, features_of_B) = self.detect_feature_and_keypoints(imageB_gray)

        if features_of_A is None or features_of_B is None:
            print("[ERROR] Could not extract features from one or both images.")
            return None

        # Get valid matched keypoints
        Values = self.match_keypoints(
            key_points_A, key_points_B,
            features_of_A, features_of_B,
            lowe_ratio, max_Threshold
        )

        if Values is None:
            print("[ERROR] Not enough matching keypoints found between images.")
            return None

        # Warp imageA onto imageB using the computed homography
        (matches, Homography, status) = Values
        result_image = self.get_warp_perspective(imageA, imageB, Homography)
        result_image[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # Crop out black borders introduced by warping
        result_image = self.crop_black_borders(result_image)

        if match_status:
            vis = self.draw_matches(imageA, imageB, key_points_A, key_points_B, matches, status)
            return result_image, vis

        return result_image

    # ------------------------------------------------------------------
    # Warping
    # ------------------------------------------------------------------

    def get_warp_perspective(self, imageA, imageB, Homography):
        """Warp imageA into the coordinate frame of imageB using the Homography matrix."""
        width = imageA.shape[1] + imageB.shape[1]
        height = max(imageA.shape[0], imageB.shape[0])
        result_image = cv2.warpPerspective(imageA, Homography, (width, height))
        return result_image

    # ------------------------------------------------------------------
    # Feature Detection
    # ------------------------------------------------------------------

    def detect_feature_and_keypoints(self, image):
        """
        Detect SIFT keypoints and compute descriptors for the given grayscale image.

        Returns:
            (keypoints as float32 array of (x, y), feature descriptors)
        """
        sift = cv2.SIFT_create()
        (keypoints, features) = sift.detectAndCompute(image, None)

        if len(keypoints) == 0:
            return np.array([]), None

        keypoints = np.float32([kp.pt for kp in keypoints])
        return keypoints, features

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def get_all_possible_matches(self, featuresA, featuresB):
        """Use brute-force KNN matcher (k=2) to find candidate matches for Lowe's ratio test."""
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        all_matches = matcher.knnMatch(featuresA, featuresB, 2)
        return all_matches

    def get_all_valid_matches(self, all_matches, lowe_ratio):
        """
        Apply Lowe's ratio test: keep a match only if the nearest neighbour distance
        is significantly smaller than the second-nearest neighbour distance.
        This filters out ambiguous matches.
        """
        valid_matches = []
        for val in all_matches:
            if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
                valid_matches.append((val[0].trainIdx, val[0].queryIdx))
        return valid_matches

    def compute_homography(self, pointsA, pointsB, max_Threshold):
        """Compute homography matrix using RANSAC to handle outliers."""
        (homography, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, max_Threshold)
        return homography, status

    def match_keypoints(self, keypointsA, keypointsB, featuresA, featuresB, lowe_ratio, max_Threshold):
        """
        Full matching pipeline:
          1. Find all KNN matches between feature descriptors
          2. Filter with Lowe's ratio test
          3. Compute homography with RANSAC

        Returns:
            (valid_matches, homography_matrix, inlier_status) or None if insufficient matches
        """
        all_matches = self.get_all_possible_matches(featuresA, featuresB)
        valid_matches = self.get_all_valid_matches(all_matches, lowe_ratio)

        if len(valid_matches) <= 4:
            return None

        points_A = np.float32([keypointsA[i] for (_, i) in valid_matches])
        points_B = np.float32([keypointsB[i] for (i, _) in valid_matches])

        (homography, status) = self.compute_homography(points_A, points_B, max_Threshold)

        if homography is None:
            return None

        return valid_matches, homography, status

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_image_dimension(self, image):
        """Return (height, width) of an image."""
        return image.shape[:2]

    def get_points(self, imageA, imageB):
        """Create a side-by-side canvas containing both images."""
        (hA, wA) = self.get_image_dimension(imageA)
        (hB, wB) = self.get_image_dimension(imageB)
        canvas = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        canvas[0:hA, 0:wA] = imageA
        canvas[0:hB, wA:] = imageB
        return canvas, wA

    def draw_matches(self, imageA, imageB, keypointsA, keypointsB, matches, status):
        """
        Draw green lines between matched inlier keypoints on a side-by-side canvas.

        Returns:
            Visualization image with match lines
        """
        vis, wA = self.get_points(imageA, imageB)

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(keypointsA[queryIdx][0]), int(keypointsA[queryIdx][1]))
                ptB = (int(keypointsB[trainIdx][0]) + wA, int(keypointsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        return vis

    # ------------------------------------------------------------------
    # Post-Processing
    # ------------------------------------------------------------------

    def crop_black_borders(self, image):
        """
        Remove black borders left over after warpPerspective.

        Converts to grayscale, thresholds to find non-black pixels,
        then crops to the tightest bounding box.

        Returns:
            Cropped image, or original image if cropping is not possible
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        col_sums = np.sum(thresh, axis=0)
        row_sums = np.sum(thresh, axis=1)

        non_zero_cols = np.where(col_sums > 0)[0]
        non_zero_rows = np.where(row_sums > 0)[0]

        if len(non_zero_cols) == 0 or len(non_zero_rows) == 0:
            return image

        x_start, x_end = non_zero_cols[0], non_zero_cols[-1]
        y_start, y_end = non_zero_rows[0], non_zero_rows[-1]

        return image[y_start:y_end + 1, x_start:x_end + 1]
