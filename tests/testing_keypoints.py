import my_keypoints
import numpy as np


def run_sample_test():
	# Boxes (same as the C++ sample)
	boxes = [[100.0, 150.0, 300.0, 450.0], [400.0, 200.0, 520.0, 380.0]]

	N = len(boxes)
	K = 2
	H = 5
	W = 5

	# Create a zeroed (N, K, H, W) float32 array
	heatmaps = np.zeros((N, K, H, W), dtype=np.float32)

	# Set peaks similar to the C++ sample: positions chosen inside the 5x5 grid
	# For variety, place peaks at (y, x) coordinates: (3,2), (4,1), etc.
	heatmaps[0, 0, 3, 2] = 1.0
	heatmaps[0, 1, 2, 1] = 0.8
	heatmaps[1, 0, 1, 4] = 0.9
	heatmaps[1, 1, 4, 3] = 1.0

	try:
		# Use the numpy-aware wrapper we added
		result = my_keypoints.keypoints_from_heatmaps_with_boxes_np(heatmaps, boxes, 1.25)

		print("Test successful! Sample outputs:")
		for n in range(len(result.preds)):
			print(f"Person {n}:")
			for k in range(len(result.preds[n])):
				x = result.preds[n][k][0]
				y = result.preds[n][k][1]
				score = result.maxvals[n][k]
				print(f"  Keypoint {k}: ({x}, {y}) with score {score}")
	except Exception as e:
		print("Error:", e)


if __name__ == "__main__":
	run_sample_test()