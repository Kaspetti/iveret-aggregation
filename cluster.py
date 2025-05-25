from numpy.typing import NDArray
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster


def from_geo(coords: NDArray[np.floating]) -> NDArray[np.floating]:
	"""Converts a list of latitude/longitude coordinates to 3D

	Parameters:
		- coords -- An NDArray of latitude/longitude pairs to convert to 3D.
			The NDArray must be of shape (n, 2).

	Returns:
		An NDArray with the latitude/longitude pairs converted to 3D.
	"""

	if coords.ndim == 1:
		coords = coords.reshape(1, coords.shape[0])

	if coords.shape[1] != 2:
		raise ValueError("Invalid shape of 'coords'. Must be (n, 2).")

	xs = np.cos(np.radians(coords[:, 0])) * np.cos(np.radians(coords[:, 1]))
	ys = np.cos(np.radians(coords[:, 0])) * np.sin(np.radians(coords[:, 1]))
	zs = np.sin(np.radians(coords[:, 0]))

	return np.column_stack((xs, ys, zs))


def to_geo(coords: NDArray[np.floating]) -> NDArray[np.floating]:
	"""Converts a list of 3D coordinates to latitude/longitude

	Parameters:
		- coords -- An NDArray of 3D coordinates to convert to latitude/longitude.
			The NDArray must be of shape (n, 3).

	Returns:
		An NDArray of the 3D coordinates converted to latitude/longitude pairs.
	"""

	if coords.ndim == 1:
		coords = coords.reshape(1, coords.shape[0])

	if coords.shape[1] != 3:
		raise ValueError("Invalid shape of 'coords'. Must be (n, 3).")

	radii = coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2
	zs_normalized = np.maximum(-1.0, np.minimum(1.0, coords[:, 2] / radii))

	lats = np.degrees(np.asin(zs_normalized))
	lons = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))

	return np.column_stack((lats, lons))


def get_centroid(coords: NDArray[np.floating]) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
	pts = from_geo(coords)
	mean_pt = np.mean(pts, axis=0)

	mean_coord = to_geo(mean_pt)
	
	return mean_coord, mean_pt


def cluster_centroids(centroids: NDArray[np.float64], min_k: int, max_k: int) -> NDArray[np.float64]:
	inertias: list[float | None]= [KMeans(n_clusters=k, random_state=1, n_init="auto").fit(centroids).inertia_ for k in range(min_k, max_k)]	# type: ignore

	kneedle = KneeLocator(range(min_k, max_k), inertias, S=1.0, curve="convex", direction="decreasing")	# type: ignore

	kmeans = KMeans(n_clusters=kneedle.elbow, random_state=0, n_init="auto").fit(centroids) # type: ignore

	return kmeans.labels_# type: ignore


def hierarchical_cluster(lines: pd.DataFrame, k: int) -> NDArray[np.float64]:
	lines_grouped = lines.groupby(["line_id"])
	line_amount = len(lines_grouped)
	vert_count = lines_grouped.size().max()

	A = np.zeros((line_amount, vert_count * 3))

	for i, (_, line) in enumerate(lines_grouped):
		coords = line[["latitude", "longitude"]].to_numpy()
		pts = from_geo(coords)
		repeat = np.repeat(pts[-1:], vert_count - pts.shape[0], axis=0)
		pts_repeated = np.concatenate((pts, repeat), axis=0)

		A[i] = pts_repeated.reshape(-1)

	pca = PCA(n_components=4).fit(A)
	lines_pca = pca.transform(A)

	hierarchy = linkage(lines_pca, method="average")

	if k:
		return np.subtract(fcluster(hierarchy, t=k, criterion="maxclust"), 1)
	else:
		raise NotImplemented

    # for i, line in enumerate(lines):
    #     coords = np.array([coord.to_3D().to_list() for coord in line.coords]) 
    #     repeat = np.repeat(coords[-1:], vert_count - len(coords), axis=0)
    #     coords_repeated = np.concatenate((coords, repeat), axis=0)
    #
    #     A[i] = coords_repeated.reshape(-1)


# def hierarchical_cluster(lines: NDArray[np.float64], k: int):
# 	A = lines.reshape((lines.shape[0], lines.shape[1] * lines.shape[2]))
#
# 	pca = PCA(n_components=4).fit(A)
# 	lines_pca = pca.transform(A)
#
# 	hierarchy = linkage(lines_pca, method="average")
#
# 	if k:
# 		return np.subtract(fcluster(hierarchy, t=k, criterion="maxclust"), 1)
# 	else:
# 		raise NotImplemented
