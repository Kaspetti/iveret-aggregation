"""
File: main.py
Author: Kaspar Tvedt Moberg
Email: kaspartmoberg@gmail.com
Github: https://github.com/kaspetti
Description: Main file for local testing fo statistical aggregation visualization of jet and MTA lines for iveret
"""


from aggregation import aggregate
from cluster import cluster_centroids, from_geo, hierarchical_cluster
from fitting import pw_evaluate, pw_fitting
from line_reader import read_lines

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as crs
import cartopy.feature as cfeature
import colorcet as cc


SIM_START = "2025052200"
LINE_TYPE = "mta"
t_offset = 0

colors = cc.b_glasbey_bw


if __name__ == "__main__":
	# fig = plt.figure(figsize=(16,9))
	# ax = fig.add_subplot(projection=crs.PlateCarree())
	# ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="black")   # type: ignore
	# ax.add_feature(cfeature.OCEAN, facecolor="lightgrey")     # type: ignore 
	# ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor="darkgrey")    # type: ignore

	lines = read_lines(SIM_START, LINE_TYPE)

	start_time = np.datetime64(
			f"{SIM_START[0:4]}-{SIM_START[4:6]}-{SIM_START[6:8]}T{SIM_START[8:10]}:00:00"
	)

	t = start_time + np.timedelta64(t_offset, "h")
	lines_t = lines.where(lines["date"] == t).dropna(how="all")

	line_ids = []
	grouped_lines = lines_t.groupby("line_id")

	segment_detail = 5
	segments = 5
	degree = 3

	total_line_detail = segments * segment_detail - (segments - 1)
	line_pts_fitted = np.zeros((len(grouped_lines), total_line_detail, 3))
	centroids = np.zeros((len(grouped_lines), 3))

	for i, (line_id, line) in enumerate(grouped_lines):
		coords = line[["latitude", "longitude"]].to_numpy()
		pts = from_geo(coords)
		centroids[i] = np.mean(pts, axis=0)
		line_ids.append(line_id)

		# cx, cy, cz, ts = pw_fitting(pts, degree, segments)
		# pts_fitted = pw_evaluate(cx, cy, cz, ts, segment_detail)
		# line_pts_fitted[i] = pts_fitted

	clusters = cluster_centroids(centroids, 15, 30)
	cluster_mapping = pd.DataFrame({
		"line_id": line_ids,
		"cluster": clusters
	})
	lines_with_clusters = lines_t.merge(cluster_mapping, on='line_id', how='left')

	selected_cluster = 0
	cluster_lines = lines_with_clusters.where(lines_with_clusters["cluster"] == selected_cluster).dropna(how="all")

	k = 3
	local_clusters = hierarchical_cluster(cluster_lines, k)

	for i in range(1):
		unique_line_ids = cluster_lines["line_id"].unique()
		line_ids_in_local_cluster = unique_line_ids[local_clusters == i]

		lines_from_local_cluster: pd.DataFrame = cluster_lines[cluster_lines["line_id"].isin(line_ids_in_local_cluster)]	# type: ignore
		aggregate(lines_from_local_cluster)

	# 	for j, (_, line) in enumerate(lines_from_local_cluster.groupby("line_id")):
	# 		coords = line[["latitude", "longitude"]].to_numpy()
	# 		if j == longest_id:
	# 			ax.plot(coords[:, 1], coords[:, 0], c=colors[i], linewidth=3)
	# 		else:
	# 			ax.plot(coords[:, 1], coords[:, 0], c=colors[i] + "33")
	#
	# 		if j == start_id:
	# 			ax.scatter(coords[0, 1], coords[0, 0], s=100, c=colors[i])
	# 		if j == end_id:
	# 			ax.scatter(coords[-1, 1], coords[-1, 0], s=100, c=colors[i])
	#
	# # ax.set_global()	# type: ignore
	# plt.tight_layout()
	# plt.show()
