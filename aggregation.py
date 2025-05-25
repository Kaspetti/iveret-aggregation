import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cartopy.crs as crs

from cluster import from_geo, to_geo
from fitting import pw_evaluate, pw_fitting

def aggregate(lines: pd.DataFrame):
	fig = plt.figure(figsize=(16, 9))
	ax = fig.add_subplot(projection=crs.PlateCarree())

	lines_grouped = lines.groupby("line_id")
	lines_pts: list[NDArray[np.floating]] = []
	for _, line in lines_grouped:
		coords: NDArray[np.floating] = np.column_stack((line["latitude"].values, line["longitude"].values))	# type: ignore
		lines_pts.append(from_geo(coords))
	
	lines_pts = align_lines(lines_pts)

	start_id, end_id = identify_start_end_line(lines_pts)
	param_pts = lines_pts[start_id]
	
	segment_id = -1
	while segment_id != end_id:
		next_segment, segment_id = identify_next_segment(param_pts, lines_pts)
		param_pts = np.concatenate((param_pts, next_segment))

	param_ts = get_ts(param_pts)

	for line in lines_pts:
		line_ts = get_line_ts(line, param_pts, param_ts)

		cx, cy, cz, ts = pw_fitting(line, 3, 3, line_ts)
		pts_fitted = pw_evaluate(cx, cy, cz, ts, 10)
		coords_fitted = to_geo(pts_fitted)

		ax.plot(coords_fitted[:, 1], coords_fitted[:, 0])

	plt.show()

	return start_id, end_id


def get_ts(line_pts: NDArray[np.floating]) -> NDArray[np.floating]:
	"""Assigns points a t value in the range 0-1 by how far along the line the point is.

	Parameters:
		- line_pts -- An NDArray of points to generate the t values by.

	Returns:
		An NDArray of the same length as line_pts containing the t value for each point.
	"""

	total_length = 0
	dists: list[float] = []
	for i in range(line_pts.shape[0]-1):
		l0 = line_pts[i]
		l1 = line_pts[i+1]
		dist = np.sqrt(np.sum((l0 - l1)**2))

		dists.append(dist)
		total_length += dist

	ts = np.zeros(len(dists) + 1)
	cur_dist = 0

	for i, dist in enumerate(dists):
		ts[i] = cur_dist / total_length
		cur_dist += dist

	ts[-1] = cur_dist / total_length

	return ts


def get_line_ts(line: NDArray[np.floating], param_line: NDArray[np.floating], param_ts: NDArray[np.floating]) -> NDArray[np.floating]:
	ts = np.zeros(line.shape[0])

	for i, pt in enumerate(line):
		shortest_dist = float("inf")
		shortest_t = 0
		for j in range(param_line.shape[0]-1):
			c0 = param_line[j]
			c1 = param_line[j+1]

			AB = c1 - c0
			AC = pt - c0
			t = np.dot(AB, AC) / np.dot(AB, AB)

			if t <= 0:
				dist = np.sum((pt - c0)**2)
				t_param = param_ts[j]
			elif t >= 1:
				dist = np.sum((pt - c1)**2)
				t_param = param_ts[j+1]
			else:
				dist = np.sum((pt - c0 + AB*t)**2)
				t_param = param_ts[j] + (param_ts[j+1] - param_ts[j]) * t

			if dist < shortest_dist:
				shortest_dist = dist
				shortest_t = t_param

		ts[i] = shortest_t

	return ts




def align_lines(lines: list[NDArray[np.floating]]) -> list[NDArray[np.floating]]:
	"""Flips lines to align them to the same direction

	Checks if lines all follow the same direction as the first line in the list and 
	flips them if that is not the case.

	Parameters:
		- lines -- The lines to align.

	Return:
		The lines aligned across a common direction.
	"""

	desired_direction = lines[0][-1] - lines[0][0]

	for i, line in enumerate(lines):
		direction = line[-1] - line[0]
		if np.dot(direction, desired_direction) < 0:
			lines[i] = np.flip(line)

	return lines


def identify_next_segment(prev_segment: NDArray[np.floating], lines: list[NDArray[np.floating]]) -> tuple[NDArray[np.floating], np.intp]:
	end_pt = prev_segment[-1]
	remaining_pts = np.zeros(len(lines), dtype=int)

	for i, line_pts in enumerate(lines):
		dists = np.sum((line_pts - end_pt)**2, axis=1)
		closest_pt = np.argmin(dists)
		pts_after = line_pts.shape[0] - closest_pt
		remaining_pts[i] = pts_after

	next_id = np.argmax(remaining_pts)
	return lines[next_id][-remaining_pts[next_id]+1:], next_id


def identify_start_end_line(lines: list[NDArray[np.floating]]) -> tuple[np.intp, np.intp]:
	"""Identifies the start and end lines in a collections of lines.
	
	This checks each lines start/end points distance to the other lines. The lines
	with the greatest total distance from its start/end points is set to be the
	start/end line in the collection.

	Parameters:
		- lines -- The lines to identify start and end of.
	
	Returns:
		A tuple containing the index of the start line and the index of the end line.
	"""

	dists_start = np.zeros(len(lines))	
	dists_end = np.zeros(len(lines))	
	for i, l0 in enumerate(lines):
		start = l0[0]
		end = l0[-1]

		for l2 in lines:
			shortest_start = float("inf")
			shortest_end = float("inf")
			for j in range(l2.shape[0]-1):
				c0 = l2[j]
				c1 = l2[j+1]

				AB = c1 - c0

				AC_start = c0 - start
				t_start = np.dot(AB, AC_start) / np.dot(AB, AB)

				if t_start <= 0:
					dist = np.sum((start - c0)**2)
				elif t_start >= 1:
					dist = np.sum((start - c1)**2)
				else:
					dist = np.sum((start - c0 + AB*t_start)**2)

				shortest_start = min(shortest_start, dist)

				AC_end = c0 - end
				t_end = np.dot(AB, AC_end) / np.dot(AB, AB)

				if t_end <= 0:
					dist = np.sum((end - c0)**2)
				elif t_end >= 1:
					dist = np.sum((end - c1)**2)
				else:
					dist = np.sum((end - c0 + AB*t_end)**2)

				shortest_end = min(shortest_end, dist)

			dists_start[i] += shortest_start
			dists_end[i] += shortest_end

	return np.argmax(dists_start), np.argmax(dists_end)
