"""
File: fitting.py
Author: Kaspar Tvedt Moberg
Email: kaspartmoberg@gmail.com
Github: https://github.com/kaspetti
Description: Functions for fitting lines to splines.
"""

from numpy.typing import NDArray
import numpy as np


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


def pw_fitting(
		pts: NDArray[np.floating], degree: int, segments: int, ts: NDArray[np.floating] | None = None
		) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
	"""Performs piecewise fitting of functions to points in 3D space.

	The functions are fitted per dimension and must be combined to get the function in 3D space.

	Parameters:
		- pts -- The points to fit the functions to.
		- degree -- The degree of each function.
		- segments -- The amount of functions to fit to the points.
		- sample_detail -- The amount of points along the fitted line returned. (default=100)

	Returns:
		A tuple containing the coefficients for x, y, and z dimensions. Aswell as the t ranges for each segment.
	"""
	
	if ts is None:
		print("hei")
		ts = get_ts(pts)

	n_ts = len(ts)

	if n_ts < 2 or segments <= 0:
		print(f"Warning: Cannot create {segments} segments with {n_ts} points.")
		return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
	if n_ts < segments:
		print(f"Warning: More segments ({segments}) than points ({n_ts}). Some segments will be empty.")

	all_indices = np.arange(n_ts)
	split_indices = np.array_split(all_indices, segments)

	segments_range = np.zeros((segments, 2))

	n_cs = degree + 1

	A = np.zeros((n_ts, n_cs * segments))

	for i in range(segments):
		segment_indices = split_indices[i]
		t_start = ts[segment_indices[0]]
		t_end = ts[segment_indices[-1]]
		segments_range[i][0] = t_start
		segments_range[i][1] = t_end

		mask = np.zeros(n_ts, dtype=bool)
		if len(segment_indices) > 0:
			mask[segment_indices] = True

		start = i * n_cs
		A[:, start] = np.where(mask, 1, 0)

		for j in range(1, degree + 1):
			A[:, start + j] = np.where(mask, ts**j, 0)

	polynomials_row = np.arange(0, degree+1)

	M = np.identity((degree + 1) * segments)
	constraint_rows = np.zeros(degree * (segments - 1), dtype=int)
	for i in range(segments - 1):
		mid_ts = np.repeat(segments_range[i][1], n_cs)

		coeffs = np.ones(polynomials_row.shape)
		for j in range(degree):
			new_polynomials = np.where(polynomials_row == 0, 0, polynomials_row - j)
			cj = mid_ts**new_polynomials * coeffs

			start = i * n_cs
			end = start + (2 * n_cs)

			constraint_row = (i * n_cs) + (n_cs * 2) - degree + j
			constraint_rows[i * degree + j] = constraint_row
			M[constraint_row, start:end] = np.append(cj, -cj)

			coeffs *= new_polynomials

	Ah: NDArray[np.float64] = np.matmul(A, np.linalg.pinv(M))
	Ah[:, constraint_rows] = 0
	cxh_best = np.linalg.lstsq(Ah, pts[:, 0], rcond=None)
	cyh_best = np.linalg.lstsq(Ah, pts[:, 1], rcond=None)
	czh_best = np.linalg.lstsq(Ah, pts[:, 2], rcond=None)

	cx_best = np.array(np.array_split(np.linalg.lstsq(M, cxh_best[0], rcond=None)[0], segments))
	cy_best = np.array(np.array_split(np.linalg.lstsq(M, cyh_best[0], rcond=None)[0], segments))
	cz_best = np.array(np.array_split(np.linalg.lstsq(M, czh_best[0], rcond=None)[0], segments))

	return cx_best, cy_best, cz_best, segments_range


def pw_evaluate(cx: NDArray[np.floating], cy: NDArray[np.floating], cz: NDArray[np.floating], t_ranges: NDArray[np.floating], segment_detail: int = 5) -> NDArray[np.floating]:
	if not (cx.shape[1] == cy.shape[1] == cz.shape[1]):
		raise ValueError("Each dimension (x,y,z) must have the same number of coefficients.")

	if not (cx.shape[0] == cy.shape[0] == cz.shape[0]):
		raise ValueError("Each dimension (x,y,z) must have the same number of segments.")

	segments = cx.shape[0]
	powers = np.arange(0, cx.shape[1])

	n_pts = (segments * segment_detail) - (segments - 1)
	n = 0

	pts = np.zeros((n_pts, 3))

	for i in range(segments):
		ts = np.linspace(t_ranges[i][0], t_ranges[i][1], segment_detail)
		for t in (ts if i == segments - 1 else ts[:-1]):
			t_pow = t**powers

			pt = np.array([np.sum(cx[i] * t_pow),
				  		   np.sum(cy[i] * t_pow),
				  		   np.sum(cz[i] * t_pow)])

			pts[n] = pt
			n += 1

	return pts
