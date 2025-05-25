from typing import Literal

from numpy.typing import NDArray
import numpy as np
import xarray as xr
import pandas as pd


def read_lines(sim_start: str, line_type: Literal["jet", "mta"]) -> pd.DataFrame:
	"""Reads all lines of the given type at the simulation sim_start

	Parameters:
		- sim_start -- The start of the simulation to read from. At the format 'YYYYMMDDHH'.
		- line_type -- The type of line to get. One of 'jet' or 'mta'.
	
	Returns:
		A pandas dataframe containing the lines from all ensemble members at sim start.
	"""

	frames = []

	for i in range(50):
		base_path = f"./data/{line_type}/{sim_start}/"
		file_path = f"ec.ens_{i:02d}.{sim_start}.sfc.mta.nc"

		if line_type == "jet":
			file_path = f"ec.ens_{i:02d}.{sim_start}.pv2000.jetaxis.nc"
		full_path = base_path + file_path

		ds = xr.open_dataset(full_path)
		df = ds.to_dataframe()
		df["line_id"] = str(i) + "|" + df["line_id"].astype(str)

		frames.append(df)

	return pd.concat(frames, ignore_index=True, sort=False)


def dateline_fix(coords: NDArray[np.floating]) -> NDArray[np.floating]:
	"""Checks if a line crosses the anti meridian and shifts it by 360 if it does.

	Parameters:
		- coords -- List of latitude/longitude pairs making up a line.

	Returns:
		Shifted coordinates if the line crosses the anti meridian or the original array if not.
	"""

	if np.max(coords[:, 1]) - min(coords[:, 1]) > 180:
		coords[:, 1] = np.where(coords[:, 1] < 0, coords[:, 1] + 360, coords[:, 1])

	return coords
