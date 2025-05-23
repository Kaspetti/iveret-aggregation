"""
File: main.py
Author: Kaspar Moberg
Email: kaspartmoberg@gmail.com
Github: https://github.com/kaspetti
Description: Main file for local testing fo statistical aggregation visualization of jet and MTA lines for iveret
"""


import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass
class Settings:
    """Dataclass for holding the parameters passed to the program.

    Instance Variables:
    	- sim_start -- The start of the simulation in the format 'YYYYMMDDHH'.
    	- time_offset -- The time offset from the simulation start. In hours.
        - line_type -- The type of line to visualize. 'jet' or 'mta'.
    """

    sim_start: str
    time_offset: int
    line_type: Literal["jet", "mta"]


def init() -> Settings:
    valid_timeoffsets = list(range(0, 73, 3)) + list(range(78, 241, 6))

    parser = argparse.ArgumentParser("iveret line aggregation")
    _ = parser.add_argument("--simstart", type=str, default="2025052000", help="Start of the simulation in the format 'YYYYMMDDHH'")
    _ = parser.add_argument("--timeoffset", type=int, default=0, choices=valid_timeoffsets, help="Time offset from the simstart")
    _ = parser.add_argument("--linetype", type=str, default="jet", choices=["jet", "mta"], help="Type of line (must be 'jet' or 'mta')")

    args = parser.parse_args()
    settings = Settings(sim_start=args.simstart,
                        time_offset=args.timeoffset,
                        line_type=args.linetype)

    return settings


if __name__ == "__main__":
    settings = init()
    print(settings)
