import metpy as mpy
import metpy.calc as mpcalc
import pint
import math
import numpy as np
import sys

from ecape.calc import calc_ecape, calc_el_height
from metpy.units import check_units, units
from pint import UnitRegistry

from typing import Tuple
