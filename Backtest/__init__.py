
import matplotlib
matplotlib.use("agg")
try:
    matplotlib.pyplot.switch_backend('agg')
except AttributeError:
    pass
# matplotlib.use("TkAgg")

from .strategy import Strategy 
from .event import EventType
from .backtest import Backtest
from .data import OHLCDataHandler
from .portfolio import PortfolioHandler
from .execution import SimulatedExecutionHandler
from .performance import Performance
from .compliance import Compliance
