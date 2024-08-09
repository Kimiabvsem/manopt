import autograd.numpy as anp
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import torch

import pymanopt
from examples._tools import ExampleRunner
from pymanopt import Problem
from pymanopt.manifolds import Sphere
from pymanopt.tools.diagnostics import check_hessian
