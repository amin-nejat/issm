# -*- coding: utf-8 -*-
"""
@author: Amin
"""


from jaxtyping import Array, Float
from typing import NamedTuple


# %%
class ParamsLinearEmissions(NamedTuple):
    C: Float[Array, "emission_dim state_dim"]

class ParamsNNEmissions(NamedTuple):
    theta: tuple

class ParamsNormal(NamedTuple):
    mu: Float[Array, "emission_dim"]
    scale_tril: Float[Array, "emission_dim emission_dim"]

class ParamsConditionalNormal(NamedTuple):
    scale_tril: Float[Array, "emission_dim emission_dim"]

class ParamsLinearDynamics(NamedTuple):
    scale_tril: Float[Array, "state_dim state_dim"]
    A: Float[Array, "state_dim state_dim"]
    B: Float[Array, "input_dim state_dim"]
    initial: ParamsNormal

class ParamsfLDS(NamedTuple):
    emissions: ParamsNNEmissions
    dynamics: ParamsLinearDynamics
    likelihood: ParamsConditionalNormal

class ParamsVariationalLSTM(NamedTuple):
    theta_mu: tuple
    theta_scale: tuple
    B: Float[Array, "input_dim state_dim"]