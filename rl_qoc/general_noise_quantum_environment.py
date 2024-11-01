"""
Class to generate a General Context-Aware RL environment suitable for usage with Gym and PyTorch, 
leveraging Qiskit modules to simulate quantum system (could also include QUA code in the future)

Author: Aniket Chatterjee
Created on 01/11/2023
"""

from __future__ import annotations

import sys
from typing import (
    Dict,
    Optional,
    List,
    Any,
    TypeVar,
    SupportsFloat,
    Union,
    Tuple,
    Sequence,
    Callable,
)

import numpy as np
from gymnasium.spaces import Box
from qiskit import schedule, pulse, ClassicalRegister
from qiskit.circuit.library.standard_gates import (
    get_standard_gate_name_mapping as gate_map,
)
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.providers import BackendV2
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import (
    Layout,
    InstructionProperties,
    TransformationPass,
    CouplingMap,
)
from qiskit.transpiler.passes import FilterOpNodes
from qiskit_dynamics import DynamicsBackend
from qiskit_experiments.library import ProcessTomography
from qiskit_ibm_runtime import EstimatorV2

from .helper_functions import (
    get_instruction_timings,
    retrieve_neighbor_qubits,
    simulate_pulse_input,
)
from .qconfig import QEnvConfig
from .context_aware_quantum_environment import (
    create_array,
    target_instruction_timings,
    ContextAwareQuantumEnvironment,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s INFO %(message)s",  # hardcoded INFO level
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class 