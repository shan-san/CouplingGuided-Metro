"""Local CPLEX adapter for MetroRL's existing linear formulations.

Expressions, variables and constraints are native DOcplex objects. This module
keeps the small model/result interface used by the experiment runners, not a
replacement implementation of gurobipy. Status numbers retain the existing
result contract; the raw CPLEX status and parameter translations are explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
import sys

import cplex
from docplex.mp.constants import EffortLevel, WriteLevel
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model as _Model
from docplex.mp.utils import DOcplexException


BINARY, INTEGER, CONTINUOUS = "B", "I", "C"
MINIMIZE, MAXIMIZE = 1, -1
LOADED, OPTIMAL, INFEASIBLE, INF_OR_UNBD, UNBOUNDED = 1, 2, 3, 4, 5
NODE_LIMIT, TIME_LIMIT, SOLUTION_LIMIT, INTERRUPTED, NUMERIC, SUBOPTIMAL = 8, 9, 10, 11, 12, 13
MEM_LIMIT = 17


class SolverError(RuntimeError):
    pass


def normalize_status(code, has_incumbent=False):
    if code in (1, 101, 102, 129, 130):
        return OPTIMAL
    if code in (3, 103):
        return INFEASIBLE
    if code in (4, 119):
        return INF_OR_UNBD
    if code in (2, 118):
        return UNBOUNDED
    if code in (11, 25, 107, 108, 131, 132):
        return TIME_LIMIT
    if code in (13, 113, 114):
        return INTERRUPTED
    if code in (105, 106):
        return NODE_LIMIT
    if code in (104, 128):
        return SOLUTION_LIMIT
    if code in (111, 112):
        return MEM_LIMIT
    if code in (5, 10, 109, 110, 115, 116, 117, 23, 127):
        return SUBOPTIMAL if has_incumbent else NUMERIC
    # An unrecognized status must not be accepted as a usable solve result.
    return LOADED


@dataclass
class Column:
    coefficients: list
    constraints: list


class Env:
    def __init__(self, empty=False):
        self._cplex = None
        if not empty:
            self.start()

    def start(self):
        try:
            if self._cplex is None:
                self._cplex = cplex.Cplex()
        except cplex.exceptions.CplexError as exc:
            raise SolverError(str(exc)) from exc

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._cplex is not None:
            self._cplex.end()


class _Parameters:
    def __init__(self, owner):
        object.__setattr__(self, "_owner", owner)

    def __setattr__(self, name, value):
        self._owner.setParam(name, value)

    def __getattr__(self, name):
        return self._owner.parameter_mapping[name]["requested"]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class Model:
    def __init__(self, name="metro"):
        self.native_model = _Model(name=name, checker="off", log_output=False)
        self.native_model.parameters.clocktype = 2  # Wall time, as in the CLI budgets.
        self.Params = _Parameters(self)
        self.parameter_mapping = {}
        self._starts, self._priorities = {}, {}
        self._output, self._console, self._log = True, True, None
        self._conflict_indices = set()
        self._conflict_refined = False
        self._disposed = False

    def sum(self, values):
        return self.native_model.sum(values)

    def linear_expr(self, *args, **kwargs):
        return self.native_model.linear_expr(*args, **kwargs)

    def addVar(self, lb=0., ub=None, obj=0., vtype=CONTINUOUS, name=None, column=None):
        if vtype == BINARY:
            variable = self.native_model.binary_var(name=name)
            variable.lb, variable.ub = lb, 1. if ub is None else ub
        elif vtype == INTEGER:
            variable = self.native_model.integer_var(lb=lb, ub=ub, name=name)
        elif vtype == CONTINUOUS:
            variable = self.native_model.continuous_var(lb=lb, ub=ub, name=name)
        else:
            raise SolverError(f"Unsupported variable type: {vtype}")
        if obj:
            self.set_variable_objective(variable, obj)
        if column is not None:
            if len(column.coefficients) != len(column.constraints):
                raise SolverError("Column coefficients and constraints must have equal length")
            for coefficient, constraint in zip(column.coefficients, column.constraints):
                left = constraint.left_expr
                left.set_coefficient(variable, left.get_coef(variable) + coefficient)
        return variable

    def addVars(self, *dimensions, lb=0., ub=None, obj=0., vtype=CONTINUOUS, name=""):
        if len(dimensions) == 1:
            keys = range(dimensions[0]) if isinstance(dimensions[0], int) else dimensions[0]
        else:
            keys = product(*(range(d) if isinstance(d, int) else d for d in dimensions))
        result = {}
        for key in keys:
            indices = key if isinstance(key, tuple) else (key,)
            variable_name = f"{name}[{','.join(map(str, indices))}]" if name else None
            def at(value):
                return value[key] if isinstance(value, dict) else value
            result[key] = self.addVar(lb=at(lb), ub=at(ub), obj=at(obj), vtype=vtype, name=variable_name)
        return result

    def addConstr(self, constraint, name=None):
        # Keep constant rows explicit: offline flow balances receive columns later.
        if isinstance(constraint, bool):
            constraint = self.linear_expr() == (0. if constraint else 1.)
        return self.native_model.add_constraint(constraint, ctname=name)

    def addGenConstrMax(self, result, variables, constant=None, name=None):
        operands = list(variables)
        if constant is not None:
            operands.append(constant)
        return self.addConstr(result == self.native_model.max(operands), name)

    def setObjective(self, expression, sense=MINIMIZE):
        self.native_model.set_objective("min" if sense == MINIMIZE else "max", expression)

    def set_variable_objective(self, variable, coefficient):
        expression = self.native_model.objective_expr
        if not isinstance(expression, LinearExpr):
            expression = self.linear_expr(dict(expression.iter_terms()), constant=expression.constant)
            self.native_model.set_objective_expr(expression)
        expression.set_coefficient(variable, coefficient)

    def set_start(self, variable, value):
        self._starts[variable] = value

    def set_priority(self, variable, value):
        if int(value) != value or not 0 <= value <= 2000000000:
            raise SolverError("Branch priority must be a nonnegative integer")
        self._priorities[variable] = int(value)

    def setParam(self, name, value):
        path, translated, exact = name, value, True
        direct = {"TimeLimit": "timelimit", "Threads": "threads", "MIPGap": "mip.tolerances.mipgap",
                  "MIPGapAbs": "mip.tolerances.absmipgap", "Seed": "randomseed",
                  "MIPFocus": "emphasis.mip"}
        if name in ("OutputFlag", "LogToConsole"):
            if value not in (0, 1):
                raise SolverError(f"{name} must be 0 or 1")
            setattr(self, "_output" if name == "OutputFlag" else "_console", bool(value))
            path = "log_streams"
        elif name == "LogFile":
            if self._log is not None:
                self._log.close()
            self._log = Path(value).open("w", encoding="utf-8") if value else None
            path = "log_streams.file"
        else:
            if name in direct:
                path = direct[name]
                if name == "TimeLimit" and value == math.inf:
                    translated = self.native_model.parameters.timelimit.default_value
            elif name == "Heuristics":
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise SolverError("Heuristics must be between zero and one")
                # Preserve relative effort intent, not Gurobi's time fraction.
                path, translated, exact = "mip.strategy.heuristiceffort", value / .05, False
            elif name == "RINS":
                path = "mip.strategy.rinsheur"
                translated = 0 if value == -1 else -1 if value == 0 else value
            elif name == "Presolve":
                if value not in (-1, 0, 1, 2):
                    raise SolverError("Presolve must be -1, 0, 1 or 2")
                path, translated, exact = "preprocessing.presolve", int(value != 0), value == 0
            elif name == "DualReductions":
                if value not in (0, 1):
                    raise SolverError("DualReductions must be 0 or 1")
                path, translated = "preprocessing.reduce", 1 if value == 0 else 3
            elif "." not in name and name not in ("timelimit", "threads", "randomseed", "clocktype"):
                raise SolverError(f"No CPLEX parameter mapping for {name}")
            try:
                parameter = self.native_model.parameters
                for part in path.split("."):
                    parameter = getattr(parameter, part)
                parameter.set(translated)
            except (AttributeError, ValueError, DOcplexException, cplex.exceptions.CplexError) as exc:
                raise SolverError(f"Invalid CPLEX parameter {name}: {exc}") from exc
        requested = "unlimited" if name == "TimeLimit" and value == math.inf else value
        self.parameter_mapping[name] = dict(requested=requested, cplex_parameter=path, value=translated,
                                            exact=exact)

    def update(self):
        pass  # DOcplex keeps the local CPLEX engine synchronized incrementally.

    def optimize(self):
        native = self.native_model
        if self._starts:
            native.clear_mip_starts()
            start = native.new_solution(var_value_dict=self._starts)
            native.add_mip_start(start, effort_level=EffortLevel.Repair,
                                 write_level=WriteLevel.AllVars, complete_vars=False)
        if self._priorities:
            native.get_cplex().order.set([(v.index, p, 0) for v, p in self._priorities.items()])
        streams = ([sys.stdout] if self._console else []) + ([self._log] if self._log else [])
        output = _Tee(*streams) if self._output and streams else False
        try:
            native.solve(log_output=output)
        except (DOcplexException, cplex.exceptions.CplexError) as exc:
            raise SolverError(str(exc)) from exc

    def getVarByName(self, name):
        return self.native_model.get_var_by_name(name)

    def getConstrs(self):
        return list(self.native_model.iter_linear_constraints())

    def remove(self, variable):
        # A zero-fixed frontier arc contributes neither flow nor cost. Retaining
        # its column keeps DOcplex/native indices valid during state expansion.
        variable.lb, variable.ub = 0., 0.
        self._starts.pop(variable, None)
        self._priorities.pop(variable, None)

    def computeIIS(self):
        try:
            engine = self.native_model.get_cplex()
            engine.conflict.refine(engine.conflict.all_constraints())
            groups = engine.conflict.get_groups()
            statuses = engine.conflict.get()
            self._conflict_indices = {index for (_, members), status in zip(groups, statuses)
                if status in (engine.conflict.group_status.member, engine.conflict.group_status.possible_member)
                for kind, index in members if kind == engine.conflict.constraint_type.linear}
            self._conflict_refined = True
        except cplex.exceptions.CplexError as exc:
            raise SolverError(str(exc)) from exc

    def is_conflict(self, constraint):
        return constraint.index in self._conflict_indices

    def write(self, path):
        engine = self.native_model.get_cplex()
        if self._conflict_refined and Path(path).suffix.lower() in (".ilp", ".clp"):
            engine.conflict.write(str(path))
        else:
            engine.write(str(path))

    @property
    def cplex_status(self):
        details = self.native_model.solve_details
        return details.status_code if details is not None else 0

    @property
    def Status(self):
        return normalize_status(self.cplex_status, bool(self.SolCount))

    status = property(lambda self: self.Status)
    SolCount = property(lambda self: int(self.native_model.solution is not None))
    ObjVal = property(lambda self: self.native_model.objective_value)
    objVal = property(lambda self: self.ObjVal)
    NumVars = property(lambda self: self.native_model.number_of_variables)
    NumConstrs = property(lambda self: self.native_model.number_of_constraints)

    @property
    def ObjBound(self):
        details = self.native_model.solve_details
        bound = details.best_bound if details is not None else math.nan
        if (self.Status == OPTIMAL and self.SolCount
                and self.native_model.get_cplex().get_problem_type() == cplex.Cplex.problem_type.LP):
            return self.ObjVal
        if bound is not None and not math.isnan(bound):
            return bound
        return -math.inf if self.native_model.is_minimized() else math.inf

    @property
    def MIPGap(self):
        if not self.SolCount:
            return math.inf
        value = self.native_model.solve_details.mip_relative_gap
        return 0. if self.Status == OPTIMAL and math.isnan(value) else value

    NodeCount = property(lambda self: self.native_model.solve_details.nb_nodes_processed or 0)
    IterCount = property(lambda self: self.native_model.solve_details.nb_iterations or 0)

    def dispose(self):
        if not self._disposed:
            self.native_model.end()
            if self._log is not None:
                self._log.close()
            self._disposed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()
