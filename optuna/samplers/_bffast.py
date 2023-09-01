from dataclasses import dataclass
import decimal
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


@experimental_class("3.1.0")
class BFFast(BaseSampler):
    def __init__(self, seed: Optional[int] = None) -> None:
        self.params: List[str] = []
        self.param_candidates: Dict[str, List[Any]] = {}
        self.ALL_ITERATIONS_COUNT = 1
        return None

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if trial.number == 0:
            self.params.append(param_name)
            self.param_candidates[param_name] = _enumerate_candidates(param_distribution)
            self.ALL_ITERATIONS_COUNT *= len(self.param_candidates[param_name])
            return self.param_candidates[param_name][0]

        nu = trial.number
        idx = self.params.index(param_name)
        mult = 1
        for i in range(idx):
            mult *= len(self.param_candidates[self.params[i]])
        candidate_idx = (nu // mult) % len(self.param_candidates[param_name])
        val = self.param_candidates[param_name][candidate_idx]
        return val

    def before_trial(
        self,
        study: Study,
        trial: FrozenTrial,
    ):
        self.trial_params = {trial.number: {}}

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if trial.number + 1 == self.ALL_ITERATIONS_COUNT:
            study.stop()


def _enumerate_candidates(param_distribution: BaseDistribution) -> List[Any]:
    if isinstance(param_distribution, FloatDistribution):
        if param_distribution.step is None:
            raise ValueError(
                "FloatDistribution.step must be given for BruteForceSampler"
                " (otherwise, the search space will be infinite)."
            )
        low = decimal.Decimal(str(param_distribution.low))
        high = decimal.Decimal(str(param_distribution.high))
        step = decimal.Decimal(str(param_distribution.step))

        ret = []
        value = low
        while value <= high:
            ret.append(float(value))
            value += step

        return ret
    elif isinstance(param_distribution, IntDistribution):
        return list(
            range(param_distribution.low, param_distribution.high + 1, param_distribution.step)
        )
    elif isinstance(param_distribution, CategoricalDistribution):
        return list(param_distribution.choices)
    else:
        raise ValueError(f"Unknown distribution {param_distribution}.")
