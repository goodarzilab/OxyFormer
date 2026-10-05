"""Pinned TabPFN v2 checkpoints under the approved 9.1.0 package.

No remote client, telemetry service, default checkpoint or download path is used.
The installed 9.1.0 loader explicitly disables downloads, then ModelSpecs passes
isolated in-memory weights to the estimator. Package imports remain lazy.
Supervised fits inherit the training-only LoadedData handoff and original-ID
alignment from TabICLComparator; prediction receives covariates only.
"""
from copy import deepcopy
from importlib import import_module

import torch

from oxyformer.models.tabicl_comparator import TabICLComparator
from oxyformer.provenance import ContractError, require


class TabPFNComparator(TabICLComparator):
    package = "tabpfn"
    # Conservative CPU registration; the package refuses larger CPU contexts
    # unless a global override is set. No automatic override or subsampling.
    max_context_rows = 1_000
    max_features = 500

    def _make_estimator(self, checkpoint_path):
        try:
            module = import_module("tabpfn")
            loader = import_module("tabpfn.model_loading")
        except ImportError as exc:
            raise ContractError("missing optional dependency tabpfn; request provisioning") from exc
        regression = self.family == "identity"
        models, criterion, configs, inference = loader.load_model_criterion_config(
            checkpoint_path, check_bar_distribution_criterion=regression,
            estimator_type="regressor" if regression else "classifier", version="v2",
            download_if_not_exists=False, n_estimators_override=8,
            devices=[torch.device("cpu")], force_inference_dtype=torch.float32)
        require(len(models) == len(configs) == 1, "exactly one pinned checkpoint required")
        # Loader caches may share checkpoint objects. Never share mutable
        # estimator/context/model state across fitted fold instances.
        model, config, inference, criterion = deepcopy((models[0], configs[0], inference, criterion))
        specs = module.ModelSpecs(model=model, architecture_config=config,
                                  inference_config=inference,
                                  norm_criterion=criterion if regression else None)
        cls = module.TabPFNRegressor if regression else module.TabPFNClassifier
        return cls(model_path=specs, device="cpu", inference_precision=torch.float32,
                   n_estimators=8, auto_scale_n_estimators=False, random_state=self.seed,
                   fit_mode="fit_preprocessors", n_preprocessing_jobs=1,
                   ignore_pretraining_limits=False, show_progress_bar=False)
