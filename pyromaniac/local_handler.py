import numpy as np
import pyro
import torch
from pyro.infer import Predictive, Trace_ELBO
from tqdm import tqdm

from .handler import SVIBaseHandler


class SVILocalHandler(SVIBaseHandler):
    """
    Extend SVIBaseHandler to enable to use a separate model and guide
    for prediction. Assumes theat model and guide accept an idx argument
    that is an torch array of indices.
    """

    def __init__(
        self,
        model,
        guide,
        loss: Trace_ELBO = pyro.infer.TraceMeanField_ELBO,
        optimizer=torch.optim.Adam,
        scheduler=pyro.optim.ReduceLROnPlateau,
        seed=None,
        num_epochs: int = 30000,
        log_freq: int = 10,
        checkpoint_freq: int = 500,
        to_numpy: bool = True,
        optimizer_kwargs: dict = {"lr": 1e-2},
        scheduler_kwargs: dict = {"factor": 0.99},
        loss_kwargs: dict = {"num_particles": 1},
        predict_model=None,
        predict_guide=None,
        idx: torch.Tensor = None,
    ):
        super().__init__(
            model=model,
            guide=guide,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            seed=seed,
            num_epochs=num_epochs,
            log_freq=log_freq,
            checkpoint_freq=checkpoint_freq,
            to_numpy=to_numpy,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
            loss_kwargs=loss_kwargs,
        )
        self.predict_model = predict_model
        self.predict_guide = predict_guide
        self.idx = idx

    def predict(self, return_sites, num_samples=25, *args, **kwargs):
        if self.params is not None:
            pyro.clear_param_store()
            pyro.get_param_store().set_state(self.params)

        predictive = Predictive(
            self.predict_model,
            guide=self.predict_guide,
            num_samples=num_samples,
            return_sites=return_sites,
        )

        posterior = predictive(*args, **kwargs)
        self.posterior = self._to_numpy(posterior) if self.to_numpy else posterior
        torch.cuda.empty_cache()

    def predict_global_variable(self, var: str, num_samples: int = 25):
        """
        Sample global variables from the posterior.

        Parameters
        ----------
        var : str
            Name of the variable to sample.
        num_samples : int
            Number of samples to draw.
        """

        self.predict([var], num_samples=num_samples, idx=self.idx[0:1])

        return self.posterior[var]

    def predict_local_variable(
        self,
        var: str,
        num_samples: int = 25,
        num_split: int = 2048,
        obs_dim: int = 1,
    ):
        """
        Sample local variables from the posterior. In order to
        avoid memory issues, the sampling is performed in batches.

        Parameters
        ----------
        var : str
            Name of the variable to sample.
        num_samples : int
            Number of samples to draw.
        num_split : int
            The parameter determines the size of the batches. The actual
            batch size is total number of observations divided by num_split.
        obs_dim : int
            The dimension of the observations. After sampling, the output
            is concatenated along this dimension.
        """
        split_obs = torch.split(self.idx, num_split)

        # create status bar
        pbar = tqdm(range(len(split_obs)))

        results = []
        for i in pbar:
            self.predict([var], num_samples=num_samples, idx=split_obs[i])
            results.append(self.posterior[var])
            # update status bar
            pbar.set_description(
                f"Predicting {var} for obs {torch.min(split_obs[i])}-{torch.max(split_obs[i])}."
            )

        return np.concatenate(results, obs_dim)
