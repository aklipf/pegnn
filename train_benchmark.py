import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader

import json
from src.datasets import CSVDataset
from src.utils.scaler import LatticeScaler
from src.utils.visualize import get_fig
from src.utils.debug import check_grad
from src.utils.io import AggregateBatch


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import argparse

    import os
    import random
    import datetime

    parser = argparse.ArgumentParser(description="train denoising model")
    parser.add_argument("--hparams", "-H", default=None, help="json file")
    parser.add_argument("--tensorboard", "-t", default="./runs_benchmark")
    parser.add_argument("--dataset", "-D", default="./data/carbon_24")
    parser.add_argument("--device", "-d", default="cuda")
    parser.add_argument("--verbose", "-v", default=False, action="store_true")
    parser.add_argument("--log-interval", "-l", default=128, type=int)
    parser.add_argument("--debug", "-g", default=False, action="store_true")

    args = parser.parse_args()

    from src.models.operator.denoise import Denoise
    from src.models.operator.loss import (
        get_loss,
        LossLatticeParameters,
        LossActionMatrixDistance,
        LossLatticeMetric
    )
    from src.models.operator.utils import (
        LogSpike,
        AggregateMetrics,
        training_iterator,
        validation_iterator,
        testing_iterator,
        Checkpoints,
        Hparams,
    )

    # run name
    dataset_name = os.path.split(args.dataset)[1]
    tday = datetime.datetime.now()
    run_name = tday.strftime(
        f"training_%Y_%m_%d_%H_%M_%S_{dataset_name}_{random.randint(0,1000):<03d}"
    )
    print("run name:", run_name)

    # basic setup
    device = args.device
    log_interval = args.log_interval

    output_directory = args.tensorboard

    # setup hyperparameters
    hparams = Hparams()

    if args.hparams is not None:
        hparams.from_json(args.hparams)

    print("hyper-parameters:")
    print(json.dumps(hparams.dict(), indent=4))

    # setup logs
    log_dir = os.path.join(output_directory, run_name)

    os.makedirs(output_directory, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    hparams.to_json(os.path.join(log_dir, "hparams.json"))

    log_spike = LogSpike(log_dir, threshold=0.5,
                         verbose=args.verbose, debug=args.debug)
    log_metrics_train = AggregateMetrics(writer, "train")
    log_metrics_valid = AggregateMetrics(writer, "valid")
    log_metrics_test = AggregateMetrics(writer, "test")

    # load data and data scaler
    dataset_train = CSVDataset(os.path.join(
        args.dataset, "train.csv"), verbose=args.verbose, knn=hparams.knn, multithread=False)
    dataset_val = CSVDataset(os.path.join(
        args.dataset, "val.csv"), verbose=args.verbose, knn=hparams.knn, multithread=False)
    dataset_test = CSVDataset(os.path.join(
        args.dataset, "test.csv"), verbose=args.verbose, knn=hparams.knn, multithread=False)

    dataloader_train = DataLoader(dataset_train, batch_size=hparams.batch_size)
    dataloader_val = DataLoader(dataset_val, batch_size=hparams.batch_size)
    dataloader_test = DataLoader(dataset_test, batch_size=hparams.batch_size)

    lattice_scaler = LatticeScaler()
    lattice_scaler.fit(dataloader_train, args.verbose)
    lattice_scaler = lattice_scaler.to(args.device)

    # setup model, loss and optimizer
    model = Denoise(
        features=hparams.features,
        knn=hparams.knn,
        ops_config=hparams.ops_config,
        mpnn=hparams.mpnn_layers,
        steps=hparams.steps,
        scale_limit_weights=hparams.scale_limit_weights,
        scale_hidden_dim=hparams.scale_hidden_dim,
        scale_layers=hparams.scale_layers,
        scale_limit_actions=hparams.scale_limit_actions,
        scale_reduce_rho=hparams.scale_reduce_rho,
        repeated=hparams.repeated,
        mlp_lattice=hparams.mlp_lattice,
        lattice_scaler=lattice_scaler
    ).to(device)

    if hparams.loss == "metric_l1":
        loss_fn = LossLatticeMetric(distance="l1")
    elif hparams.loss == "metric_mse":
        loss_fn = LossLatticeMetric(distance="mse")
    elif hparams.loss == "metric_trace":
        loss_fn = LossLatticeMetric(distance="trace")
    elif hparams.loss == "actions_l1":
        loss_fn = LossActionMatrixDistance(distance="l1")
    elif hparams.loss == "actions_mse":
        loss_fn = LossActionMatrixDistance(distance="mse")
    elif hparams.loss == "actions_trace":
        loss_fn = LossActionMatrixDistance(distance="trace")
    elif hparams.loss == "parameters_l1":
        loss_fn = LossLatticeParameters(
            lattice_scaler=lattice_scaler, distance="l1")
    elif hparams.loss == "parameters_mse":
        loss_fn = LossLatticeParameters(
            lattice_scaler=lattice_scaler, distance="mse")
    else:
        raise Exception(f"unknown loss {hparams.loss}")

    opti = optim.Adam(model.parameters(), lr=hparams.lr,
                      betas=(hparams.beta1, 0.999))

    # setup checkpoint and training loop
    checkpoints = Checkpoints(log_dir, model, opti)

    data_it, tqdm_bar = training_iterator(
        dataloader_train, hparams.total_step, verbose=args.verbose
    )
    for opt_step, batch in data_it:
        model.train()
        batch = batch.to(args.device)

        # training step
        opti.zero_grad()

        loss, metrics = get_loss(batch, model, loss_fn)
        loss.backward()

        check_grad(model, verbose=args.verbose, debug=args.debug)

        if hparams.grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), hparams.grad_clipping)

        opti.step()

        # logs
        log_spike.log(loss, opt_step, batch, model, opti)

        log_metrics_train.append(loss, metrics)

        if args.verbose:
            tqdm_bar.set_description(log_metrics_train.preview())

        # validation
        if (opt_step % log_interval) == 0:
            log_metrics_train.log(opt_step)

            model.eval()
            with torch.no_grad():
                fig = None
                for batch in validation_iterator(dataloader_val, verbose=args.verbose):
                    batch = batch.to(device)

                    if fig is None:
                        fig = get_fig(batch, model, 8)

                    loss, metrics = get_loss(batch, model, loss_fn)

                    log_metrics_valid.append(loss, metrics)

            metrics = log_metrics_valid.log(opt_step)
            writer.add_figure("denoising", fig, opt_step)

            checkpoints.step(opt_step, metrics)

    # testing from the best checkpoint
    model = checkpoints.load_best()
    model = model.to(device)
    model.eval()

    aggregate = AggregateBatch()
    with torch.no_grad():
        for batch in testing_iterator(dataloader_test, verbose=args.verbose):
            batch = batch.to(device)

            loss, metrics, full_batch = get_loss(
                batch, model, loss_fn, return_batch=True)

            aggregate.append(*full_batch)

            log_metrics_test.append(loss, metrics)

    metrics = log_metrics_test.log(opt_step, hparams=hparams.dict())

    with open(os.path.join(log_dir, "metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=4)

    aggregate.write(os.path.join(log_dir, "output/test"), verbose=args.verbose)

    print("\ntest metrics:")
    print(json.dumps(metrics, indent=4))
