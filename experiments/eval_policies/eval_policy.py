import click
import pytorch_lightning as pl
import wandb

from experiments.data.data_module import (
    UKRegDataModule,
    UNOS2UKRegDataModule,
    UNOSDataModule,
)
from organsync.models.inference import (
    Inference_OrganITE,
    Inference_OrganITE_VAE,
    Inference_OrganSync,
)
from organsync.models.organite import OrganITE_Network, OrganITE_Network_VAE
from organsync.models.organsync import OrganSync_Network

# OWN MODULES
from organsync.policies import OrganITE, OrganSync
from organsync.simulation import Sim

pl.utilities.seed.seed_everything(seed=42)


@click.command()
@click.option("--K", type=int, default=10)
@click.option("--wl", type=int, default=450)
@click.option("--od", type=float, default=0.4)
@click.option("--pc", type=int, default=2000)
@click.option("--data", type=str, default="UKReg")
@click.option(
    "--location", type=str, default="./data/processed_UKReg/clinical_ukeld_2_ukeld"
)
@click.option("--wandb_project", type=str, default="organsync-pol-ukreg")
def main(
    k: int, wl: int, od: float, pc: int, data: str, location: str, wandb_project: str
) -> None:
    batch_size = 256

    if data == "UNOS":
        project = "organsync-net"
        data_dir = "../data/processed"
        dm = UNOSDataModule(data_dir, batch_size=batch_size)
    elif data == "U2U":
        project = "organsync-net-u2u"
        project_vae = "organsync-organite-pnet-u2u"
        project_oite = "organsync-organite-net-u2u"
        data_dir = "../data/processed_UNOS2UKReg_no_split"
        dm = UNOS2UKRegDataModule(location, batch_size=batch_size)
    else:
        project = "organsync-net-ukreg"
        project_vae = "organsync-organite-pnet-ukreg"
        project_oite = "organsync-organite-net-ukreg"
        data_dir = "../data/processed_UKReg/clinical_ukeld_2_ukeld"
        dm = UKRegDataModule(location, batch_size=batch_size, control=False)
        dm_control = UKRegDataModule(location, batch_size=batch_size, control=True)

        dm_control.prepare_data()  # move below once all have dm_control
        dm_control.setup(stage="fit")

    dm.prepare_data()
    dm.setup(stage="fit")

    # INFERENCE LOADING
    # OrganSync
    model_id_0 = "2gsswo91"
    model_id_1 = "8298slm5"

    params_0 = wandb.restore(
        "organsync_net.ckpt-v0.ckpt",
        run_path=f"jeroenbe/{project}/{model_id_0}",
        replace=True,
    )
    model_0 = OrganSync_Network.load_from_checkpoint(
        params_0.name, map_location="cpu"
    ).double()

    params_1 = wandb.restore(
        "organsync_net.ckpt-v0.ckpt",
        run_path=f"jeroenbe/{project}/{model_id_1}",
        replace=True,
    )
    model_1 = OrganSync_Network.load_from_checkpoint(
        params_1.name, map_location="cpu"
    ).double()

    trainer_0 = pl.Trainer(gpus=0)
    trainer_1 = pl.Trainer(gpus=0)

    trainer_0.datamodule = dm_control
    trainer_1.datamodule = dm
    model_0.trainer = trainer_0
    model_1.trainer = trainer_1

    inference_0 = Inference_OrganSync(
        model=model_0, mean=dm_control.mean, std=dm_control.std
    )
    inference_1 = Inference_OrganSync(model=model_1, mean=dm.mean, std=dm.std)

    lambd = 0.1

    inference_0.model.lambd = lambd
    inference_1.model.lambd = lambd

    # OrganITE
    model_vae_id = "122jaxpq"
    params_vae = wandb.restore(
        "organite_vae_net-v0.ckpt",
        run_path=f"jeroenbe/{project_vae}/{model_vae_id}",
        replace=True,
    )
    O_VAE = OrganITE_Network_VAE.load_from_checkpoint(
        params_vae.name, map_location="cpu"
    ).double()
    inference_oite_vae = Inference_OrganITE_VAE(model=O_VAE, mean=dm.mean, std=dm.std)
    inference_oite_vae.model.trainer = trainer_1

    model_oite_id = "or6o700x"
    params_oite = wandb.restore(
        "organite_net.ckpt.ckpt",
        run_path=f"jeroenbe/{project_oite}/{model_oite_id}",
        replace=True,
    )
    organite_net = OrganITE_Network.load_from_checkpoint(
        params_oite.name, map_location="cpu"
    ).double()
    inference_oite = Inference_OrganITE(model=organite_net, mean=dm.mean, std=dm.std)
    inference_oite.model.trainer = trainer_1

    wandb.init(project=wandb_project)
    wandb.config.K = k
    wandb.config.od = od
    wandb.config.wl = wl
    wandb.config.pc = pc

    sim_organsync = Sim(
        dm=dm,
        initial_waitlist_size=wl,
        inference_0=inference_0,
        inference_1=inference_1,
        organ_deficit=od,
        patient_count=pc,
    )
    organsync = OrganSync(
        name="O-Sync",
        initial_waitlist=[p.id for p in sim_organsync.waitlist],
        dm=dm,
        K=k,
        inference_0=inference_0,
        inference_1=inference_1,
    )

    stats_os, _ = sim_organsync.simulate(organsync)

    sim_organite = Sim(
        dm=dm,
        initial_waitlist_size=wl,
        inference_0=inference_0,
        inference_1=inference_1,
        organ_deficit=od,
        patient_count=pc,
    )
    organite = OrganITE(
        name="O-ITE",
        initial_waitlist=[p.id for p in sim_organite.waitlist],
        dm=dm,
        inference_ITE=inference_oite,
        inference_VAE=inference_oite_vae,
    )

    stats_oite, _ = sim_organite.simulate(organite)

    wandb.run.summary["lifeyears"] = (
        stats_os.population_life_years - stats_oite.population_life_years
    )
    wandb.run.summary["deaths"] = stats_os.deaths - stats_oite.deaths
    wandb.run.summary["first_empty_day"] = stats_os.first_empty_day


if __name__ == "__main__":
    main()
