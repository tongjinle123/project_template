from test_tube import Experiment
from src.base.base import Base
from src.base.base_config import ConfigDict
from pytorch_lightning import Trainer
from src.utils import init_module
from src import Model


class Solver(Base):
    def __init__(self, config):
        super(Solver, self).__init__(config)
        self._init_experiment()
        self._init_checkpoint()
        self._init_earlystopping()
        self._init_trainer()

    @Base.log
    def init_model(self, model):
        self.model = model

    @Base.log
    def train(self):
        self.trainer.fit(self.model)

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        config.add(
            gradient_clip=0,
            process_position=0,
            nb_gpu_nodes=1,
            gpus=None,
            log_gpu_memory=False,
            show_progress_bar=True,
            overfit_pct=0.0,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            accumulate_grad_batches=1,
            max_nb_epochs=1000,
            min_nb_epochs=1,
            train_percent_check=1.0,
            val_percent_check=1.0,
            test_percent_check=1.0,
            val_check_interval=1.0,
            log_save_interval=100,
            add_log_row_interval=10,
            distributed_backend=None,
            use_amp=False,
            print_nan_grads=False,
            print_weights_summary=True,
            weights_save_path=None,
            amp_level='O2',
            nb_sanity_val_steps=5
        )
        return config

    @Base.log
    def _init_experiment(self):
        self.experiment = Experiment(
            save_dir=None, name='default', debug=False, version=None, autosave=False, description=None
        )
        # TODO save config

    @Base.log
    def _init_checkpoint(self):
        self.checkpoint = None
        # TODO build

    @Base.log
    def _init_earlystopping(self):
        self.earlystopping = None
        # TODO build

    @Base.log
    def _init_trainer(self):
        assert self.experiment is not None
        self.trainer = Trainer(
            experiment=self.experiment,
            early_stop_callback=self.earlystopping,
            checkpoint_callback=self.checkpoint,
            gradient_clip=self.config.gradient_clip,
            process_position=self.config.process_position,
            nb_gpu_nodes=self.config.nb_gpu_nodes,
            gpus=self.config.gpus,
            log_gpu_memory=self.config.log_gpu_memory,
            show_progress_bar=self.config.show_progress_bar,
            overfit_pct=self.config.overfit_pct,
            track_grad_norm=self.config.track_grad_norm,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            fast_dev_run=self.config.fast_dev_run,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            max_nb_epochs=self.config.max_nb_epochs,
            min_nb_epochs=self.config.min_nb_epochs,
            train_percent_check=self.config.train_percent_check,
            val_percent_check=self.config.val_percent_check,
            test_percent_check=self.config.test_percent_check,
            val_check_interval=self.config.val_check_interval,
            log_save_interval=self.config.log_save_interval,
            add_log_row_interval=self.config.add_log_row_interval,
            distributed_backend=self.config.distributed_backend,
            use_amp=self.config.use_amp,
            print_nan_grads=self.config.print_nan_grads,
            print_weights_summary=self.config.print_weights_summary,
            weights_save_path=self.config.weights_save_path,
            amp_level=self.config.amp_level,
            nb_sanity_val_steps=self.config.nb_sanity_val_steps
        )
        """

        # :param experiment: Test-tube experiment
        # :param early_stop_callback: Callback for early stopping
        # :param checkpoint_callback: Callback for checkpointing
        # :param gradient_clip: int. 0 means don't clip.
        # :param process_position: shown in the tqdm bar
        # :param nb_gpu_nodes: number of GPU nodes
        # :param gpus: int. (ie: 2 gpus) OR list to specify which GPUs [0, 1] or '0,1'
        # :param log_gpu_memory: Bool. If true, adds memory logs
        # :param show_progress_bar: Bool. If true shows tqdm bar
        # :param overfit_pct: float. uses this much of all datasets
        # :param track_grad_norm: int. -1 no tracking. Otherwise tracks that norm
        # :param check_val_every_n_epoch: int. check val every n train epochs
        # :param fast_dev_run: Bool. runs full iteration over everything to find bugs
        # :param accumulate_grad_batches: int. Accumulates grads every k batches
        # :param max_nb_epochs: int.
        # :param min_nb_epochs: int.
        # :param train_percent_check: int. How much of train set to check
        # :param val_percent_check: int. How much of val set to check
        # :param test_percent_check: int. How much of test set to check
        # :param val_check_interval: int. Check val this frequently within a train epoch
        # :param log_save_interval: int. Writes logs to disk this often
        # :param add_log_row_interval: int. How often to add logging rows
        # :param distributed_backend: str. dp, or ddp.
        # :param use_amp: Bool. If true uses apex for 16bit precision
        # :param print_nan_grads: Bool. Prints nan gradients
        # :param print_weights_summary: Bool. Prints summary of weights
        # :param weights_save_path: Bool. Where to save weights if on cluster
        # :param amp_level: str. Check nvidia docs for level
        # :param nb_sanity_val_steps: int. How many val steps before a full train loop.
        """
