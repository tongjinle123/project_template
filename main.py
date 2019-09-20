import fire
from src.base import ConfigDict
from .src.utils import init_module
from src import Solver
from src import Model
from src import Parser
from src import Corpus


def train(**kwargs):
    config = ConfigDict()
    config.add(
        solver_name='Solver',
        model_name='Model',
        corpus_name='Corpus',
        parser_name='Parser',
        from_ckpt=None,
    )
    config.update(kwargs)
    # default train configs
    solver = build_pipe(config)
    solver.train()
    print('/n...done.../n')


def build_pipe(config):
    """
    init sub modules and construct solver
    :param config:
    :return: solver
    """
    solver = init_module(Solver, config.solver_name, change_args=config)
    model = init_module(Model, config.model_name, change_args=config)
    parser = init_module(Parser, config.parser_name, change_args=config)
    corpus = init_module(Corpus, config.corpus_name, change_args=config)

    #combine
    corpus.init_parser(parser)
    model.init_corpus(corpus)
    solver.init_model(model)
    return solver


def get_model_for_inference(config):
    model = init_module(Model, config.model_name, change_args=config)
    parser = init_module(Parser, config.parser_name, change_args=config)
    #TODO finish this func
    return model, parser


if __name__ == '__main__':
    fire.Fire()














#
#
# from test_tube import HyperOptArgumentParser
# import os
# from test_tube import Experiment
# import os
# from collections import OrderedDict
# import torch.nn as nn
# from torchvision.datasets import MNIST
# import torchvision.transforms as transforms
# import torch
# import torch.nn.functional as F
# from test_tube import HyperOptArgumentParser
# from torch import optim
# from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
# import pytorch_lightning as pl
# from pytorch_lightning.root_module.root_module import LightningModule
# import sys
#
#
# class LightningTemplateModel(LightningModule):
#     """
#     Sample Model to show how to define a template
#     """
#
#     def __init__(self, hparams):
#         """
#         Pass in parsed HyperOptArgumentParser to the Model
#         :param hparams:
#         """
#         # init superclass
#         super(LightningTemplateModel, self).__init__()
#         self.hparams = hparams
#
#         self.batch_size = hparams.batch_size
#
#         # if you specify an example input, the summary will show input/output for each layer
#         self.example_input_array = torch.rand(5, 28 * 28)
#
#         # build Model
#         self.__build_model()
#
#     # ---------------------
#     # MODEL SETUP
#     # ---------------------
#     def __build_model(self):
#         """
#         Layout Model
#         :return:
#         """
#         self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
#                               out_features=self.hparams.hidden_dim)
#         self.c_d1_bn = nn.BatchNorm1d(self.hparams.hidden_dim)
#         self.c_d1_drop = nn.Dropout(self.hparams.drop_prob)
#
#         self.c_d2 = nn.Linear(in_features=self.hparams.hidden_dim,
#                               out_features=self.hparams.out_features)
#
#     # ---------------------
#     # TRAINING
#     # ---------------------
#     def forward(self, x):
#         """
#         No special modification required for lightning, define as you normally would
#         :param x:
#         :return:
#         """
#
#         x = self.c_d1(x)
#         x = torch.tanh(x)
#         x = self.c_d1_bn(x)
#         x = self.c_d1_drop(x)
#
#         x = self.c_d2(x)
#         logits = F.log_softmax(x, dim=1)
#
#         return logits
#
#     def loss(self, labels, logits):
#         nll = F.nll_loss(logits, labels)
#         return nll
#
#     def training_step(self, data_batch, batch_i):
#         """
#         Lightning calls this inside the training loop
#         :param data_batch:
#         :return:
#         """
#         # forward pass
#         x, y = data_batch
#         x = x.view(x.size(0), -1)
#
#         y_hat = self.forward(x)
#
#         # calculate loss
#         loss_val = self.loss(y, y_hat)
#
#         # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
#         if self.trainer.use_dp:
#             loss_val = loss_val.unsqueeze(0)
#
#         output = OrderedDict({
#             'loss': loss_val
#         })
#
#         # can also return just a scalar instead of a dict (return loss_val)
#         return output
#
#     def validation_step(self, data_batch, batch_i):
#         """
#         Lightning calls this inside the validation loop
#         :param data_batch:
#         :return:
#         """
#         x, y = data_batch
#         x = x.view(x.size(0), -1)
#         y_hat = self.forward(x)
#
#         loss_val = self.loss(y, y_hat)
#
#         # acc
#         labels_hat = torch.argmax(y_hat, dim=1)
#         val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
#         val_acc = torch.tensor(val_acc)
#
#         if self.on_gpu:
#             val_acc = val_acc.cuda(loss_val.device.index)
#
#         # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
#         if self.trainer.use_dp:
#             loss_val = loss_val.unsqueeze(0)
#             val_acc = val_acc.unsqueeze(0)
#
#         output = OrderedDict({
#             'val_loss': loss_val,
#             'val_acc': val_acc,
#         })
#
#         # can also return just a scalar instead of a dict (return loss_val)
#         return output
#
#     def validation_end(self, outputs):
#         """
#         Called at the end of validation to aggregate outputs
#         :param outputs: list of individual outputs of each validation step
#         :return:
#         """
#         # if returned a scalar from validation_step, outputs is a list of tensor scalars
#         # we return just the average in this case (if we want)
#         # return torch.stack(outputs).mean()
#
#         val_loss_mean = 0
#         val_acc_mean = 0
#         for output in outputs:
#             val_loss = output['val_loss']
#
#             # reduce manually when using dp
#             if self.trainer.use_dp:
#                 val_loss = torch.mean(val_loss)
#             val_loss_mean += val_loss
#
#             # reduce manually when using dp
#             val_acc = output['val_acc']
#             if self.trainer.use_dp:
#                 val_acc = torch.mean(val_acc)
#
#             val_acc_mean += val_acc
#
#         val_loss_mean /= len(outputs)
#         val_acc_mean /= len(outputs)
#         tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
#         return tqdm_dic
#
#     # ---------------------
#     # TRAINING SETUP
#     # ---------------------
#     def configure_optimizers(self):
#         """
#         return whatever optimizers we want here
#         :return: list of optimizers
#         """
#         optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#         return [optimizer], [scheduler]
#
#     def __dataloader(self, train):
#         # init data generators
#         transform = transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize((0.5,), (1.0,))])
#         dataset = MNIST(root=self.hparams.data_root, train=train,
#                         transform=transform, download=True)
#
#         # when using multi-node (ddp) we need to add the datasampler
#         train_sampler = None
#         batch_size = self.hparams.batch_size
#
#         if self.use_ddp:
#             train_sampler = DistributedSampler(dataset, rank=self.trainer.proc_rank)
#             batch_size = batch_size // self.trainer.world_size  # scale batch size
#
#         should_shuffle = train_sampler is None
#         loader = DataLoader(
#             dataset=dataset,
#             batch_size=batch_size,
#             shuffle=should_shuffle,
#             sampler=train_sampler
#         )
#
#         return loader
#
#     @pl.data_loader
#     def tng_dataloader(self):
#         print('tng data loader called')
#         return self.__dataloader(train=True)
#
#     @pl.data_loader
#     def val_dataloader(self):
#         print('val data loader called')
#         return self.__dataloader(train=False)
#
#     @pl.data_loader
#     def test_dataloader(self):
#         print('test data loader called')
#         return self.__dataloader(train=False)
#
#     @staticmethod
#     def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
#         """
#         Parameters you define here will be available to your Model through self.hparams
#         :param parent_parser:
#         :param root_dir:
#         :return:
#         """
#         parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])
#
#         # param overwrites
#         # parser.set_defaults(gradient_clip=5.0)
#
#         # network params
#         parser.add_argument('--in_features', default=28 * 28, type=int)
#         parser.add_argument('--out_features', default=10, type=int)
#         # use 500 for CPU, 50000 for GPU to see speed difference
#         parser.add_argument('--hidden_dim', default=50000, type=int)
#         parser.opt_list('--drop_prob', default=0.2, options=[0.2, 0.5], type=float, tunable=True)
#         parser.opt_list('--learning_rate', default=0.001 * 8, type=float,
#                         options=[0.0001, 0.0005, 0.001],
#                         tunable=True)
#
#         # data
#         parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)
#
#         # training params (opt)
#         parser.opt_list('--optimizer_name', default='adam', type=str,
#                         options=['adam'], tunable=False)
#
#         # if using 2 nodes with 4 gpus each the batch size here
#         #  (256) will be 256 / (2*8) = 16 per gpu
#         parser.opt_list('--batch_size', default=256 * 8, type=int,
#                         options=[32, 64, 128, 256], tunable=False,
#                         help='batch size will be divided over all gpus being used across all nodes')
#         return parser
#
#
# def main(hparams, cluster, results_dict):
#     """
#     Main training routine specific for this project
#     :param hparams:
#     :return:
#     """
#     # init experiment
#     log_dir = os.path.dirname(os.path.realpath(__file__))
#     exp = Experiment(
#         name='test_tube_exp',
#         debug=True,
#         save_dir=log_dir,
#         version=0,
#         autosave=False,
#         description='test demo'
#     )
#
#     # set the hparams for the experiment
#     exp.argparse(hparams)
#     exp.save()
#
#     # build Model
#     model = MyLightningModule(hparams)
#
#     # callbacks
#     early_stop = EarlyStopping(
#         monitor=hparams.early_stop_metric,
#         patience=hparams.early_stop_patience,
#         verbose=True,
#         mode=hparams.early_stop_mode
#     )
#
#     model_save_path = '{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
#     checkpoint = ModelCheckpoint(
#         filepath=model_save_path,
#         save_function=None,
#         save_best_only=True,
#         verbose=True,
#         monitor=hparams.model_save_monitor_value,
#         mode=hparams.model_save_monitor_mode
#     )
#
#     # configure trainer
#     trainer = Trainer(
#         experiment=exp,
#         cluster=cluster,
#         checkpoint_callback=checkpoint,
#         early_stop_callback=early_stop,
#     )
#
#     # train Model
#     trainer.fit(model)
#
# if __name__ == '__main__':
#
#     # use default args given by lightning
#     root_dir = os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0]
#     parent_parser = HyperOptArgumentParser(strategy='random_search', add_help=False)
#     add_default_args(parent_parser, root_dir)
#
#     # allow Model to overwrite or extend args
#     parser = ExampleModel.add_model_specific_args(parent_parser)
#     hyperparams = parser.parse_args()
#
#     # train Model
#     main(hyperparams)
