from os.path import isdir, join
from os import mkdir
from pytorch_lightning.callbacks import Callback
from datasets.seco_datamodule import ChangeAwareContrastMultiAugDataModule
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('Agg')


class HistogramCallback(Callback):
    def __init__(self, what="epochs", datamodule=None):
        self.what = what
        self.state = {"epochs": 0}
        self.datamodule = datamodule

    @property
    def state_key(self):
        return self._generate_state_key(what=self.what)

    def on_train_epoch_end(self, *args, **kwargs):
        if self.what == "epochs":
            self.state["epochs"]+=1
            if isinstance(self.datamodule, ChangeAwareContrastMultiAugDataModule):
                    with np.errstate(invalid='ignore', divide='ignore'):
                        val = self.datamodule.train_dataset.loss/np.expand_dims(self.datamodule.train_dataset.lossdeno, axis=1)
                    nonnaninds = np.argwhere(~np.isnan(val[:, 2]))[:, 0]
                    ratio = val[:, 2][~np.isnan(val[:, 2])]
                    gmm = GaussianMixture(n_components=2).fit(np.expand_dims(ratio, axis=1))
                    mean = gmm.means_
                    labs = gmm.predict(np.expand_dims(ratio, axis=1))
                    if mean[0]>mean[1]:
                        labs = 1-labs
                    for i in range(len(self.datamodule.train_dataset.lossdeno)):
                        self.datamodule.train_dataset.gmmlabel[i] = -1

                    for i in range(len(nonnaninds)):
                        self.datamodule.train_dataset.gmmlabel[nonnaninds[i]] = labs[i]

                    if isinstance(self.datamodule, ChangeAwareContrastMultiAugDataModule):
                        odir = 'histogram_caco'
                    if not isdir(odir):
                        mkdir(odir)

                    plt.figure(figsize=(9, 9))
                    plt.subplot(3, 1, 1)
                    plt.title("Long-term Difference")
                    plt.hist(val[nonnaninds, :][:, 0][labs==0], bins=np.arange(0, 2 + 0.02, 0.02), alpha=0.5)
                    plt.hist(val[nonnaninds, :][:, 0][labs==1], bins=np.arange(0, 2 + 0.02, 0.02), alpha=0.5)
                    plt.subplot(3, 1, 2)
                    plt.title("Short-term Difference")
                    plt.hist(val[nonnaninds, :][:, 1][labs==0], bins=np.arange(0, 2 + 0.02, 0.02), alpha=0.5)
                    plt.hist(val[nonnaninds, :][:, 1][labs==1], bins=np.arange(0, 2 + 0.02, 0.02), alpha=0.5)
                    plt.subplot(3, 1, 3)
                    plt.title("Ratio Difference")
                    plt.hist(val[nonnaninds, :][:, 2][labs==0], bins=np.arange(0, 8 + 0.05, 0.05), alpha=0.5)
                    plt.hist(val[nonnaninds, :][:, 2][labs==1], bins=np.arange(0, 8 + 0.05, 0.05), alpha=0.5)

                    plt.savefig(join(odir, 'histogram_epoch={}.png'.format(str(self.state["epochs"]).zfill(3))))
                    plt.close()
