#!/usr/bin/env python

import sys
import h5py
import logging
import csv


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_log = logging.getLogger(__name__)


def get_epoch_metrics(hdf5_path, csv_path):
    """
        Deeprank-GNN outputs hdf5 files, containing the predicted class values.
        This function counts the number of correctly predicted classes and
        outputs the accuracy values to the hdf5 file
    """

    with h5py.File(hdf5_path, 'r') as fi:

        with open(csv_path, 'w') as fo:

            w = csv.writer(fo)

            w.writerow(["epoch", "train accuracy", "validation accuracy", "test accuracy"])

            for name in fi:
                if name.startswith("epoch_"):
                    epoch_number = int(name.replace("epoch_", ""))

                    accs = {'train': "", 'eval': "", 'test': ""}
                    for phase_name in accs:

                        if phase_name not in fi[name]:
                            continue

                        outputs = fi["{}/{}/outputs".format(name, phase_name)][()]
                        targets = fi["{}/{}/targets".format(name, phase_name)][()]

                        fp_count = 0
                        fn_count = 0
                        tp_count = 0
                        tn_count = 0
                        for index, prediction in enumerate(outputs):
                            target = targets[index]

                            _log.debug("at epoch {}:compare prediction {} to target {}".format(epoch_number, prediction, target))

                            if prediction == 1:
                                if prediction != target:
                                    fn_count += 1
                                else:
                                    tn_count += 1

                            elif prediction == 0:
                                if prediction != target:
                                    fp_count += 1
                                else:
                                    tp_count += 1
                            else:
                                raise ValueError("prediction = {}".format(prediction))

                        accs[phase_name] = (tp_count + tn_count) / (fp_count + fn_count + tp_count + tn_count)

                    w.writerow([epoch_number, accs['train'], accs['eval'], accs['test']])

if __name__ == "__main__":
    get_epoch_metrics(sys.argv[1], sys.argv[2])
