#!/usr/bin/env python

import traceback
import logging
import sys
import os
import gzip
from time import sleep
from argparse import ArgumentParser
from queue import Empty
from multiprocessing import Process, Lock, Queue, Pipe
from math import isnan
from glob import glob
import traceback
from time import time

import h5py
import numpy
import pandas
import csv
import h5py
from pdb2sql import pdb2sql
import gzip


# Assure that python can find the package files:
package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, package_root)

from do.operate.pdb import is_xray
from do.variant_atomic_graph import add_as_graph
from do.models.variant import PdbVariantSelection, VariantClass
from do.domain.amino_acid import amino_acids


arg_parser = ArgumentParser(description="Preprocess variants from a parquet file into HDF5")
arg_parser.add_argument("variant_path", help="the path to the (dataset) variant parquet file")
arg_parser.add_argument("map_path", help="the path to the (dataset) mapping hdf5 file")
arg_parser.add_argument("pdb_root", help="the path to the pdb root directory")
arg_parser.add_argument("pssm_root", help="the path to the pssm root directory, containing files generated with PSSMgen")
arg_parser.add_argument("out_path_prefix", help="the path prefix to the output hdf5 file (becomes: prefix-{number}.hdf5)")
arg_parser.add_argument("--process-count", type=int, help="number of processes to use for preprocessing the data", default=10)


logging.basicConfig(filename="preprocess_bioprodict-%d.log" % os.getpid(), filemode="w", level=logging.INFO)
_log = logging.getLogger(__name__)


class Preprocess(Process):
    "process that runs the preprocessing function"

    def __init__(self, queues, hdf5_path):
        Process.__init__(self)

        self._queues = queues
        self._hdf5_path = hdf5_path
        self._receiver, self._sender = Pipe(duplex=False)

    def start(self):
        Process.start(self)

    def stop(self):
        self._sender.send(1)

    def _should_stop(self):
        return self._receiver.poll()

    def run(self):
        queue_index = 0
        while not self._should_stop():

            # alternate between queues:
            queue = self._queues[queue_index % len(self._queues)]

            try:
                variant = queue.get_nowait()
            except Empty:
                queue_index += 1
                continue

            _log.debug("preprocessing {} to {} ..".format(variant, self._hdf5_path))
            try:
                add_as_graph(variant, self._hdf5_path)

                queue_index += 1
            except:
                _log.error(traceback.format_exc())


def get_pssm_paths(pssm_root, pdb_ac):
    """ Finds the PSSM files associated with a PDB entry

        Args:
            pssm_root (str):  path to the directory where the PSSMgen output files are located
            pdb_ac (str): pdb accession code of the entry of interest

        Returns (dict of strings): the PSSM file paths per PDB chain identifier
    """

    paths = glob(os.path.join(pssm_root, "%s/pssm/%s.?.pdb.pssm" % (pdb_ac.lower(), pdb_ac.lower())))

    d = {}
    for path in paths:
        filename = os.path.basename(path)  # get rid of .. and .
        chain_id = filename.split('.')[1]

        d[chain_id] = path

    return d


def pdb_meets_criteria(pdb_path):
    "some criteria to filter pdb files by"

    if not os.path.isfile(pdb_path):
        _log.warning("no such pdb: {}".format(pdb_path))
        return False

    if not is_xray(pdb_path):
        _log.warning("not an xray structure: {}".format(pdb_path))
        return False

    return True


def get_variant_data(parq_path, hdf5_path, pdb_root, pssm_root, queues):
    """ Extract data from the dataset and convert to variant objects.

        Args:
            parq_path (str): path to the bioprodict parq file, containing the variants
            hdf5_path (str): path to the bioprodict hdf5 file, mapping the variants to pdb entries
            pdb_root (str): path to the directory where the pdb files are located as: pdb????.ent
            pssm_root (str): path to the directory where the PSSMgen output files are located

        Returns (list of PdbVariantSelection objects): the variants in the dataset
        Raises (ValueError): if data is inconsistent
    """

    amino_acids_by_code = {amino_acid.code: amino_acid for amino_acid in amino_acids}

    class_table = pandas.read_parquet(parq_path)
    mappings_table = pandas.read_hdf(hdf5_path, "mappings")

    # Get all variants in the parq file:
    variant_data = []
    for variant_index, variant_row in class_table.iterrows():

        if type(variant_index) == int:
            variant_name = variant_row["variant"]
        else:
            variant_name = variant_index.split('.')[1]

        variant_class = variant_row["class"]

        # Convert class to deeprank format (0: benign, 1: pathogenic):
        if variant_class == 0.0:
            variant_class = VariantClass.BENIGN

        elif variant_class == 1.0:
            variant_class = VariantClass.PATHOGENIC
        else:
            raise ValueError("Unknown class: {}".format(variant_class))

        _log.debug("add variant {} = {}".format(variant_name, variant_class))

        variant_data.append((variant_name, variant_class))

    # Get all mappings to pdb and use them to create variant objects:
    objects = set([])
    for variant_name, variant_class in variant_data:

        map_rows = mappings_table.loc[mappings_table.variant == variant_name].dropna()
        for map_row in map_rows:

            map_row = map_rows.iloc[0]

            enst_ac = variant_name[:15]
            swap = variant_name[15:]
            wt_amino_acid_code = swap[:3]
            residue_number = int(swap[3: -3])
            var_amino_acid_code = swap[-3:]

            pdb_ac = map_row["pdb_structure"]
            pdb_number_s = map_row["pdbnumber"]
            if pdb_number_s[-1].isalpha():
                insertion_code = pdb_number_s[-1]
                pdb_number = int(pdb_number_s[:-1])
            else:
                insertion_code = None
                pdb_number = int(pdb_number_s)

            chain_id = pdb_ac[4]
            pdb_ac = pdb_ac[:4]

            pdb_path = os.path.join(pdb_root, "pdb%s.ent" % pdb_ac.lower())
            if not pdb_meets_criteria(pdb_path):
                continue

            pssm_paths = get_pssm_paths(pssm_root, pdb_ac)
            if len(pssm_paths) == 0:
                _log.warning("no pssms for: {}".format(pdb_ac))
                continue

            o = PdbVariantSelection(pdb_path, chain_id, pdb_number, insertion_code,
                                    amino_acids_by_code[wt_amino_acid_code],
                                    amino_acids_by_code[var_amino_acid_code],
                                    pssm_paths, variant_class)

            _log.debug("add variant job for {}".format(o))
            queues[variant_class].put(o)


if __name__ == "__main__":
    args = arg_parser.parse_args()

    # To make sure the dataset is balanced, use two queues
    pathogenic_queue = Queue()
    benign_queue = Queue()
    queues = {VariantClass.BENIGN: benign_queue,
              VariantClass.PATHOGENIC: pathogenic_queue}

    processes = []
    for process_index in range(args.process_count):

        hdf5_path = "%s-%d.hdf5" % (args.out_path_prefix, process_index)

        process = Preprocess(list(queues.values()), hdf5_path)
        process.start()
        processes.append(process)

    try:
        get_variant_data(args.variant_path, args.map_path, args.pdb_root, args.pssm_root, queues)

        _log.info("waiting for {} processes to finish ..".format(len(processes)))

        while not all([queue.empty() for queue in queues.values()]):
            sleep(1)

    except:
        _log.fatal(traceback.format_exc())

    finally:
        for process in processes:
            process.stop()
        for process in processes:
            process.join()

