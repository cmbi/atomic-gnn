#!/usr/bin/env python

import traceback
import logging
import sys
import os
import gzip
from argparse import ArgumentParser
from multiprocessing import Process, Lock
from math import isnan
from glob import glob
import traceback
from time import time

import h5py
import numpy
import pandas
import csv
from Bio import SeqIO
from mpi4py import MPI
import h5py
from pdb2sql import pdb2sql
import gzip
from pdbecif.mmcif_io import CifFileReader


# Assure that python can find the deeprank files:
deeprank_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deeprank_root)

from deeprank_gnn.variant_atomic_graph import add_as_graph
from deeprank_gnn.models.variant import PdbVariantSelection, VariantClass
from deeprank_gnn.domain.amino_acid import amino_acids


arg_parser = ArgumentParser(description="Preprocess variants from a parquet file into HDF5")
arg_parser.add_argument("variant_path", help="the path to the (dataset) variant parquet file")
arg_parser.add_argument("map_path", help="the path to the (dataset) mapping hdf5 file")
arg_parser.add_argument("pdb_root", help="the path to the pdb root directory")
arg_parser.add_argument("pssm_root", help="the path to the pssm root directory, containing files generated with PSSMgen")
arg_parser.add_argument("out_path_prefix", help="the path prefix to the output hdf5 file (becomes: prefix-{number}.hdf5)")
arg_parser.add_argument("--process-count", type=int, help="number of processes to use for preprocessing the data", default=10)


logging.basicConfig(filename="preprocess_bioprodict-%d.log" % os.getpid(), filemode="w", level=logging.DEBUG)
_log = logging.getLogger(__name__)


class Preprocess(Process):
    "process that runs the preprocessing function"

    def __init__(self, variants, hdf5_path):
        Process.__init__(self)

        self._running_lock = Lock()

        self._variants = variants
        self._hdf5_path = hdf5_path

    def start(self):
        with self._running_lock:
            self._running = True

        Process.start(self)

    def stop(self):
        _log.info("stopping the preprocessing ..")
        with self._running_lock:
            self._running = False

    def run(self):
        _log.info("preprocessing {} variants to {} ..".format(len(self._variants), self._hdf5_path))

        for variant in self._variants:

            with self._running_lock:
                if not self._running:
                    break

            try:
                add_as_graph(variant, self._hdf5_path)
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
    return {path.split('.')[1]: path for path in paths}


def get_variant_data(parq_path, hdf5_path, pdb_root, pssm_root):
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
    variant_classes = {}
    for variant_name, variant_class in class_table['class'].items():
        variant_name = variant_name.split('.')[1]

        # Convert class to deeprank format (0: benign, 1: pathogenic):
        if variant_class == 0.0:
            variant_class = VariantClass.BENIGN

        elif variant_class == 1.0:
            variant_class = VariantClass.PATHOGENIC
        else:
            raise ValueError("Unknown class: {}".format(variant_class))

        variant_classes[variant_name] = variant_class

    # Get all mappings to pdb and use them to create variant objects:
    objects = set([])
    for variant_index, variant_row in mappings_table.iterrows():

        variant_name = variant_row["variant"]
        if variant_name not in variant_classes:
            _log.warning("no such variant: {}".format(variant_name))
            continue

        variant_class = variant_classes[variant_name]

        enst_ac = variant_name[:15]
        swap = variant_name[15:]
        wt_amino_acid_code = swap[:3]
        residue_number = int(swap[3: -3])
        var_amino_acid_code = swap[-3:]

        pdb_ac = variant_row["pdb_structure"]
        pdb_number_s = variant_row["pdbnumber"]
        if pdb_number_s[-1].isalpha():
            insertion_code = pdb_number_s[-1]
            pdb_number = int(pdb_number_s[:-1])
        else:
            insertion_code = None
            pdb_number = int(pdb_number_s)

        chain_id = pdb_ac[4]
        pdb_ac = pdb_ac[:4]

        pdb_path = os.path.join(pdb_root, "pdb%s.ent" % pdb_ac.lower())
        if not os.path.isfile(pdb_path):
            _log.warning("no such pdb: {}".format(pdb_path))
            continue

        pssm_paths = get_pssm_paths(pssm_root, pdb_ac)
        if len(pssm_paths) == 0:
            _log.warning("no pssms for: {}".format(pdb_ac))
            continue

        _log.info("add variant on {} {} {} {}->{} = {}"
                  .format(pdb_path, chain_id, pdb_number,
                          wt_amino_acid_code, var_amino_acid_code,
                          variant_class))

        o = PdbVariantSelection(pdb_path, chain_id, pdb_number, insertion_code,
                                amino_acids_by_code[wt_amino_acid_code],
                                amino_acids_by_code[var_amino_acid_code],
                                pssm_paths, variant_class)
        objects.add(o)
        if len(objects) >= 1000:
            break

    return list(objects)


def get_subset(variants):
    """ Take a subset of the input list of variants so that the ratio benign/pathogenic is 50 / 50

        Args:
            variants (list of PdbVariantSelection objects): the input variants

        Returns (list of PdbVariantSelection objects): the subset of variants, taken from the input
    """

    benign = []
    pathogenic = []
    for variant in variants:
        if variant.variant_class == VariantClass.PATHOGENIC:
            pathogenic.append(variant)
        elif variant.variant_class == VariantClass.BENIGN:
            benign.append(variant)

    _log.info("variants: got {} benign and {} pathogenic".format(len(benign), len(pathogenic)))

    count = min(len(benign), len(pathogenic))

    numpy.random.seed(int(time()))

    numpy.random.shuffle(benign)
    numpy.random.shuffle(pathogenic)

    variants = benign[:count] + pathogenic[:count]
    numpy.random.shuffle(variants)

    _log.info("variants: taking a subset of {}".format(len(variants)))

    return variants


if __name__ == "__main__":
    args = arg_parser.parse_args()

    variants = get_variant_data(args.variant_path, args.map_path, args.pdb_root, args.pssm_root)

    variants = get_subset(variants)

    variant_count = len(variants)
    subset_size = int(variant_count / args.process_count)
    processes = []
    for process_index in range(args.process_count):
        start_index = process_index * subset_size
        end_index = min(start_index + subset_size, variant_count)

        # make sure we take all variants, add the remainder to the last process:
        if (variant_count - end_index) < subset_size:
            end_index = variant_count

        hdf5_path = "%s-%d.hdf5" % (args.out_path_prefix, process_index)

        process = Preprocess(variants[start_index: end_index], hdf5_path)
        process.start()
        processes.append(process)

    _log.info("waiting for {} processes to finish ..".format(len(processes)))

    try:
        for process in processes:
            process.join()

    except:  # interrupted
        for process in processes:
            process.stop()
