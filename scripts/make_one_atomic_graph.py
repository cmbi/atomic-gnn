#!/usr/bin/env python

import sys
import os
import logging

import h5py

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
_log = logging.getLogger(__name__)

deeprank_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, deeprank_root)

from do import variant_atomic_graph
from do.models.variant import PdbVariantSelection, VariantClass
from do.domain.amino_acid import alanine, glycine


if __name__ == "__main__":
    variant = PdbVariantSelection("tests/data/pdb/1ATN/1ATN_1w.pdb", "A", 404, None, glycine, alanine,
                                  {"A": "tests/data/pssm/1ATN/1ATN.A.pdb.pssm",
                                   "B": "tests/data/pssm/1ATN/1ATN.B.pdb.pssm"}, VariantClass.PATHOGENIC)

    # Generate one atomic graph file, to inspect manually
    hdf5_path = "test_atomic.hdf5"

    if os.path.isfile(hdf5_path):
        os.remove(hdf5_path)

    variant_atomic_graph.add_as_graph(variant, hdf5_path)
