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
from do.domain.amino_acid import alanine, histidine


if __name__ == "__main__":
    variant = PdbVariantSelection("tests/data/101M.pdb", "A", 24, None, histidine, alanine,
                                  {"A": "tests/data/101m/101m.A.pdb.pssm"}, VariantClass.PATHOGENIC)

    # Generate one atomic graph file, to inspect manually
    hdf5_path = "test_atomic.hdf5"

    if os.path.isfile(hdf5_path):
        os.remove(hdf5_path)

    variant_atomic_graph.add_as_graph(variant, hdf5_path)
