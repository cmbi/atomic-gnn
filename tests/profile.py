#!/usr/bin/env python

import os
import sys
import csv
import logging
from glob import glob
from tempfile import mkstemp
from argparse import ArgumentParser
import traceback

# Make sure that python can find the deeprank modules
deeprank_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(deeprank_root)

from deeprank_gnn.variant_atomic_graph import add_as_graph
from deeprank_gnn.models.variant import VariantClass, PdbVariantSelection
from deeprank_gnn.domain.amino_acid import amino_acids

logging.basicConfig(filename="profile-%d.log" % os.getpid(), filemode="w", level=logging.DEBUG)
_log = logging.getLogger(__name__)


arg_parser = ArgumentParser(description="make a time profile of the preprocessing")
arg_parser.add_argument("pdb_root", help="root directory where the pdb files are stored")
arg_parser.add_argument("pssm_root", help="root directory where the pssm files are stored")


def load_variants(pdb_root, pssm_root):
    variants = []
    set_path = "tests/data/variant-set.csv"
    with open(set_path, 'r') as f:
        r = csv.reader(f)
        for row in r:
            enst, swap, residue_number, pdb_ac_chain, class_s = row
            wt_code = swap[:3]
            var_code = swap[-3:]
            residue_number = int(residue_number)
            pdb_ac = pdb_ac_chain[:4]
            chain_id = pdb_ac_chain[4]

            amino_acids_by_code = {amino_acid.code: amino_acid for amino_acid in amino_acids}
            wt_amino_acid = amino_acids_by_code[wt_code]
            var_amino_acid = amino_acids_by_code[var_code]

            if class_s == "1.0":
                variant_class = VariantClass.PATHOGENIC
            elif class_s == "0.0":
                variant_class = VariantClass.BENIGN
            else:
                raise ValueError("unknown class: {}".format(class_s))

            pdb_path = os.path.join(pdb_root, "pdb%s.ent" % pdb_ac.lower())

            pssm_paths = {os.path.basename(path).split('.')[1]: path
                          for path in glob(os.path.join(pssm_root, pdb_ac.lower(), "pssm", "%s.?.pdb.pssm" % pdb_ac.lower()))}

            variant = PdbVariantSelection(pdb_path, chain_id, residue_number, None, wt_amino_acid, var_amino_acid, pssm_paths, variant_class)
            variants.append(variant)

    return variants


if __name__ == "__main__":
    args = arg_parser.parse_args()
    variants = load_variants(args.pdb_root, args.pssm_root)

    hdf5_file, hdf5_path = mkstemp(".hdf5", "profile-")
    os.close(hdf5_file)

    _log.info("{} variants loaded, writing graphs to {}".format(len(variants), hdf5_path))

    count_success = 0
    for variant in variants:
        try:
            add_as_graph(variant, hdf5_path)

            count_success += 1
        except:
            _log.error("error running on {}: {}".format(variant, traceback.format_exc()))

    _log.info("{} variants preprocessed successfully".format(count_success))
