from nose.tools import ok_
from pdb2sql import pdb2sql

from do.operate.pdb import get_atoms, get_residue_contact_atom_pairs


def test_atoms():
    pdb = pdb2sql("tests/data/2ogv.pdb")
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    ok_(len(atoms) > 0)


def test_atoms_around_residue():
    pdb = pdb2sql("tests/data/2ogv.pdb")
    try:
        atom_pairs = get_residue_contact_atom_pairs(pdb, "A", 845, 10.0)
    finally:
        pdb._close()

    ok_(len(atom_pairs) > 0)
