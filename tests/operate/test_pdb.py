from nose.tools import ok_
from pdb2sql import pdb2sql

from do.operate.pdb import get_atoms, get_residue_contact_atom_pairs, is_xray


def test_xray():
    for path in ["tests/data/101M.pdb", "tests/data/1CRN.pdb"]:
        with open(path, 'rt') as f:
            assert is_xray(f), "{} is not identified as x-ray".format(path)

    with open("tests/data/1A6B.pdb", 'rt') as f:
        assert not is_xray(f), "1A6B was identified as x-ray"


def test_atoms():
    pdb = pdb2sql("tests/data/1CRN.pdb")
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    ok_(len(atoms) > 0)


def test_atoms_around_residue():
    pdb = pdb2sql("tests/data/1CRN.pdb")
    try:
        atom_pairs = get_residue_contact_atom_pairs(pdb, "A", 22, 10.0)
    finally:
        pdb._close()

    ok_(len(atom_pairs) > 0)
