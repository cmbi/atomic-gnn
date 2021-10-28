from pdb2sql import pdb2sql
from nose.tools import eq_

from do.operate.pdb import get_atoms
from do.parse.pssm import parse_pssm
from do.domain.amino_acid import threonine, arginine, leucine
from do.models.residue import Residue


def test_pssm_parse_1():

    pssm_path = "tests/data/pssm/1CRN/1crn.A.pdb.pssm"
    with open(pssm_path, 'rt') as f:
        pssm = parse_pssm(f, 'A')

    eq_(pssm.get_conservation(Residue(1, "THR", "A"), arginine), -1)


def test_pssm_parse_2():
    pssm_path = "tests/data/pssm/1ol7.A.pdb.pssm"
    with open(pssm_path, 'rt') as f:
        pssm = parse_pssm(f, 'A')

    assert pssm.has_residue(Residue(270, "LEU", "A")), "270 is missing from 1ol7"


def test_pssm_pdb():
    pdb = pdb2sql("tests/data/pdb/1CRN.pdb")
    try:
        atoms = get_atoms(pdb)
    finally:
        pdb._close()

    pssm_path = "tests/data/pssm/1CRN/1crn.A.pdb.pssm"
    with open(pssm_path, 'rt') as f:
        pssm = parse_pssm(f, 'A')

    for atom in atoms:
        assert pssm.has_residue(atom.residue), "{} is missing from pssm".format(atom.residue)
