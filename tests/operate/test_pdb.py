from nose.tools import ok_
from pdb2sql import pdb2sql

from do.operate.pdb import get_covalent_bonds, get_atoms, get_residue_contact_atom_pairs


def test_bonds():

    pdb = pdb2sql("tests/data/pdb/1CRN.pdb")
    try:
        atoms = get_atoms(pdb)
        bonds = get_covalent_bonds(atoms)
    finally:
        pdb._close()

    ok_(any([{atom1.name, atom2.name} == {"CA", "CB"} for atom1, atom2 in bonds]))

    ok_(any([{atom1.name, atom2.name} == {"N", "C"} for atom1, atom2 in bonds]))

    ok_(any([{atom1.name, atom2.name} == {"SG", "SG"} for atom1, atom2 in bonds]))

    ok_(not any([{atom1.name, atom2.name} == {"N", "CE"} for atom1, atom2 in bonds]))

    ok_(not any([{atom1.name, atom2.name} == {"CB", "ND2"} for atom1, atom2 in bonds]))

    # Don't bond atoms with themselves !
    ok_(not any([atom1 == atom2 for atom1, atom2 in bonds]))


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
