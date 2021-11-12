import logging

import numpy

from do.models.pair import Pair
from do.models.atom import Atom
from do.models.residue import Residue

_log = logging.getLogger(__name__)


def is_xray(pdb_file):
    "check that an open pdb file is an x-ray structure"

    for line in pdb_file:
        if line.startswith("EXPDTA") and "X-RAY DIFFRACTION" in line:
            return True

    return False


def get_distance(position1, position2):
    """ Get euclidean distance between two positions in space.

        Args:
            position1 (numpy vector): first position
            position2 (numpy vector): second position

        Returns (float): the distance
    """

    return numpy.sqrt(get_squared_distance(position1, position2))


def get_squared_distance(position1, position2):
    """ Get the squared euclidean distance between two positions in space.
        (which is faster to compute than the euclidean distance)

        Args:
            position1 (numpy vector): first position
            position2 (numpy vector): second position

        Returns (float): the squared distance
    """

    return numpy.sum(numpy.square(position1 - position2))


def get_atoms(pdb2sql):
    """ Builds a list of atom objects, according to the contents of the pdb file.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating

        Returns ([Atom]): all the atoms in the pdb file.
    """

    # This is a working dictionary of residues, identified by their chains and numbers.
    residues = {}

    # This is the list of atom objects, that will be returned.
    atoms = []

    # Iterate over the atom output from pdb2sql
    for row in pdb2sql.get("x,y,z,rowID,name,element,chainID,resSeq,resName,iCode",model=0):

        x, y, z, atom_number, atom_name, element, chain_id, residue_number, residue_name, insertion_code = row

        if insertion_code == "":
            insertion_code = None

        # Make sure that the residue is in the working directory:
        residue_id = (chain_id, residue_number)
        if residue_id not in residues:
            residues[residue_id] = Residue(residue_number, residue_name, chain_id, insertion_code=insertion_code)

        # Turn the x,y,z into a vector:
        atom_position = numpy.array([x, y, z])

        # Create the atom object and link it to the residue:
        atom = Atom(atom_number, atom_position, chain_id, atom_name, element, residues[residue_id])
        residues[residue_id].atoms.append(atom)
        atoms.append(atom)

    return atoms


def get_residue_contact_atom_pairs(pdb2sql, chain_id, residue_number, max_interatomic_distance):
    """ Find interatomic contacts around a residue.

        Args:
            pdb2sql (pdb2sql object): the pdb structure that we're investigating
            chain_id (str): the chain identifier, where the residue is located
            residue_number (int): the residue number of interest within the chain
            max_interatomic_distance (float): maximum distance between two atoms in Angstrom

        Returns ([Pair(int, int)]): pairs of atom numbers that contact each other
    """

    # to compare squared distances to:
    squared_max_interatomic_distance = numpy.square(max_interatomic_distance)

    # get all the atoms in the pdb file:
    atoms = get_atoms(pdb2sql)

    # List all the atoms in the selected residue, take the coordinates while we're at it:
    residue_atoms = [atom for atom in atoms if atom.chain_id == chain_id and
                                               atom.residue.number == residue_number]
    if len(residue_atoms) == 0:
        raise ValueError("{}: no atoms found in pdb chain {} with residue number {}"
                         .format(pdb2sql.pdbfile, chain_id, residue_number))

    # Iterate over all the atoms in the pdb, to find neighbours.
    contact_atom_pairs = set([])
    for atom in atoms:

        # Check that the atom is not one of the residue's own atoms:
        if atom.chain_id == chain_id and atom.residue.number == residue_number:
            continue

        # Within the atom iteration, iterate over the atoms in the residue:
        for residue_atom in residue_atoms:

            # Check that the atom is close enough:
            if get_squared_distance(atom.position, residue_atom.position) < squared_max_interatomic_distance:

                contact_atom_pairs.add(Pair(residue_atom, atom))


    return contact_atom_pairs
