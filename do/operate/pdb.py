import logging

import numpy

from do.models.pair import Pair
from do.models.atom import Atom
from do.models.residue import Residue
from do.domain.bonds import get_bond_data

_log = logging.getLogger(__name__)


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


def get_covalent_bonds(atoms, distance_cutoff=3.0):
    """ Lists the bonds between atoms using a knowledge-based neighbour approach

        Args:
            atoms (list(Atom)): the atoms of the pdb structure that we're investigating
            distance_cutoff (float): maximum distance between two atoms in Angstrom

        Returns ([Pair(int, int)]): pairs of atom numbers that bond each other
    """

    bonds = set([])

    squared_cutoff = distance_cutoff * distance_cutoff

    residues = sorted({atom.residue for atom in atoms},
                      key=lambda residue: (residue.chain_id, residue.number, residue.insertion_code))

    amino_acid_codes = {residue.name for residue in residues}
    allowed_bond_data = {amino_acid_code: get_bond_data(amino_acid_code) for amino_acid_code in amino_acid_codes}


    for residue_index, residue in enumerate(residues):
        allowed_bond_name_pairs = {obj.atom_names for obj in allowed_bond_data[residue.name]}

        for atom1_index, atom1 in enumerate(residue.atoms):
            for atom2 in residue.atoms[atom1_index + 1:]:
                name_pair = Pair(atom1.name, atom2.name)
                squared_distance = numpy.sum(numpy.square(atom1.position - atom2.position))
                if squared_distance < squared_cutoff and name_pair in allowed_bond_name_pairs:
                    bonds.add(Pair(atom1, atom2))

            if atom1.name == "N" and residue_index > 0:  # petide bonds
                previous_residue = residues[residue_index - 1]
                for atom2 in [atom for atom in previous_residue.atoms if atom.name == "C"]:
                    name_pair = Pair(atom1.name, atom2.name)
                    squared_distance = numpy.sum(numpy.square(atom1.position - atom2.position))
                    if squared_distance < squared_cutoff:
                        bonds.add(Pair(atom1, atom2))

            elif atom1.name == "SG":  # disulfid bonds
                for atom2 in [atom for atom in atoms if atom.name == "SG"]:
                    if atom1 == atom2:
                        continue  # don't bond a cysteine with itself

                    name_pair = Pair(atom1.name, atom2.name)
                    squared_distance = numpy.sum(numpy.square(atom1.position - atom2.position))
                    if squared_distance < squared_cutoff:
                        bonds.add(Pair(atom1, atom2))

    return bonds


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
