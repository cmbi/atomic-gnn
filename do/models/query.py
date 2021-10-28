

class ResidueQuery:
    pass


class AroundResidueQuery(ResidueQuery):
    "query residues around a residue"

    def __init__(self, chain_id, residue_number, distance_cutoff):
        """
            Args:
                chain_id(str): pdb chain identifier
                residue_number(int): pdb residue number
                distance_cutoff(float): max distance between two atoms in Å
        """

        self.chain_id
        self.residue_number
        self.distance_cutoff = distance_cutoff


class ProteinInterfaceResidueQuery(ResidueQuery):
    "query residues around a protein-protein interface"

    def __init__(self, chain_id1, chain_id2, distance_cutoff):
        """
            Args:
                chain_id1(str): identifier of the first protein in the pdb
                chain_id2(str): identifier of the second protein in the pdb
                distance_cutoff(float): max distance between two atoms of the two proteins in Å
        """

        self.chain_id1 = chain_id1
        self.chain_id2 = chain_id2
        self.distance_cutoff = distance_cutoff
