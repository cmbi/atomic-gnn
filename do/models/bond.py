from do.models.pair import Pair


class BondDataObject:
    def __init__(self, atom1_name, atom2_name, bond_order, bond_length):
        self.atom_names = Pair(atom1_name, atom2_name)

        self.bond_order = bond_order

        self.bond_length = bond_length


