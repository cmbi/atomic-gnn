import os
from enum import Enum

from do.models.amino_acid import AminoAcid


class VariantClass(Enum):
    BENIGN = 0
    PATHOGENIC = 1

    def __int__(self):
        return self.value


class PdbVariantSelection:
    """Refers to a variant in a pdb file.

    Args:
        pdb_path (str): on disk file path to the pdb file
        chain_id (str): chain within the pdb file, where the variation is
        residue_number (int): the identifying number of the residue within the protein chain
        wildtype_amino_acid (AminoAcid): the amino acid to be replaced at this position
        variant_amino_acid (AminoAcid): the amino acid to place at this position
        pssm_paths_by_chain (dict(str, str), optional): the paths of the pssm files per chain id, associated with the pdb file
        variant_class (VariantClass, optional): if known, the expected classification of the variant
    """

    def __init__(self, pdb_path, chain_id, residue_number, insertion_code, wildtype_amino_acid, variant_amino_acid, pssm_paths_by_chain=None, variant_class=None):

        # Because the run depends on the correct initialization of this type of object,
        # some integrity checks have been placed here:

        if not os.path.isfile(pdb_path):
            raise FileNotFoundError(pdb_path)
        self._pdb_path = pdb_path

        if type(chain_id) != str or len(chain_id) != 1:
            raise ValueError("Invalid chain identifier: {}".format(chain_id))
        self._chain_id = chain_id

        self._residue_number = residue_number
        self._insertion_code = insertion_code

        if type(wildtype_amino_acid) != AminoAcid:
            raise ValueError("Invalid wildtype amino acid: {}".format(wildtype_amino_acid))
        self._wildtype_amino_acid = wildtype_amino_acid

        if type(variant_amino_acid) != AminoAcid:
            raise ValueError("Invalid variant amino acid: {}".format(variant_amino_acid))
        self._variant_amino_acid = variant_amino_acid

        if pssm_paths_by_chain is not None:
            for chain_id, path in pssm_paths_by_chain.items():
                if type(chain_id) != str or len(chain_id) != 1:
                    raise ValueError("Invalid chain identifier: {} pointing to {}".format(chain_id, path))

                elif not os.path.isfile(path):
                    raise FileNotFoundError(path)

        self._pssm_paths_by_chain = pssm_paths_by_chain

        if variant_class is not None and type(variant_class) != VariantClass:
            raise ValueError("Invalid variant class: {}".format(variant_class))
        self._variant_class = variant_class

    @property
    def pdb_path(self):
        return self._pdb_path

    @property
    def pdb_ac(self):
        filename = os.path.basename(self._pdb_path)
        name = filename.split('.')[0]
        ac = name.replace("pdb", "")

        return ac

    def has_pssm(self):
        "are the pssm files included?"
        return self._pssm_paths_by_chain is not None

    def get_pssm_chains(self):
        "returns the chain ids for which pssm files are available"
        if self._pssm_paths_by_chain is not None:
            return self._pssm_paths_by_chain.keys()
        else:
            return set([])

    def get_pssm_path(self, chain_id):
        "returns the pssm path for the given chain id"
        if self._pssm_paths_by_chain is None:
            raise ValueError("pssm paths are not set in this variant selection")

        if chain_id in self._pssm_paths_by_chain:
            return self._pssm_paths_by_chain[chain_id]
        else:
            raise ValueError("{}: no PSSM for chain {}, candidates are {}"
                             .format(self._pdb_path, chain_id, ",".join(self._pssm_paths_by_chain.keys())))

    def is_at(self, residue):
        """ returns true if the variant matches with the residue chain_id, number and insertion code

            Args:
                residue (Residue)
            Returns (bool)
        """

        return self._residue_number == residue.number and self._chain_id == residue.chain_id and self._insertion_code == residue.insertion_code

    @property
    def residue_id(self):
        "residue identifier within the pdb file"

        s = "{}:{}".format(self._chain_id, self._residue_number)
        if self._insertion_code is not None:
            s += self._insertion_code
        return s

    @property
    def chain_id(self):
        return self._chain_id

    @property
    def residue_number(self):
        return self._residue_number

    @property
    def insertion_code(self):
        return self._insertion_code

    @property
    def wildtype_amino_acid(self):
        return self._wildtype_amino_acid

    @property
    def variant_amino_acid(self):
        return self._variant_amino_acid

    def __eq__(self, other):
        return self._pdb_path == other._pdb_path and \
               self._chain_id == other._chain_id and \
               self._residue_number == other._residue_number and \
               self._wildtype_amino_acid == other._wildtype_amino_acid and \
               self._variant_amino_acid == other._variant_amino_acid and \
               self._pssm_paths_by_chain == other._pssm_paths_by_chain and \
               self._variant_class == other._variant_class

    def __hash__(self):
        s = "pdb=%s;" % self._pdb_path + \
            "chain=%s;" % self._chain_id + \
            "residue_number_insertion=%d%s;" % (self._residue_number, self._insertion_code) + \
            "variant_amino_acid=%s;" % self._variant_amino_acid.name + \
            "wildtype_amino_acid=%s;" % self._wildtype_amino_acid.name

        if self._pssm_paths_by_chain is not None:
            for chain_id, path in self._pssm_paths_by_chain.items():
                s += "pssm_%s=%s;" % (chain_id, path)

        if self._variant_class is not None:
            s += "class=%s" % self._variant_class.name

        return hash(s)

    def __repr__(self):
        residue_id = str(self._residue_number)
        if self._insertion_code is not None:
            residue_id += self._insertion_code

        return "{}:{}:{}:{}->{}".format(self.pdb_ac, self._chain_id, residue_id, self._wildtype_amino_acid, self._variant_amino_acid)

    @property
    def variant_class(self):
        return self._variant_class
