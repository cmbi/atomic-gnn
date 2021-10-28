import logging

from do.domain.amino_acid import amino_acids


_log = logging.getLogger(__name__)

_amino_acid_by_name = {amino_acid.name: amino_acid for amino_acid in amino_acids}
_amino_acid_by_code = {amino_acid.code: amino_acid for amino_acid in amino_acids}


class Residue:
    def __init__(self, number, name, chain_id, insertion_code=None):
        if type(number) != int:
            raise TypeError("{} inserted as number".format(type(number)))

        if type(chain_id) != str:
            raise TypeError("{} inserted as chain_id".format(type(chain_id)))

        if len(chain_id) != 1:
            raise TypeError("chain_id with length {} inserted".format(len(chain_id)))

        self.number = number
        self.insertion_code = insertion_code
        self.name = name
        self.chain_id = chain_id
        self.atoms = []

    @property
    def amino_acid(self):
        "returns the amino acid that most closely resembles this residue's name (if any)"

        if self.name in _amino_acid_by_name:
            return _amino_acid_by_name[self.name]

        elif self.name in _amino_acid_by_code:
            return _amino_acid_by_code[self.name]

        return None

    def __hash__(self):
        return hash((self.chain_id, self.number, self.insertion_code))

    def __eq__(self, other):
        return self.chain_id == other.chain_id and self.number == other.number and self.insertion_code == other.insertion_code

    def __lt__(self, other):
        if self.chain_id == other.chain_id:

            if self.number == other.number:

                return self.insertion_code < other.insetion_code
            else:
                return self.number < other.number
        else:
            return self.chain_id < other.chain_id

    def __repr__(self):
        if self.insertion_code is None:
            return "Residue {} {} in {}".format(self.name, self.number, self.chain_id)
        else:
            return "Residue {} {}{} in {}".format(self.name, self.number, self.insertion_code, self.chain_id)
