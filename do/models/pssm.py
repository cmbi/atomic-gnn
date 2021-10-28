

class _PssmRecord:
    def __init__(self):
        self.conservations = {}  # uses amion_acids as keys
        self.information_content = None


class Pssm:
    "This object stores pssm data"

    def __init__(self):
        self._residue_records = {}  # the keys should be residue identifiers

    def set_conservation(self, residue_id, amino_acid, value):
        """ Set conservation to the pssm object for one specific amino acid on this specific residue position

            Args:
              residue_id (Residue, unique): identifier of the residue in the protein
              amino_acid (AminoAcid)
              value (float): specific for this amino acid on this position
        """

        if residue_id not in self._residue_records:
            self._residue_records[residue_id] = _PssmRecord()

        self._residue_records[residue_id].conservations[amino_acid] = value

    def set_information_content(self, residue_id, value):
        """ Set information content to the pssm object for one specific residue_position

            Args:
                residue_id (Residue, unique): identifier of the residue in the protein
                value (float): information content, specific for this position
        """

        if residue_id not in self._residue_records:
            self._residue_records[residue_id] = _PssmRecord()

        self._residue_records[residue_id].information_content = value

    def merge_with(self, other):
        new = Pssm()
        new._residue_records = self._residue_records
        new._residue_records.update(other._residue_records)

        return new

    def get_conservation(self, residue_id, amino_acid):
        """ Get the pssm's conservation value of the given amino acid at the given residue position

            Args:
                residue_id (Residue, unique): identifier of the residue in the protein
                amino_acid (AminoAcid)
        """

        if residue_id not in self._residue_records:
            raise ValueError("No such residue: {}".format(residue_id))

        return self._residue_records[residue_id].conservations[amino_acid]

    def get_information_content(self, residue_id):
        """ Get the pssm's information content for a specific residue position

            Args:
                residue_id (Residue, unique): identifier of the residue in the protein
        """

        if residue_id not in self._residue_records:
            raise ValueError("No such residue: {}".format(residue_id))

        return self._residue_records[residue_id].information_content

    def has_residue(self, residue_id):
        return residue_id in self._residue_records

    def items(self):
        return self._residue_records.items()

    def residues(self):
        return self._residue_records.keys()

    def __len__(self):
        return len(self._residue_records)
