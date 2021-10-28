from do.models.residue import Residue
from do.models.pssm import Pssm
from do.domain.amino_acid import amino_acids


def parse_pssm(file_, chain_id):
    pssm = Pssm()
    header = file_.readline().split()

    amino_acids_by_letter = {amino_acid.letter: amino_acid for amino_acid in amino_acids}
    for line in file_:
        data = line.split()
        record = {header[i]: data[i] for i in range(len(header))}

        residue_number_s = record['pdbresi']
        if residue_number_s[-1].isalpha():

            insertion_code = residue_number_s[-1]
            residue_number = int(residue_number_s[:-1])
        else:
            insertion_code = None
            residue_number = int(residue_number_s)

        amino_acid = amino_acids_by_letter[record['pdbresn']]
        residue = Residue(residue_number, amino_acid.code, chain_id, insertion_code)
        for amino_acid in amino_acids:
            if amino_acid.letter in record:
                value = float(record[amino_acid.letter])

                pssm.set_conservation(residue, amino_acid, value)

        pssm.set_information_content(residue, float(record['IC']))

    return pssm

