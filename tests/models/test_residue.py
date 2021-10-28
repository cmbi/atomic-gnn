from do.models.residue import Residue


def test_hash():
    r1 = Residue(1, "ALA", "A")
    r2 = Residue(1, "ALA", "A")

    d = {r1: 0}
    assert r2 in d, "{} inserted, but not found".format(r2)



