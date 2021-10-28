from nose.tools import eq_, ok_
import numpy

from do.domain.forcefield import atomic_forcefield
from do.models.residue import Residue
from do.models.atom import Atom


def test_atomic_forcefield():
    chain_id = "A"

    # The arginine C-zeta should get a positive charge
    arg = Residue(2, "ARG", chain_id)
    arg.atoms.append(Atom(11, (0,0,0), chain_id, "N", "N", arg))
    arg.atoms.append(Atom(12, (0,0,0), chain_id, "CA", "C", arg))
    arg.atoms.append(Atom(13, (0,0,0), chain_id, "C", "C", arg))
    arg.atoms.append(Atom(14, (0,0,0), chain_id, "O", "O", arg))
    arg.atoms.append(Atom(15, (0,0,0), chain_id, "CB", "C", arg))
    arg.atoms.append(Atom(16, (0,0,0), chain_id, "CG", "C", arg))
    arg.atoms.append(Atom(17, (0,0,0), chain_id, "CD", "C", arg))
    arg.atoms.append(Atom(18, (0,0,0), chain_id, "NE", "N", arg))
    cz = Atom(19, (0,0,0), chain_id, "CZ", "C", arg)
    arg.atoms.append(cz)
    arg.atoms.append(Atom(20, (0,0,0), chain_id, "NH1", "N", arg))
    arg.atoms.append(Atom(21, (0,0,0), chain_id, "NH2", "N", arg))

    eq_(atomic_forcefield.get_charge(cz), 0.640)

    # The glutamate O-epsilon should get a negative charge
    glu = Residue(4, "GLU", chain_id)
    glu.atoms.append(Atom(51, (0,0,0), chain_id, "N", "N", glu))
    glu.atoms.append(Atom(52, (0,0,0), chain_id, "CA", "C", glu))
    glu.atoms.append(Atom(53, (0,0,0), chain_id, "C", "C", glu))
    glu.atoms.append(Atom(54, (0,0,0), chain_id, "O", "O", glu))
    glu.atoms.append(Atom(55, (0,0,0), chain_id, "CB", "C", glu))
    glu.atoms.append(Atom(56, (0,0,0), chain_id, "CG", "C", glu))
    glu.atoms.append(Atom(57, (0,0,0), chain_id, "CD", "C", glu))
    glu.atoms.append(Atom(58, (0,0,0), chain_id, "OE1", "O", glu))
    oe2 = Atom(59, (0,0,0), chain_id, "OE2", "O", glu)
    glu.atoms.append(oe2)

    eq_(atomic_forcefield.get_charge(oe2), -0.800)

    # The forcefield should treat terminal oxygen differently
    cter = Residue(100, "ALA", chain_id)
    cter.atoms.append(Atom(1001, (0,0,0), chain_id, "N", "N", cter))
    cter.atoms.append(Atom(1002, (0,0,0), chain_id, "CA", "C", cter))
    cter.atoms.append(Atom(1003, (0,0,0), chain_id, "C", "C", cter))
    o = Atom(1004, (0,0,0), chain_id, "O", "O", cter)
    cter.atoms.append(o)
    oxt = Atom(1005, (0,0,0), chain_id, "OXT", "O", cter)
    cter.atoms.append(oxt)
    cter.atoms.append(Atom(1006, (0,0,0), chain_id, "CB", "C", cter))

    eq_(atomic_forcefield.get_charge(oxt), -0.800)
    eq_(atomic_forcefield.get_charge(o), -0.800)

    # at longer distance, the vanderwaals energy should be less negative
    vdw_close = atomic_forcefield.get_vanderwaals_energy(Atom(1, numpy.array([0.0, 0.0, 0.0]), chain_id, "C", "C", Residue(1, "ALA", chain_id)),
                                                         Atom(2, numpy.array([0.0, 5.0, 0.0]), chain_id, "O", "O", Residue(1, "ALA", chain_id)))
    vdw_far = atomic_forcefield.get_vanderwaals_energy(Atom(3, numpy.array([0.0,-5.0, 0.0]), chain_id, "C", "C", Residue(2, "ALA", chain_id)),
                                                       Atom(4, numpy.array([0.0, 5.0, 0.0]), chain_id, "O", "O", Residue(2, "ALA", chain_id)))
    ok_(vdw_far > vdw_close)


    # a negative and positive charge should attract each other
    # so coulomb energy should be negative and less negative at longer distance
    c_close = atomic_forcefield.get_coulomb_energy(Atom(1, numpy.array([0.0, 0.0, 0.0]), chain_id, "C", "C", Residue(1, "ALA", chain_id)),
                                                   Atom(2, numpy.array([0.0, 5.0, 0.0]), chain_id, "O", "O", Residue(1, "ALA", chain_id)))
    c_far = atomic_forcefield.get_coulomb_energy(Atom(3, numpy.array([0.0,-5.0, 0.0]), chain_id, "C", "C", Residue(2, "ALA", chain_id)),
                                                 Atom(4, numpy.array([0.0, 5.0, 0.0]), chain_id, "O", "O", Residue(2, "ALA", chain_id)))
    ok_(c_far < 0.0)
    ok_(c_far < 0.0)
    ok_(c_far > c_close)

    # two positive charges should repulse each other
    # so coulomb energy should be positive and less positive at longer distance
    c_close = atomic_forcefield.get_coulomb_energy(Atom(1, numpy.array([0.0, 0.0, 0.0]), chain_id, "C", "C", Residue(1, "ALA", chain_id)),
                                                   Atom(2, numpy.array([0.0, 5.0, 0.0]), chain_id, "C", "C", Residue(1, "ALA", chain_id)))
    c_far = atomic_forcefield.get_coulomb_energy(Atom(3, numpy.array([0.0,-5.0, 0.0]), chain_id, "C", "C", Residue(2, "ALA", chain_id)),
                                                 Atom(4, numpy.array([0.0, 5.0, 0.0]), chain_id, "C", "C", Residue(2, "ALA", chain_id)))
    ok_(c_far > 0.0)
    ok_(c_far > 0.0)
    ok_(c_far < c_close)
