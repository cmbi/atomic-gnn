import logging
from time import time

import h5py
import numpy
from pdb2sql import pdb2sql
import networkx
import freesasa
from torch.cuda import FloatTensor
from torch_scatter import scatter_max, scatter_mean
from deeprank_gnn.community_pooling import community_detection, community_pooling

from do.models.pair import Pair
from do.domain.graph import POSITION_FEATURE_NAME, EDGETYPE_FEATURE_NAME, EDGETYPE_ENCODING
from do.operate.hdf5data import get_variant_group_name, store_variant
from do.operate.pdb import get_squared_distance, get_distance, get_residue_contact_atom_pairs
from do.operate.graph import graph_to_data, data_to_deeprank_hdf5
from do.domain.forcefield import atomic_forcefield
from do.profile import time_profile
from do.parse.pssm import parse_pssm
from do.domain.amino_acid import amino_acids


_log = logging.getLogger(__name__)

def _make_label_one_hots(label_set):

    representations = {}
    for index, label in enumerate(label_set):
        one_hot_value = numpy.zeros(len(label_set), dtype=float)
        one_hot_value[index] = 1.0

        representations[label] = one_hot_value

    return representations


ELEMENTS = {"C", "N", "O", "S", "H", "P"}
ELEMENT_NN_REPRESENTATIONS = _make_label_one_hots(ELEMENTS)

AMINO_ACID_NN_REPRESENTATIONS = _make_label_one_hots(amino_acids)

EDGENAME_BONDED = "bonded"
EDGENAME_NONBONDED = "nonbonded"


@time_profile
def _add_pssm(graph, variant):
    pssms = {}
    for chain_id in variant.get_pssm_chains():
        pssm_path = variant.get_pssm_path(chain_id)
        with open(pssm_path, 'rt') as f:
            pssms[chain_id] = parse_pssm(f, chain_id)

    for atom in graph.nodes:
        chain_id = atom.chain_id
        if chain_id not in pssms:
            raise ValueError("no PSSM for chain {} of {}".format(chain_id, variant.pdb_ac))

        pssm = pssms[chain_id]
        residue = atom.residue
        if variant.is_at(residue):

            wildtype_amino_acid = variant.wildtype_amino_acid
            variant_amino_acid = variant.variant_amino_acid
        else:
            amino_acid = residue.amino_acid
            if amino_acid is None:
                continue  # not an amino acid residue

            wildtype_amino_acid = amino_acid
            variant_amino_acid = amino_acid

        if pssm.has_residue(residue):
            graph.nodes[atom]['wildtype_conservation'] = pssm.get_conservation(residue, wildtype_amino_acid)
            graph.nodes[atom]['variant_conservation'] = pssm.get_conservation(residue, variant_amino_acid)
            graph.nodes[atom]['information_content'] = pssm.get_information_content(residue)
        else:
            raise ValueError("{} is missing from {}".format(residue, variant.get_pssm_path(chain_id)))


@time_profile
def _add_sasa(graph, pdb_path):
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)

    for atom in graph.nodes:
        if atom.element == "H":  # SASA doesn't have these
            area = 0.0
        else:
            select_str = ('atom, (name %s) and (resi %d) and (chain %s)' % (atom.name, atom.residue.number, atom.chain_id),)
            area = freesasa.selectArea(select_str, structure, result)['atom']

        if numpy.isnan(area):
            raise ValueError("freesasa returned {} for {}:{}".format(area, pdb_path, atom))

        graph.nodes[atom]['sasa'] = area


@time_profile
def _cluster(graph_data, entry_group):
    clustering_group = entry_group.create_group("clustering")

    method_name = "louvain"

    method_group = clustering_group.create_group(method_name)

    data = graph_data.as_torch(EDGENAME_NONBONDED, EDGENAME_BONDED, POSITION_FEATURE_NAME)

    # as in deeprank:

    cluster0 = community_detection(data.internal_edge_index, data.num_nodes, method=method_name)
    method_group.create_dataset('depth_0', data=cluster0)

    data = community_pooling(cluster0, data)

    cluster1 = community_detection(data.internal_edge_index, data.num_nodes, method=method_name)
    method_group.create_dataset('depth_1', data=cluster1)


_MAX_BONDED_DISTANCE = 1.6
_MAX_BONDED_DISTANCE_SS = 2.2
_MAX_NONBONDED_DISTANCE = 10.0

_MAX_SQUARED_BONDED_DISTANCE = numpy.square(_MAX_BONDED_DISTANCE)
_MAX_SQUARED_BONDED_DISTANCE_SS = numpy.square(_MAX_BONDED_DISTANCE_SS)
_MAX_SQUARED_NONBONDED_DISTANCE = numpy.square(_MAX_NONBONDED_DISTANCE)


def _build_graph(variant, radius_around_variant):

    # extract pdb data
    t0 = time()
    pdb = pdb2sql(variant.pdb_path)
    try:
        _log.debug("looking for nearby atoms..")
        involved_atom_pairs = get_residue_contact_atom_pairs(pdb, variant.chain_id, variant.residue_number, radius_around_variant)

        t1 = time()
        _log.debug("took {} seconds to find nearby atoms".format(t1 - t0))

        # list all the involved atoms and chain ids
        all_atoms = set([])
        all_chains = set([])
        for atom1, atom2 in involved_atom_pairs:  # iterate over variant atom to surrouding atom pairs
            all_atoms.add(atom1)
            all_atoms.add(atom2)
            all_chains.add(atom1.chain_id)
            all_chains.add(atom2.chain_id)
    finally:
        pdb._close()

    # init networkx graph
    graph = networkx.Graph()
    graph.name = get_variant_group_name(variant)

    # compare every atom with every atom:
    _log.debug("looking for edges between {} atoms..".format(len(all_atoms)))
    for atom1 in all_atoms:
        if atom1.element == "H":
            continue  # may not be in forcefield

        for atom2 in all_atoms:
            if atom2.element == "H":
                continue  # may not be in forcefield

            if atom1 != atom2:
                squared_distance = get_squared_distance(atom1.position, atom2.position)
                if squared_distance <= 0.0:
                    continue  # prevent nan values

                if squared_distance < _MAX_SQUARED_BONDED_DISTANCE or \
                        atom1.element == "S" and atom2.element == "S" and squared_distance < _MAX_SQUARED_BONDED_DISTANCE_SS:

                    distance = get_distance(atom1.position, atom2.position)

                    graph.add_edge(atom1, atom2)
                    graph.edges[atom1, atom2]["dist"] = distance
                    graph.edges[atom1, atom2][EDGETYPE_FEATURE_NAME] = bytes(EDGENAME_BONDED, encoding=EDGETYPE_ENCODING)
                    graph.edges[atom1, atom2]["coulomb"] = atomic_forcefield.get_coulomb_energy(atom1, atom2)
                    graph.edges[atom1, atom2]["vanderwaals"] = atomic_forcefield.get_vanderwaals_energy(atom1, atom2)

                elif (variant.is_at(atom1.residue) or variant.is_at(atom2.residue)) and squared_distance < _MAX_SQUARED_NONBONDED_DISTANCE:

                    distance = get_distance(atom1.position, atom2.position)

                    graph.add_edge(atom1, atom2)
                    graph.edges[atom1, atom2]["dist"] = distance
                    graph.edges[atom1, atom2][EDGETYPE_FEATURE_NAME] = bytes(EDGENAME_NONBONDED, encoding=EDGETYPE_ENCODING)
                    graph.edges[atom1, atom2]["coulomb"] = atomic_forcefield.get_coulomb_energy(atom1, atom2)
                    graph.edges[atom1, atom2]["vanderwaals"] = atomic_forcefield.get_vanderwaals_energy(atom1, atom2)

    t4 = time()
    _log.debug("took {} seconds to find edges + features".format(t4 - t1))

    # add features to nodes
    _log.debug("adding node features..")
    chain_nn_representations = {chain_id: index for index, chain_id in enumerate(all_chains)}
    for atom, atom_node in graph.nodes.items():
        atom_node[POSITION_FEATURE_NAME] = atom.position
        atom_node["element"] = ELEMENT_NN_REPRESENTATIONS[atom.element]
        atom_node["chain"] = chain_nn_representations[atom.chain_id]

        if variant.is_at(atom.residue):

            atom_node["wildtype"] = AMINO_ACID_NN_REPRESENTATIONS[variant.wildtype_amino_acid]
            atom_node["variant"] = AMINO_ACID_NN_REPRESENTATIONS[variant.variant_amino_acid]
        else:
            atom_node["wildtype"] = AMINO_ACID_NN_REPRESENTATIONS[atom.residue.amino_acid]
            atom_node["variant"] = AMINO_ACID_NN_REPRESENTATIONS[atom.residue.amino_acid]

        charge = atomic_forcefield.get_charge(atom)
        if numpy.isnan(charge):
            raise ValueError("got NaN charge for {}:{}".format(variant.pdb_path, atom))

        atom_node["charge"] = atomic_forcefield.get_charge(atom)

    t5 = time()
    _log.debug("took {} seconds to add node features".format(t5 - t4))

    return graph


def add_as_graph(variant, hdf5_path, radius_around_variant=10.0):
    """
        Args:
            variant(PdbVariantSelection): represents the single-residue variant
            hdf5_file(h5py.File): file to add the preprocessed data to
            radius_around_variant(float): ångström radius of residues around a variant to include in the graph
    """

    _log.debug("building graph for {}".format(variant))

    # build graph
    deeprank_graph = _build_graph(variant, radius_around_variant)

    # add additional features
    _add_sasa(deeprank_graph, variant.pdb_path)

    if variant.has_pssm():
        _add_pssm(deeprank_graph, variant)

    graph_data = graph_to_data(deeprank_graph)

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        if deeprank_graph.name in hdf5_file:
            raise ValueError("file '{}' already has a group named '{}'".format(hdf5_path, deeprank_graph.name))

        entry_group = hdf5_file.create_group(deeprank_graph.name)

        try:
            # write the graph to the hdf5 entry
            data_to_deeprank_hdf5(graph_data, entry_group,
                                  edge_type_chosen=EDGENAME_NONBONDED,
                                  internal_edge_type_chosen=EDGENAME_BONDED)

            # write the variant class to the hdf5 entry
            if variant.variant_class is not None:
                score_group = entry_group.create_group('score')
                score_group.create_dataset("bin_class", data=[int(variant.variant_class)])

            # write the variant metadata to the hdf5 entry
            store_variant(entry_group, variant)
        except:
            # On error, don't leave behind a partly finished entry!
            del hdf5_file[deeprank_graph.name]
            raise
