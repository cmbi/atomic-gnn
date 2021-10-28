from enum import Enum
import logging

import h5py
import numpy
import pdb2sql
import networkx
import freesasa

from deeprank_gnn.Graph import Graph as DeepRankGraph
from deeprank_gnn.ResidueGraph import ResidueGraph
from deeprank_gnn.operate.pdb import get_residue_contact_atom_pairs, get_distance
from deeprank_gnn.operate.hdf5data import get_variant_group_name, store_variant
from deeprank_gnn.models.residue import Residue
from deeprank_gnn.tools.PSSM import PSSM_aligned
from deeprank_gnn.tools import BioWrappers


_log = logging.getLogger(__name__)


class ResiduePolarity(Enum):
    APOLAR = 0
    POLAR = 1
    CHARGE_NEGATIVE = 2
    CHARGE_POSITIVE = 3

    def __int__(self):
        return self.value


AMINO_ACID_CHARGES = {'CYS': -0.64, 'HIS': -0.29, 'ASN': -1.22, 'GLN': -1.22, 'SER': -0.80, 'THR': -0.80, 'TYR': -0.80,
                      'TRP': -0.79, 'ALA': -0.37, 'PHE': -0.37, 'GLY': -0.37, 'ILE': -0.37, 'VAL': -0.37, 'MET': -0.37,
                      'PRO': 0.0, 'LEU': -0.37, 'GLU': -1.37, 'ASP': -1.37, 'LYS': -0.36, 'ARG': -1.65}

AMINO_ACID_ORDER = sorted(AMINO_ACID_CHARGES.keys())
AMINO_ACID_VALUES = {code: index for index, code in enumerate(AMINO_ACID_ORDER)}


AMINO_ACID_POLARITY = {'CYS': ResiduePolarity.POLAR, 'HIS': ResiduePolarity.POLAR, 'ASN': ResiduePolarity.POLAR,
                       'GLN': ResiduePolarity.POLAR, 'SER': ResiduePolarity.POLAR, 'THR': ResiduePolarity.POLAR,
                       'TYR': ResiduePolarity.POLAR, 'TRP': ResiduePolarity.POLAR,
                       'ALA': ResiduePolarity.APOLAR, 'PHE': ResiduePolarity.APOLAR, 'GLY': ResiduePolarity.APOLAR,
                       'ILE': ResiduePolarity.APOLAR, 'VAL': ResiduePolarity.APOLAR, 'MET': ResiduePolarity.APOLAR,
                       'PRO': ResiduePolarity.APOLAR, 'LEU': ResiduePolarity.APOLAR,
                       'GLU': ResiduePolarity.CHARGE_NEGATIVE, 'ASP': ResiduePolarity.CHARGE_NEGATIVE,
                       'LYS': ResiduePolarity.CHARGE_NEGATIVE, 'ARG': ResiduePolarity.CHARGE_POSITIVE}



def _add_pair_to_graph(graph, residue1, residue2, distance, edge_type_string):

    # add edge features
    if (residue1, residue2) in graph.edges:

        if distance < graph.edges[residue1, residue2]['dist']:

            graph.edges[residue1, residue2]['dist'] = distance

        # else keep original distance

    else:  # edge is not yet in the graph
        graph.add_edge(residue1, residue2, dist=distance, type=bytes(edge_type_string, encoding='utf-8'))

    # add node features
    for residue in [residue1, residue2]:
        mean_position = numpy.mean([atom.position for atom in residue.atoms], axis=0)
        graph.nodes[residue]["pos"] = mean_position
        graph.nodes[residue]["type"] = AMINO_ACID_VALUES[residue.name]
        graph.nodes[residue]["charge"] = AMINO_ACID_CHARGES[residue.name]
        graph.nodes[residue]["polarity"] = int(AMINO_ACID_POLARITY[residue.name])


def _add_edge_indices(graph):
    residues = list(graph.nodes)

    for edge in graph.edges:
        residue1, residue2 = edge
        graph.edge_index.append([residues.index(residue1), residues.index(residue2)])


def _add_sasa(graph, pdb_path):
    structure = freesasa.Structure(pdb_path)
    result = freesasa.calc(structure)

    for residue in graph.nodes:
        select_str = ('res, (resi %d) and (chain %s)' % (residue.number, residue.chain_id),)
        area = freesasa.selectArea(select_str, structure, result)['res']
        graph.nodes[residue]['sasa'] = area


PSSM_AMINO_ACID_INDICES = {'CYS': 4, 'HIS': 8, 'ASN': 2, 'GLN': 5, 'SER': 15, 'THR': 16, 'TYR': 18, 'TRP': 17, 
                           'ALA': 0, 'PHE': 13, 'GLY': 7, 'ILE': 9, 'VAL': 19, 'MET': 12, 'PRO': 14, 'LEU': 10, 
                           'GLU': 6, 'ASP': 3, 'LYS': 11, 'ARG': 1}


def _add_pssm(graph, pssm_paths):
    pssm, ic = PSSM_aligned(pssm_paths, style='res')

    for residue in graph.nodes:
        residue_key = (residue.chain_id, residue.number, residue.name)

        if residue_key in pssm:
            graph.nodes[residue]['pssm'] = pssm[residue_key]
            graph.nodes[residue]['cons'] = pssm[residue_key][PSSM_AMINO_ACID_INDICES[residue.name]]
        else:
            graph.nodes[residue]['pssm'] = [0.0] * len(PSSM_AMINO_ACID_INDICES)
            graph.nodes[residue]['cons'] = 0.0

        if residue_key in ic:
            graph.nodes[residue]['ic'] = ic[residue_key]
        else:
            graph.nodes[residue]['ic'] = 0.0


def _add_biopython(graph, pdb_path):
    model = BioWrappers.get_bio_model(pdb_path)

    residue_depth = BioWrappers.get_depth_contact_res(model,
                                                      [(residue.chain_id, residue.number, residue.name)
                                                       for residue in graph.nodes])
    hse = BioWrappers.get_hse(model)

    for residue in graph.nodes:
        residue_key = (residue.chain_id, residue.number, residue.name)
        graph.nodes[residue]['depth'] = residue_depth.get(residue_key, 0.0)
        graph.nodes[residue]['hse'] = hse.get(residue_key, (0, 0, 0))


def _build_graph(variant, max_interatomic_distance, max_intermolecular_distance):
    try:
        db = pdb2sql.interface(variant.pdb_path)

        graph = networkx.Graph()
        graph.edge_index = []
        graph.internal_edge_index = []

        residue_positions = {}

        for atom1, atom2 in get_residue_contact_atom_pairs(db, variant.chain_id, variant.residue_number,
                                                           max_interatomic_distance):
            if atom1.chain_id == atom2.chain_id:

                distance = get_distance(atom1.position, atom2.position)

                _add_pair_to_graph(graph, atom1.residue, atom2.residue, distance, 'internal')

        _add_edge_indices(graph)
        return graph
    finally:
        db._close()


def add_as_graph(variant, hdf5_path, max_interatomic_distance=3.0, max_intermolecular_distance=8.5):

    # build graph
    deeprank_graph = DeepRankGraph()
    deeprank_graph.name = get_variant_group_name(variant)
    deeprank_graph.nx = _build_graph(variant, max_interatomic_distance, max_intermolecular_distance)

    # add extra features
    _add_sasa(deeprank_graph.nx, variant.pdb_path)
    _add_pssm(deeprank_graph.nx, {chain_id: variant.get_pssm_path(chain_id)
                                  for chain_id in variant.get_pssm_chains()})
    _add_biopython(deeprank_graph.nx, variant.pdb_path)

    with h5py.File(hdf5_path, 'a') as hdf5_file:

        # write graph to hdf5
        deeprank_graph.nx2h5(hdf5_file)

        # write the variant class (target)
        variant_group = hdf5_file[deeprank_graph.name]
        score_group = variant_group['score']
        score_group.create_dataset("class", data=[int(variant.variant_class)])

        # precluster
        _cluster(

        # write the variant itself
        store_variant(variant_group, variant)
