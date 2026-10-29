"""The simulator's iiwa 14 is the arm cable_assembly moves. sim/scripts/build_iiwa14.py expands
lbr-stack's xacro into the URDF the Isaac Lab USD is converted from; here its kinematic chain is held
to the vendored description, which test_kinematics_iiwa14.py in turn holds to kinematics/iiwa14.py.
A policy trained on a different chain would be trained on a different robot.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
SIM_URDF = os.path.join(ROOT, 'sim', 'src', 'cable_assembly_sim', 'assets', 'data', 'iiwa14', 'iiwa14.urdf')
REAL_URDF = os.path.join(ROOT, 'cable_assembly', 'robot', 'description', 'iiwa14', 'iiwa14.urdf')
CHAIN = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'link_7-tool0']


def _joints(path):
    return {j.get('name'): j for j in ET.parse(path).getroot().findall('joint')}


def _numbers(element, tag, attr):
    node = element.find(tag)
    return None if node is None else [float(v) for v in node.get(attr).split()]


def test_chain_is_the_real_arms():
    sim, real = _joints(SIM_URDF), _joints(REAL_URDF)
    for name in CHAIN:
        s, r = sim[name], real[name]
        assert (s.find('parent').get('link'), s.find('child').get('link')) == \
            (r.find('parent').get('link'), r.find('child').get('link')), name
        for tag, attr in [('origin', 'xyz'), ('origin', 'rpy'), ('axis', 'xyz')]:
            assert _numbers(s, tag, attr) == _numbers(r, tag, attr), (name, tag, attr)
        if name != 'link_7-tool0':
            for attr in ('lower', 'upper', 'velocity'):
                assert float(s.find('limit').get(attr)) == float(r.find('limit').get(attr)), (name, attr)


def test_only_the_chain_moves():
    """No world link (the converter fixes the base) and no lab-specific link_ee: the one fixed
    joint is tool0, so nothing else can shift the frame tool offsets are measured from."""
    sim = _joints(SIM_URDF)
    assert sorted(sim) == sorted(CHAIN)
    assert [j.get('type') for j in sim.values()].count('fixed') == 1


def test_every_link_has_mass_and_meshes():
    links = {l.get('name'): l for l in ET.parse(SIM_URDF).getroot().findall('link')}
    assert sorted(links) == sorted([f'link_{i}' for i in range(8)] + ['tool0'])
    for name, link in links.items():
        inertial = link.find('inertial')
        assert float(inertial.find('mass').get('value')) > 0, name
        i = inertial.find('inertia').attrib
        I = np.array([[i['ixx'], i['ixy'], i['ixz']], [i['ixy'], i['iyy'], i['iyz']],
                      [i['ixz'], i['iyz'], i['izz']]], dtype=float)
        assert np.all(np.linalg.eigvalsh(I) > 0), name
        if name != 'tool0':
            assert link.find('visual/geometry/mesh') is not None and link.find('collision/geometry/mesh') is not None


def test_meshes_resolve_once_fetched():
    meshes = [m.get('filename') for m in ET.parse(SIM_URDF).getroot().iter('mesh')]
    assert len(meshes) == 16 and all(not m.startswith('package://') for m in meshes)
    base = os.path.dirname(SIM_URDF)
    if not os.path.isdir(os.path.join(base, 'meshes')):
        pytest.skip('meshes not fetched: python sim/scripts/build_iiwa14.py --stage urdf')
    missing = [m for m in meshes if not os.path.isfile(os.path.join(base, m))]
    assert not missing, missing
