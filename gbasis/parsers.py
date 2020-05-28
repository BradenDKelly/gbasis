"""Parsers for reading basis set files."""
import re

import numpy as np
from gbasis.contractions import GeneralizedContractionShell


def parse_nwchem(nwchem_basis_file):
    """Parse nwchem basis set file.

    Parameters
    ----------
    nwchem_basis_file : str
        Path to the nwchem basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater than "k", an error will be raised.

    """
    # pylint: disable=R0914
    with open(nwchem_basis_file, "r") as basis_fh:
        nwchem_basis = basis_fh.read()

    data = re.split(r"\n\s*(\w[\w]?)[ ]+(\w+)\s*\n", nwchem_basis)
    dict_angmom = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    # remove first part
    if "\n" in data[0]:  # pragma: no branch
        data = data[1:]
    atoms = data[::3]
    angmoms = data[1::3]
    exps_coeffs_all = data[2::3]
    # trim out headers at the end
    output = {}
    for atom, angmom_gen, exps_coeffs in zip(atoms, angmoms, exps_coeffs_all):
        output.setdefault(atom, [])
        angmom_seg = [dict_angmom[i.lower()] for i in angmom_gen]
        exps_coeffs = exps_coeffs.split("\n")
        exps = []
        coeffs_gen = []
        for line in exps_coeffs:
            test = re.search(
                r"^\s*([0-9\.DE\+\-]+)\s+((?:(?:[0-9\.DE\+\-]+)\s+)*(?:[0-9\.DE\+\-]+))\s*$", line
            )
            try:
                exp, coeff_gen = test.groups()
                coeff_gen = re.split(r"\s+", coeff_gen)
            except AttributeError:
                continue
            # clean up
            exp = float(exp.lower().replace("d", "e"))
            coeff_gen = [float(i.lower().replace("d", "e")) for i in coeff_gen if i is not None]
            exps.append(exp)
            coeffs_gen.append(coeff_gen)
        exps = np.array(exps)
        coeffs_gen = np.array(coeffs_gen)

        if len(angmom_seg) == 1:
            output[atom].append((angmom_seg[0], exps, coeffs_gen))
        else:
            for i, angmom in enumerate(angmom_seg):
                output[atom].append((angmom, exps, coeffs_gen[:, i]))

    return output


def parse_gbs(gbs_basis_file):
    """Parse Gaussian94 basis set file.

    Parameters
    ----------
    gbs_basis_file : str
        Path to the Gaussian94 basis set file.

    Returns
    -------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Dictionary of the element to the list of angular momentum, exponents, and contraction
        coefficients associated with each contraction at the given atom.

    Notes
    -----
    Angular momentum symbol is hard-coded into this function. This means that if the selected basis
    set has an angular momentum greater than "k", an error will be raised.

    Since Gaussian94 basis format does not explicitly state which contractions are generalized, we
    infer that subsequent contractions belong to the same generalized shell if they have the same
    exponents and angular momentum. If two contractions are not one after another or if they are
    associated with more than one angular momentum, they are treated to be segmented contractions.

    """
    # pylint: disable=R0914
    with open(gbs_basis_file, "r") as basis_fh:
        gbs_basis = basis_fh.read()
    # splits file into 'element', 'basis stuff', 'element',' basis stuff'
    # e.g., ['H','stuff with exponents & coefficients\n', 'C', 'stuff with etc\n']
    data = re.split(r"\n\s*(\w[\w]?)\s+\w+\s*\n", gbs_basis)
    dict_angmom = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7}
    # remove first part
    if "\n" in data[0]:  # pragma: no branch
        data = data[1:]
    atoms = data[::2]  # stride of 2 get the ['H','C', etc] from e.g., above
    basis = data[1::2]  # starting with 'stuff' take strides of 2 to skip elements
    # trim out headers at the end
    output = {}
    for atom, shells in zip(atoms, basis):
        output.setdefault(atom, [])

        shells = re.split(r"\n?\s*(\w+)\s+\w+\s+\w+\.\w+\s*\n", shells)
        # remove the ends
        atom_basis = shells[1:]
        # get angular momentums
        angmom_shells = atom_basis[::2]
        # get exponents and coefficients
        exps_coeffs_shells = atom_basis[1::2]

        for angmom_seg, exp_coeffs in zip(angmom_shells, exps_coeffs_shells):
            angmom_seg = [dict_angmom[i.lower()] for i in angmom_seg]
            exps = []
            coeffs_seg = []
            exp_coeffs = exp_coeffs.split("\n")
            for line in exp_coeffs:
                test = re.search(
                    r"^\s*([0-9\.DE\+\-]+)\s+((?:(?:[0-9\.DE\+\-]+)\s+)*(?:[0-9\.DE\+\-]+))\s*$",
                    line,
                )
                try:
                    exp, coeff_seg = test.groups()
                    coeff_seg = re.split(r"\s+", coeff_seg)
                except AttributeError:
                    continue
                # clean up
                exp = float(exp.lower().replace("d", "e"))
                coeff_seg = [float(i.lower().replace("d", "e")) for i in coeff_seg if i is not None]
                exps.append(exp)
                coeffs_seg.append(coeff_seg)
            exps = np.array(exps)
            coeffs_seg = np.array(coeffs_seg)

            # if len(angmom_seg) == 1:
            #     coeffs_seg = coeffs_seg[:, None]
            for i, angmom in enumerate(angmom_seg):

                # dummy fix for np.allclose
                if output[atom] and len(output[atom][-1][1]) == len(exps):
                    hstack = np.allclose(output[atom][-1][1], exps)
                else:
                    hstack = False

                if (
                        output[atom]
                        and output[atom][-1][0] == angmom
                        and hstack
                        # and np.allclose(output[atom][-1][1], exps)
                ):
                    output[atom][-1] = (
                        angmom,
                        exps,
                        np.hstack([output[atom][-1][2], coeffs_seg[:, i:i + 1]]),
                    )
                else:
                    output[atom].append((angmom, exps, coeffs_seg[:, i:i + 1]))

    return output


def make_contractions(basis_dict, atoms, coords):
    """Return the contractions that correspond to the given atoms for the given basis.

    Parameters
    ----------
    basis_dict : dict of str to list of 3-tuple of (int, np.ndarray, np.ndarray)
        Output of the parsers from gbasis.parsers.
    atoms : N-list/tuple of str
        Atoms at which the contractions are centered.
    coords : np.ndarray(N, 3)
        Coordinates of each atom.

    Returns
    -------
    basis : tuple of GeneralizedContractionShell
        Contractions for each atom.
        Contractions are ordered in the same order as in the values of `basis_dict`.

    Raises
    ------
    TypeError
        If `atoms` is not a list or tuple of strings.
        If `coords` is not a two-dimensional `numpy` array with 3 columns.
    ValueError
        If the length of atoms is not equal to the number of rows of `coords`.

    """
    if not (isinstance(atoms, (list, tuple)) and all(isinstance(i, str) for i in atoms)):
        raise TypeError("Atoms must be provided as a list or tuple.")
    if not (isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] == 3):
        raise TypeError(
            "Coordinates must be provided as a two-dimensional `numpy` array with three columns."
        )
    if len(atoms) != coords.shape[0]:
        raise ValueError("Number of atoms must be equal to the number of rows in the coordinates.")

    # implement screening before saving basis as a tuple
    overlap_screen = True
    if overlap_screen:
        eps = 1E-20  # Tolerance for screening
        le = np.log(eps)
        dtest = []

        """ Each atom has several shells. Each shell on each atom, needs a list.
        The list controls if the shell interacts with the other shells.

        Each atom also needs a list saying if it even bothers interacting with 
        the shells on another atom at all."""

        # loop over all atoms, index is "A"
        counnt = 0
        max_rij_term = np.zeros((len(atoms), len(atoms)), dtype=float)
        max_shell_term = []
        rij = []
        prefactor = []
        for A, (atomA, coordA) in enumerate(zip(atoms, coords)):
            print(atomA)
            # loop over all atoms, index is "B"
            for B, (atomB, coordB) in enumerate(zip(atoms, coords)):
                shells_A2B = []
                delta_coords = []
                delta = np.sqrt(np.dot(coordA - coordB, coordA - coordB))
                # loop over all shells on atomA
                for S, (angmom_s, exp_s, coeff_s) in enumerate(basis_dict[atomA]):
                    alpha_s = min(exp_s)
                    # loop over all shells on atomB
                    for T, (angmom_t, exp_t, coeff_t) in enumerate(basis_dict[atomB]):
                        alpha_t = min(exp_t)
                        # calculate distance
                        d_AsBt = np.sqrt(-(alpha_s + alpha_t) / (alpha_s * alpha_t) * le)
                        shells_A2B.append(d_AsBt)
                        delta_coords.append(delta)
                        # print(d_AsBt)
                        counnt += 1
                        dtest.append(d_AsBt)
                max_rij_term[A, B] = max(np.asarray(shells_A2B))
            max_shell_term.append(shells_A2B)
            rij.append(delta_coords)
        max_shell_term, rij = tuple(max_shell_term), tuple(rij)
        dtest = np.asarray(dtest)
        print('minimum: {}  maximum: {}  average: {}  std_dev: {}'.format(
            np.min(dtest), np.max(dtest), np.average(dtest), np.std(dtest)
        ))
        #print(counnt)
        #print(max_rij_term)
        #print(len(rij[0]))
        assert np.shape(rij[0]) == np.shape(max_shell_term[0])
        mask = np.where(np.asarray(rij) > np.asarray(max_shell_term), False, True)
        #print("testing:", np.shape(mask))

    ##############################
    #
    # Make masks for screening
    #   1) outer loop over all shells
    #       1a) for each atom, loop over all shells
    #   2) inner loop over all shells
    #       2b) for each atom, loop over all shells
    #
    ##############################
    import time
    starrt = time.time()
    screen_mask = []
    eps = 1E-20  # Tolerance for screening
    le = np.log(eps)
    #################################################################################
    for A, (atom_a, coord_a) in enumerate(zip(atoms,coords)):                       #
        for angmom_a, exp_a, coeff_a in basis_dict[atom_a]:                         #
            alpha_a = min(exp_a)                                                    #
            mask = []                                                               #
            #########################################################################
            for B, (atom_b, coord_b) in enumerate(zip(atoms, coords)):              #
                for angmom_b, exp_b, coeff_b in basis_dict[atom_b]:                 #
                    alpha_b = min(exp_a)                                            #
                    #################################################################
                    cutoff = np.sqrt(-(alpha_a + alpha_b) / (alpha_a * alpha_b) * le)
                    if np.linalg.norm(coord_a-coord_b) > cutoff and overlap_screen:
                        mask.append(False) # do not evaluate these two shells
                    else:
                        mask.append(True)  # do evaluate these two shells
            screen_mask.append(np.asarray(mask, dtype=bool))
    endd = time.time()
    print("screening took: ", endd - starrt)
    basis = []
    index = 0
    # screen_mask : type = np.array,
    #               length = # shells in system
    for atom, coord in zip(atoms, coords):
        for angmom, exps, coeffs in basis_dict[atom]:
            basis.append(GeneralizedContractionShell(angmom, coord, coeffs, exps, screen_mask[index], index)) # , screen_mask[index]
            index += 1
    return tuple(basis)
