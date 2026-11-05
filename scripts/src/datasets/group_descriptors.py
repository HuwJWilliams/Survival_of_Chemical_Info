"""
Grouping functions for molecular descriptors (RDKit, Mordred, etc.)
Each function returns a dictionary of descriptor groups.
"""

from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors


def getRDKitGroups(prefix="_rdkit"):
    """Return RDKit descriptor groups aligned to Mordred descriptor categories."""

    def p(name):
        return f"{name}{prefix}"

    def get_fragment_names():
        return [
            f"{name}{prefix}"
            for name, _ in Descriptors.descList
            if name.startswith("fr_")
        ]

    return {
        # --- Constitutional / Weight ---
        "Constitutional": [
            p("MolWt"),
            p("HeavyAtomMolWt"),
            p("ExactMolWt"),
            p("NumValenceElectrons"),
            p("HeavyAtomCount"),
            p("FractionCSP3"),
            p("NumHeteroatoms"),
        ],
        "Weight": [p("MolWt"), p("ExactMolWt"), p("HeavyAtomMolWt")],
        # --- Electronic / EState ---
        "EState": [
            p("MaxEStateIndex"),
            p("MinEStateIndex"),
            p("MaxAbsEStateIndex"),
            p("MinAbsEStateIndex"),
            *[p(f"EState_VSA{i}") for i in range(1, 12)],
            *[p(f"VSA_EState{i}") for i in range(1, 11)],
        ],
        "TopologicalCharge": [
            p("MaxPartialCharge"),
            p("MinPartialCharge"),
            p("MaxAbsPartialCharge"),
            p("MinAbsPartialCharge"),
        ],
        # --- VSA / MOE-type ---
        "MoeType": [
            *[p(f"PEOE_VSA{i}") for i in range(1, 15)],
            *[p(f"SlogP_VSA{i}") for i in range(1, 13)],
            *[p(f"SMR_VSA{i}") for i in range(1, 11)],
        ],
        # --- Physicochemical ---
        "SLogP": [p("MolLogP"), *[p(f"SlogP_VSA{i}") for i in range(1, 13)]],
        "TopoPSA": [p("TPSA")],
        "Polarizability": [p("MolMR"), *[p(f"SMR_VSA{i}") for i in range(1, 11)]],
        # --- Topological indices ---
        "Chi": [
            p("Chi0"),
            p("Chi0n"),
            p("Chi0v"),
            p("Chi1"),
            p("Chi1n"),
            p("Chi1v"),
            p("Chi2n"),
            p("Chi2v"),
            p("Chi3n"),
            p("Chi3v"),
            p("Chi4n"),
            p("Chi4v"),
        ],
        "KappaShapeIndex": [p("HallKierAlpha"), p("Kappa1"), p("Kappa2"), p("Kappa3")],
        "BalabanJ": [p("BalabanJ")],
        "BertzCT": [p("BertzCT")],
        "InformationContent": [p("Ipc"), p("AvgIpc")],
        # --- BCUT ---
        "BCUT": [
            p("BCUT2D_MWHI"),
            p("BCUT2D_MWLOW"),
            p("BCUT2D_CHGHI"),
            p("BCUT2D_CHGLO"),
            p("BCUT2D_LOGPHI"),
            p("BCUT2D_LOGPLOW"),
            p("BCUT2D_MRHI"),
            p("BCUT2D_MRLOW"),
        ],
        # --- Surface / geometry ---
        "CPSA": [p("LabuteASA")],  # closest proxy
        "Geometrical": [
            p("Phi"),
            p("SPS"),
        ],
        # --- Rings ---
        "RingCount": [
            p("RingCount"),
            p("NumAromaticCarbocycles"),
            p("NumAromaticHeterocycles"),
            p("NumAromaticRings"),
            p("NumAliphaticCarbocycles"),
            p("NumAliphaticHeterocycles"),
            p("NumAliphaticRings"),
            p("NumSaturatedCarbocycles"),
            p("NumSaturatedHeterocycles"),
            p("NumSaturatedRings"),
            p("NumBridgeheadAtoms"),
            p("NumHeterocycles"),
        ],
        "Aromatic": [
            p("NumAromaticCarbocycles"),
            p("NumAromaticHeterocycles"),
            p("NumAromaticRings"),
        ],
        # --- Bonds / flexibility ---
        "RotatableBond": [p("NumRotatableBonds")],
        "HydrogenBond": [
            p("NumHAcceptors"),
            p("NumHDonors"),
            p("NHOHCount"),
            p("NOCount"),
        ],
        "BondCount": [p("NumAmideBonds")],
        # --- Stereo ---
        "Stereochemistry": [
            p("NumAtomStereoCenters"),
            p("NumUnspecifiedAtomStereoCenters"),
        ],
        # --- Drug-likeness ---
        "Lipinski": [p("NumHAcceptors"), p("NumHDonors"), p("MolLogP"), p("MolWt")],
        "Druglikeness": [p("qed")],
        # --- Fragment-based ---
        "FragmentComplexity": get_fragment_names(),
        # --- Morgan fingerprint density (non-Mordred but useful) ---
        "FingerprintDensity": [
            p("FpDensityMorgan1"),
            p("FpDensityMorgan2"),
            p("FpDensityMorgan3"),
        ],
    }


# %%
def getMordredGroups(prefix="_mordred"):
    """Return descriptor groups for Mordred descriptors."""

    groups = {}

    calc = Calculator(descriptors, ignore_3D=False)

    for desc in calc.descriptors:
        module = desc.__module__.split(".")[-1]
        name = str(desc).split(".")[-1]

        # apply prefix
        name = f"{name}{prefix}"

        groups.setdefault(module, []).append(name)

    return groups


def getMACCSGroups(prefix="_maccs"):
    """Returns grouped MACCS fingerprint bits.

    Bit descriptions:
    - https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/MACCSkeys.py
    - https://www.mayachemtools.org/docs/modules/pdf/MACCSKeys.pdf

    Notes:
    - Groups are semantic rather than canonical; some MACCS bits
      belong to multiple groups. e.g., nitrogen and sulfur containing motifs
    - This function uses a mostly single-assignment scheme for practicality.
    """

    def p(bit):
        return f"{bit}{prefix}"

    return {
        "Inorganic-Elements": [
            p(bit) for bit in [2, 3, 4, 5, 6, 7, 9, 10, 12, 18, 20, 35]
        ],
        "Organic-Elements": [
            p(bit) for bit in [27, 29, 42, 46, 88, 103, 134, 161, 164]
        ],
        "Ring-Heterocycle": [
            p(bit)
            for bit in [
                8,
                11,
                16,
                19,
                22,
                36,
                57,
                83,
                96,
                98,
                101,
                120,
                121,
                137,
                145,
                163,
                165,
            ]
        ],
        "Aromatic-Presence": [p(bit) for bit in [125, 144, 162]],
        "Aromatic-Connectivity": [
            p(bit)
            for bit in [26, 59, 62, 64, 75, 87, 105, 113, 127, 133, 135, 143, 150]
        ],
        "Nitrogen-motifs": [
            p(bit)
            for bit in [
                13,
                23,
                25,
                32,
                33,
                37,
                38,
                47,
                56,
                65,
                70,
                77,
                79,
                80,
                84,
                85,
                92,
                94,
                95,
                97,
                110,
                117,
                122,
                156,
            ]
        ],
        "Sulfur-motifs": [p(bit) for bit in [32, 33, 39, 47, 51, 55, 58, 61, 67, 81]],
        "Oxygen-motifs": [
            p(bit)
            for bit in [
                13,
                15,
                23,
                37,
                39,
                48,
                51,
                55,
                56,
                72,
                89,
                92,
                95,
                97,
                102,
                110,
                117,
                123,
                152,
            ]
        ],
        "Generic-Heteroatom-motifs": [
            p(bit) for bit in [28, 30, 31, 43, 53, 54, 68, 69, 106, 107, 148]
        ],
        "Alkyl-Motifs": [p(bit) for bit in [66, 74, 108, 114, 115, 116, 160]],
        "CH2-Spacers": [p(bit) for bit in [82, 86, 100, 109, 128, 129, 147, 153, 155]],
        "Branching": [p(bit) for bit in [66, 112]],
        "Heteroatom-Spacing": [p(bit) for bit in [90, 91, 93, 104, 111, 124, 132]],
        "Bond-Unsaturation": [
            p(bit)
            for bit in [
                14,
                17,
                21,
                24,
                34,
                40,
                41,
                45,
                49,
                50,
                52,
                60,
                63,
                71,
                73,
                76,
                78,
                99,
                119,
                126,
                136,
                139,
                151,
                154,
                157,
                158,
            ]
        ],
        "Count-based": [
            p(bit)
            for bit in [
                118,
                120,
                125,
                127,
                130,
                131,
                136,
                138,
                140,
                141,
                142,
                145,
                146,
                149,
                159,
            ]
        ],
        "Special": [p(bit) for bit in [1, 44, 166]],
    }


def getGroups(source):
    """Return descriptor groups depending on source."""
    if source.lower() == "rdkit":
        return getRDKitGroups()
    elif source.lower() == "mordred":
        return getMordredGroups()
    elif source.lower() == "maccs":
        return getMACCSGroups()
    else:
        raise ValueError(
            f"Unknown source '{source}'. Choose 'rdkit', 'mordred' or 'maccs'."
        )


def findDescriptorGroup(descriptor_name: str) -> tuple[str, str]:
    """
    Return (descriptor_set, group) for a descriptor name.

    Accepts either suffixed or unsuffixed descriptor names:
    - PNSA3
    - PNSA3_mordred
    - MolWt
    - MolWt_rdkit
    - 125
    - 125_maccs
    """

    descriptor_name = str(descriptor_name)

    descriptor_sets = {
        "rdkit": getRDKitGroups,
        "mordred": getMordredGroups,
        "maccs": getMACCSGroups,
    }

    matches = []

    for descriptor_set, group_func in descriptor_sets.items():
        group_map = group_func()
        suffix = f"_{descriptor_set}"

        # Normalise the query for this descriptor set
        if descriptor_name.endswith(suffix):
            candidate = descriptor_name
        else:
            candidate = f"{descriptor_name}{suffix}"

        for group, members in group_map.items():
            if candidate in members:
                matches.append((descriptor_set, group))

    if not matches:
        raise ValueError(
            f"Descriptor '{descriptor_name}' was not found in any descriptor group."
        )

    if len(matches) > 1:
        print(
            f"Warning: descriptor '{descriptor_name}' appears in multiple groups: "
            f"{matches}. Returning first match."
        )

    return matches[0]
