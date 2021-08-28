from rdkit import Chem


def is_correct(smi):
    if Chem.MolFromSmiles(smi) is None:
        # TODO: Add an exception/error/warning file.
        raise BaseException("The SMILES can not be parsed correctly in rdkit!")


def to_canonical(smi,tool="rdkit"):
    '''
    There is no universal canonical SMILES.
    Every toolkit uses a different algorithm,
    and sometimes the algorithm changes with different versions of the toolkit.
    There are even different forms of canonical SMILES,
    depending on if atomic properties like isotope are important for the result.
    https://ctr.fandom.com/wiki/Convert_a_SMILES_string_to_canonical_SMILES
    TODO: add tool: CDK/Groovy Indigo/python Openbabel/Pybel OpenEye/Python Cactvs/Python
    '''

    assert is_correct(smi)
    if tool=="rdkit":
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        raise BaseException("Can't find indicated tool to parse SMILES")




