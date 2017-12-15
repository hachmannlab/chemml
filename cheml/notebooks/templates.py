def template1():
    script = """## (Enter,xyz)
    << host = cheml
    << function = XYZreader
    << path_pattern = required_required
    >> molecules 0

## (Prepare,feature representation)
    << host = cheml
    << function = Coulomb_Matrix
    >> 0 molecules
    >> df 1

## (Store,file)
    << host = cheml
    << function = SaveFile
    << filename = required_required
    >> 1 df

"""
    return script.strip().split('\n')

