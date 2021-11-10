def tutorial1():
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    << return_X_y = True
                    >> smiles 0

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Prepare,feature representation)
                    << host = chemml
                    << function = RDKitFingerprint
                    << molfile = @molfile
                    >> 1 molfile

            """
    return script.strip().split('\n')


def tutorial2():
    script = """
                ## (Enter,datasets)
                    << host = chemml
                    << function = load_cep_homo
                    << return_X_y = True
                    >> smiles 0

                ## (Store,file)
                    << host = chemml
                    << function = SaveFile
                    << format = smi
                    << header = False
                    << filename = smiles
                    >> 0 df
                    >> filepath 1

                ## (Prepare,feature representation)
                    << host = chemml
                    << function = RDKitFingerprint
                    << molfile = @molfile
                    >> 1 molfile

            """
    return script.strip().split('\n')
