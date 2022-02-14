import pandas as pd
import numpy as np
import os
import warnings

from chemml.wrapper.base import BASE

##################################################################### 2 Prepare Data

# Enter

class XYZreader(BASE):
    def fit(self):
        try:
            from chemml.initialization import XYZreader
            model = XYZreader(**self.parameters)
            molecules = model.read()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'molecules':
                self.set_value(token, molecules)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            # elif token == 'max_n_atoms':
            #     self.set_value(token, model.max_n_atoms)
            #     self.outputs[token].count = order.count(token)
            #     self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class load_cep_homo(BASE):
    def fit(self):
        try:
            from chemml.datasets import load_cep_homo
            smiles,homo = load_cep_homo()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'smiles':
                self.set_value(token, smiles)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'homo':
                self.set_value(token, homo)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class load_organic_density(BASE):
    def fit(self):
        try:
            from chemml.datasets import load_organic_density
            smiles,density,features = load_organic_density()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'smiles':
                self.set_value(token, smiles)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'density':
                self.set_value(token, density)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'features':
                self.set_value(token, features)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class load_xyz_polarizability(BASE):
    def fit(self):
        try:
            from chemml.datasets import load_xyz_polarizability
            coordinates, pol = load_xyz_polarizability()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'coordinates':
                self.set_value(token, coordinates)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'polarizability':
                self.set_value(token, pol)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class load_comp_energy(BASE):
    def fit(self):
        try:
            from chemml.datasets import load_comp_energy
            entries, df = load_comp_energy()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'entries':
                self.set_value(token, entries)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'formation_energy':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class load_crystal_structures(BASE):
    def fit(self):
        try:
            from chemml.datasets import load_crystal_structures
            entries = load_crystal_structures()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'entries':
                self.set_value(token, entries)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs




class ConvertFile(BASE):
    def fit(self):
        self.paramFROMinput()
        # self.required('file_path',req=True)
        # file_path=self.inputs['file_path'].value
        print ('from:', self.parameters['from_format'])
        print ('to:', self.parameters['to_format'])
        # if 'file_path' not in self.parameters and '@' not in file_path:
            # self.parameters['file_path']=file_path
        try:
            from chemml.initialization import ConvertFile
            model = ConvertFile(**self.parameters)
            converted_file_paths=model.convert()
        except Exception as err:
            msg='@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__+': ' +err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg="@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1,self.Task,token)
                raise NameError(msg)
            elif token == 'converted_file_paths':
                self.set_value(token,converted_file_paths)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock,token)] = self.outputs[token]
        del self.inputs

class scatter2D(BASE):
    def fit(self):
        # self.paramFROMinput()
        self.required('dfx', req=True)
        dfx = self.inputs['dfx'].value
        self.required('dfy', req=True)
        dfy = self.inputs['dfy'].value
        if 'x' in self.parameters:
            x = self.parameters.pop('x')
        else:
            msg = "@Task #%i(%s): the x header or position is required" % (self.iblock + 1, self.Task)
            raise NameError(msg)
        if 'y' in self.parameters:
            y = self.parameters.pop('y')
        else:
            msg = "@Task #%i(%s): the y header or position is required" % (self.iblock + 1, self.Task)
            raise NameError(msg)
        try:
            from chemml.visualization import scatter2D
            sc = scatter2D(**self.parameters)
            fig = sc.plot(dfx,dfy,x,y)
            # plt.show()
        except Exception as err:
            msg='@Task #%i(%s): ' % (self.iblock + 1,self.Task)+ type(err).__name__+': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'fig':
                self.set_value(token, fig)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class hist(BASE):
    def fit(self):
        # self.paramFROMinput()
        self.required('dfx', req=True)
        dfx = self.inputs['dfx'].value
        if 'x' in self.parameters:
            x = self.parameters.pop('x')
        else:
            msg = "@Task #%i(%s): the x header or position is required" % (self.iblock + 1, self.Task)
            raise NameError(msg)
        try:
            from chemml.visualization import hist
            hg = hist(**self.parameters)
            fig = hg.plot(dfx, x)
        except Exception as err:
            msg='@Task #%i(%s): ' % (self.iblock + 1,self.Task)+ type(err).__name__+': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'fig':
                self.set_value(token, fig)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class decorator(BASE):
    def fit(self):
        # self.paramFROMinput()
        self.required('fig', req=True)
        fig = self.inputs['fig'].value
        font_params = {}
        for p in ['family','size','weight','style','variant']:
            if p in self.parameters:
                font_params[p] = self.parameters.pop(p)
        try:
            from chemml.visualization import decorator
            dec = decorator(**self.parameters)
            dec.matplotlib_font(**font_params)
            fig = dec.fit(fig)
        except Exception as err:
            msg='@Task #%i(%s): ' % (self.iblock + 1,self.Task)+ type(err).__name__+': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'fig':
                self.set_value(token, fig)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class SavePlot(BASE):
    def fit(self):
        self.required('fig', req=True)
        fig = self.inputs['fig'].value
        # self.paramFROMinput()
        try:
            from chemml.visualization import SavePlot
            sp = SavePlot(**self.parameters)
            sp.save(fig, self.Base.output_directory)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)

        del self.inputs

class PyScript(BASE):
    def fit(self):
        # step1: check inputs
        inputs = [token for token in self.inputs if self.inputs[token].value is not None]
        _locals=locals()
        # print("INPUTS: ",inputs)
        # print("self.parameters.keys(): ", self.parameters.keys())
        for token in inputs:
            # print("token,token.value : ",token,token.value)
            # print("self.inputs['token'].value: ", self.inputs[token].value)
            code = compile("%s = self.inputs['%s'].value"%(token,token), "<string>", "exec")
            exec (code)
        for line in sorted(self.parameters.keys()):
            # print("self.parameters[line]: ", self.parameters[line])
            code = compile(self.parameters[line], "<string>", "exec")
            # print("code: ",code)
            exec (code)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'ov1' or token == 'ov2' or token == 'ov3' or token == 'ov4' or token == 'ov5' or token == 'ov6':
                # print("type(token): ", type(token))
                self.set_value(token, _locals[token])
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock+1,self.Task,token)
                raise NameError(msg)
        del self.inputs


# Feature Representation

class RDKitFingerprint(BASE):
    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'molfile' in self.parameters:
            molfile = self.parameters['molfile']
            self.parameters.pop('molfile')
        else:
            self.required('molfile', req=True)
            molfile = self.inputs['molfile'].value
        # step3: check the dimension of input data frame
        # step4: extract parameters
        if 'path' in self.parameters:
            path = self.parameters.pop('path')
        else:
            path = None
        if 'arguments' in self.parameters:
            arguments = self.parameters.pop('arguments')
        else:
            arguments = []

        # step5: import module and make APIs
        try:
            from chemml.chem import RDKitFingerprint
            from chemml.chem.molecule import Molecule
            model = RDKitFingerprint(**self.parameters)
            # model.MolfromFile(molfile,path,*arguments)
            extensions = {
                    #   '.mol':       Chem.MolFromMolFile,
                    #   '.mol2':      Chem.MolFromMol2File,
                    #   '.pdb':       Chem.MolFromPDBFile,
                    #   '.tpl':       Chem.MolFromTPLFile,
                      '.smi':       'smiles',
                      '.smarts':    'smarts',
                      '.inchi':     'inchi',
                      }
            file_name, file_extension = os.path.splitext(molfile)
            if file_extension == '':
                msg = 'file extension not determined'
                raise ValueError(msg)
            elif file_extension not in extensions:
                msg = "file extension '%s' not available"%file_extension
                raise ValueError(msg)
            input_type=extensions[file_extension]
            molecules=[]
            if file_extension in ['.smi','.smarts']:
                mols = open(molfile, 'r')
                mols = mols.readlines()
                for i,x in enumerate(mols):
                    mol = Molecule(input = x.strip(), input_type = input_type)
                    if mol is None:
                        # self.removed_rows.append(i)
                        pass
                    else:
                       molecules.append(mol)
          
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, model.represent(molecules))
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            # elif token == 'removed_rows':
            #     self.set_value(token, model.removed_rows)
            #     self.outputs[token].count = order.count(token)
            #     self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class Dragon(BASE):
    def fit(self):
        # step1: check inputs
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        print ('molfile:', self.parameters['molFile'])

        # step3: check the dimension of input data frame
        # step4: extract  parameters
        if 'script' in self.parameters:
            script = self.parameters.pop('script')
        else:
            script = 'new'
        # if 'output_directory' in self.parameters:
        #     output_directory = self.parameters.pop('output_directory')
        # else:
        #     output_directory = ''
        output_directory = self.Base.output_directory #+ '/' + output_directory
        if 'SaveFilePath' not in self.parameters:
            self.parameters['SaveFilePath'] = "Dragon_descriptors.txt"

        # step5: import module and make APIs
        try:
            from chemml.chem import Dragon
            model = Dragon(**self.parameters)
            model.script_wizard(script, output_directory)
            model.run()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                df_path = model.data_path
                df = pd.read_csv(df_path, sep=None, engine='python')
                df = df.drop(['No.','NAME'],axis=1)
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete dragon descriptors and all inputs from memory
        os.remove(model.output_directory + self.parameters['SaveFilePath'])
        del self.inputs

class CoulombMatrix(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('molecules', req=True)
        molecules = self.inputs['molecules'].value
        try:
            from chemml.chem import CoulombMatrix
            model = CoulombMatrix(**self.parameters)
            # print("*************************************************************************\n \n")
            # print(molecules)
            # print("*************************************************************************\n \n")
            df = model.represent(molecules)
        except Exception as err:
            print (err.message)
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class BagofBonds(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('molecules', req=True)
        molecules = self.inputs['molecules'].value
        try:
            from chemml.chem import BagofBonds
            model = BagofBonds(**self.parameters)
            df = model.represent(molecules)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            # elif token == 'headers':
            #     self.set_value(token, model.headers)
            #     self.outputs[token].count = order.count(token)
            #     self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class DistanceMatrix(BASE):
    def fit(self):
        self.paramFROMinput()
        # check inputs
        self.required('df', req=True)
        try:
            from chemml.chem import DistanceMatrix
            model = DistanceMatrix(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                ar = self.inputs['df'].value
                df = pd.DataFrame(model.transform(ar))
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        del self.inputs

class APEAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import APEAttributeGenerator
            ape = APEAttributeGenerator()
            if 'packing_threshold' in self.parameters:
                if self.parameters['packing_threshold']:
                    ape.set_packing_threshold(self.parameters['packing_threshold'])
            if 'n_nearest_to_eval' in self.parameters:
                if self.parameters['n_nearest_to_eval']:
                    ape.set_n_nearest_to_eval(self.parameters['n_nearest_to_eval'])
            if 'radius_property' in self.parameters:
                if self.parameters['radius_property']:
                    ape.set_radius_property(self.parameters['radius_property'])
            df = ape.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class ChargeDependentAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import ChargeDependentAttributeGenerator
            cd = ChargeDependentAttributeGenerator()
            df = cd.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class ElementalPropertyAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import ElementalPropertyAttributeGenerator
            if 'elemental_properties' in self.parameters:
                elemental_properties = self.parameters.pop('elemental_properties')
            ep = ElementalPropertyAttributeGenerator(**self.parameters)
            if elemental_properties:
                ep.add_elemental_properties(elemental_properties)
            df = ep.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class ElementFractionAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import ElementFractionAttributeGenerator
            ef = ElementFractionAttributeGenerator()
            df = ef.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class ElementPairPropertyAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import ElementPairPropertyAttributeGenerator
            epp = ElementPairPropertyAttributeGenerator()
            if 'elemental_pair_properties' in self.parameters:
                if self.parameters['elemental_pair_properties']:
                    epp.add_elemental_pair_properties(
                        self.parameters['elemental_pair_properties'])
            df = epp.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class GCLPAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import GCLPAttributeGenerator
            gclp = GCLPAttributeGenerator()
            if 'count_phases' in self.parameters and self.parameters[
                'count_phases']:
                gclp.set_count_phases(self.parameters[
                'count_phases'])
            if 'phases' in self.parameters and self.parameters['phases'] and \
                    'energies' in self.parameters and self.parameters[
                'energies']:
                gclp.set_phases(self.parameters['phases'], self.parameters[
                    'energies'])
            df = gclp.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class IonicCompoundProximityAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import IonicCompoundProximityAttributeGenerator
            icp = IonicCompoundProximityAttributeGenerator()
            if 'max_formula_unit' in self.parameters and self.parameters[
                'max_formula_unit'] != 14:
                icp.set_max_formula_unit(self.parameters['max_formula_unit'])
            df = icp.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class IonicityAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import IonicityAttributeGenerator
            ig = IonicityAttributeGenerator()
            df = ig.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class MeredigAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import MeredigAttributeGenerator
            ma = MeredigAttributeGenerator()
            df = ma.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class StoichiometricAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import StoichiometricAttributeGenerator
            sg = StoichiometricAttributeGenerator()
            if 'p_norms' in self.parameters:
                if self.parameters['p_norms']:
                    sg.add_p_norms(self.parameters['p_norms'])
            df = sg.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class ValenceShellAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import ValenceShellAttributeGenerator
            vs = ValenceShellAttributeGenerator()
            df = vs.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class YangOmegaAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import YangOmegaAttributeGenerator
            yo = YangOmegaAttributeGenerator()
            df = yo.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class APRDFAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import APRDFAttributeGenerator
            aprdf = APRDFAttributeGenerator()
            if 'cut_off_distance' in self.parameters:
                if self.parameters['cut_off_distance'] != 10.0:
                    aprdf.set_cut_off_distance(
                        self.parameters['cut_off_distance'])
            if 'num_points' in self.parameters:
                if self.parameters['num_points'] != 6:
                    aprdf.set_num_points(
                        self.parameters['num_points'])
            if 'smooth_parameter' in self.parameters:
                if self.parameters['smooth_parameter'] != 4.0:
                    aprdf.set_smoothing_parameter(
                        self.parameters['smooth_parameter'])
            aprdf.add_elemental_properties(self.parameters['elemental_properties'])
            df = aprdf.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class ChemicalOrderingAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import ChemicalOrderingAttributeGenerator
            co = ChemicalOrderingAttributeGenerator()
            if 'shells' in self.parameters:
                if self.parameters['shells']:
                    co.set_shells(
                        self.parameters['shells'])
            if 'weighted' in self.parameters:
                if self.parameters['weighted']:
                    co.set_weighted(
                        self.parameters['weighted'])

            df = co.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class CoordinationNumberAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import CoordinationNumberAttributeGenerator
            cn = CoordinationNumberAttributeGenerator()
            df = cn.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class CoulombMatrixAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import CoulombMatrixAttributeGenerator
            cm = CoulombMatrixAttributeGenerator()
            if 'n_eigenvalues' in self.parameters:
                if self.parameters['n_eigenvalues']:
                    cm.set_n_eigenvalues(
                        self.parameters['n_eigenvalues'])

            df = cm.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class EffectiveCoordinationNumberAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import EffectiveCoordinationNumberAttributeGenerator
            ecn = EffectiveCoordinationNumberAttributeGenerator()
            df = ecn.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class LatticeSimilarityAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import LatticeSimilarityAttributeGenerator
            ls = LatticeSimilarityAttributeGenerator()
            df = ls.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class LocalPropertyDifferenceAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import LocalPropertyDifferenceAttributeGenerator
            lpd = LocalPropertyDifferenceAttributeGenerator()
            if 'shells' in self.parameters:
                if not (len(self.parameters['shells']) == 1 and
                            self.parameters['shells'][0] == 1):
                    lpd.add_shells(self.parameters['shells'])
            lpd.add_elemental_properties(
                self.parameters['elemental_properties'])
            df = lpd.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class LocalPropertyVarianceAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import LocalPropertyVarianceAttributeGenerator
            lpv = LocalPropertyVarianceAttributeGenerator()
            if 'shells' in self.parameters:
                if not (len(self.parameters['shells']) == 1 and
                                self.parameters['shells'][0] == 1):
                    lpv.add_shells(self.parameters['shells'])
            lpv.add_elemental_properties(
                self.parameters['elemental_properties'])
            df = lpv.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class PackingEfficiencyAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import PackingEfficiencyAttributeGenerator
            pe = PackingEfficiencyAttributeGenerator()
            df = pe.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class PRDFAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import PRDFAttributeGenerator
            prdf = PRDFAttributeGenerator()
            if 'cut_off_distance' in self.parameters:
                if self.parameters['cut_off_distance'] != 10.0:
                    prdf.set_cut_off_distance(
                        self.parameters['cut_off_distance'])
            if 'num_points' in self.parameters:
                if self.parameters['num_points'] != 20:
                    prdf.set_num_points(
                        self.parameters['num_points'])
            prdf.set_elements(entries)
            df = prdf.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class StructuralHeterogeneityAttributeGenerator(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('entries', req=True)
        entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import StructuralHeterogeneityAttributeGenerator
            sh = StructuralHeterogeneityAttributeGenerator()
            df = sh.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class CompositionEntry(BASE):
    def fit(self):
        # self.paramFROMinput()
        # self.required('entries', req=True)
        # entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import CompositionEntry
            composition_list = CompositionEntry.import_composition_list(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'entries':
                self.set_value(token, composition_list)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs

class CrystalStructureEntry(BASE):
    def fit(self):
        # self.paramFROMinput()
        # self.required('entries', req=True)
        # entries = self.inputs['entries'].value
        try:
            from chemml.chem.magpie_python import CrystalStructureEntry
            structures_list = CrystalStructureEntry.import_structures_list(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if
                 edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'entries':
                self.set_value(token, structures_list)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs


# data manipulation

class MissingValues(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # process
        try:
            from chemml.preprocessing import MissingValues
            if method == None:
                model = MissingValues(**self.parameters)
            elif method == 'fit_transform':
                model = MissingValues(**self.parameters)
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                df = model.fit_transform(df)
            elif method == 'transform':
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                self.required('api', req=True)
                model = self.inputs['api'].value
                df = model.transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
        # step7: delete all inputs from memory
        del self.inputs

class ConstantColumns(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # step4: import module and make APIs
        try:
            from chemml.preprocessing import ConstantColumns
            if method == None:
                model = ConstantColumns()
            elif method == 'fit_transform':
                model = ConstantColumns()
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                df = model.fit_transform(df)
            elif method == 'transform':
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                self.required('api', req=True)
                model = self.inputs['api'].value
                df = model.transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'removed_columns_':
                removed_columns_ = pd.DataFrame(model.removed_columns_)
                self.set_value(token, removed_columns_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class Outliers(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        method = self.parameters.pop('func_method')
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # step4: import module and make APIs
        try:
            from chemml.preprocessing import Outliers
            if method == None:
                model = Outliers(**self.parameters)
            elif method == 'fit_transform':
                model = Outliers(**self.parameters)
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                df_out = model.fit_transform(df)
            elif method == 'transform':
                self.required('df', req=True)
                df = self.inputs['df'].value
                df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')
                self.required('api', req=True)
                model = self.inputs['api'].value
                df_out = model.transform(df)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                self.set_value(token, df_out)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'removed_rows_':
                removed_columns_ = pd.DataFrame(model.removed_columns_)
                self.set_value(token, removed_columns_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


# class Trimmer(BASE):
#     def legal_IO(self):
#         self.legal_inputs = {'dfx': None, 'dfy': None}
#         self.legal_outputs = {'dfx': None, 'dfy': None, 'api': None}
#         requirements = ['chemml', 'pandas']
#         self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]
#
#     def fit(self):
#         from chemml.initializtion import Trimmer
#         dfx = self.type_check('dfx', chemml_type='dfx', req=True, py_type=pd.DataFrame)
#         dfy = self.type_check('dfy', chemml_type='dfy', req=True, py_type=pd.DataFrame)
#         try:
#             model = Trimmer(**self.parameters)
#             dfx, dfy = model.fit_transform(dfx,dfy)
#         except Exception as err:
#             msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
#                 err).__name__ + ': ' + str(err.message)
#             raise TypeError(msg)
#         order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
#         for token in set(order):
#             if token == 'dfx':
#                 self.Base.send[(self.iblock, token)] = [dfx, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             elif token == 'dfy':
#                 self.Base.send[(self.iblock, token)] = [dfy, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             elif token == 'api':
#                 self.Base.send[(self.iblock, token)] = [model, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             else:
#                 msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
#                     self.iblock + 1, self.Task, token)
#                 raise NameError(msg)
#         del self.legal_inputs

# class Uniformer(BASE):
#     def legal_IO(self):
#         self.legal_inputs = {'dfx': None, 'dfy': None}
#         self.legal_outputs = {'dfx': None, 'dfy': None, 'api': None}
#         requirements = ['chemml', 'pandas']
#         self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]
#
#     def fit(self):
#         from chemml.initializtion import Uniformer
#         dfx = self.type_check('dfx', chemml_type='dfx', req=True, py_type=pd.DataFrame)
#         dfy = self.type_check('dfy', chemml_type='dfy', req=True, py_type=pd.DataFrame)
#         try:
#             model = Uniformer(**self.parameters)
#             dfx, dfy = model.fit_transform(dfx, dfy)
#         except Exception as err:
#             msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
#                 err).__name__ + ': ' + str(err.message)
#             raise TypeError(msg)
#         order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
#         for token in set(order):
#             if token == 'dfx':
#                 self.Base.send[(self.iblock, token)] = [dfx, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             elif token == 'dfy':
#                 self.Base.send[(self.iblock, token)] = [dfy, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             elif token == 'api':
#                 self.Base.send[(self.iblock, token)] = [model, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             else:
#                 msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
#                     self.iblock + 1, self.Task, token)
#                 raise NameError(msg)
#         del self.legal_inputs


# data manipulation

class Split(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', req=True)
        df = self.inputs['df'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        df, _ = self.data_check('df', df, ndim=2, n0=None, n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from chemml.initialization import Split
            split = Split(**self.parameters)
            df1, df2 = split.fit(df)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ str(err.message)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df1':
                self.set_value(token, df1)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'df2':
                self.set_value(token, df2)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs



##################################################################### 3 Define Model

# Regression

class MLP(BASE):
    def fit(self):
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None

        # step4: import module and make APIs
        try:
            from chemml.nn.keras import MLP
            if method == None:
                model = MLP(**self.parameters)
            elif method == 'fit':
                model = MLP(**self.parameters)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
                self.required('dfy', req=True)
                dfy = self.inputs['dfy'].value
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=dfx.shape[0], n1=None, format_out='ar')
                model.fit(dfx,dfy)
            elif method == 'predict':
                self.required('dfx', req=True)
                self.required('api', req=True)
                dfx = self.inputs['dfx'].value
                dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
                model = self.inputs['api'].value
                dfy_predict = model.predict(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                    err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'dfy_predict':
                self.set_value(token, pd.DataFrame(dfy_predict))
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class MLP_sklearn(BASE):
    def fit(self):
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = None

        # step4: import module and make APIs
        try:
            from chemml.nn.keras import MLP_sklearn
            if method == None:
                model = MLP_sklearn(**self.parameters)
            elif method == 'fit':
                model = MLP_sklearn(**self.parameters)
                self.required('dfx', req=True)
                dfx = self.inputs['dfx'].value
                dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
                self.required('dfy', req=True)
                dfy = self.inputs['dfy'].value
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=dfx.shape[0], n1=None, format_out='ar')
                model.fit(dfx,dfy)
            elif method == 'predict':
                self.required('dfx', req=True)
                self.required('api', req=True)
                dfx = self.inputs['dfx'].value
                dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
                model = self.inputs['api'].value
                dfy_predict = model.predict(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                    err).__name__ + ': ' + str(err.message)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'api':
                self.set_value(token, model)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'dfy_predict':
                self.set_value(token, pd.DataFrame(dfy_predict))
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

# class mlp_hogwild(BASE):
#     def fit(self):
#         # step1: check inputs
#         self.required('dfx_train', req=True)
#         dfx_train = self.inputs['dfx_train'].value
#         self.required('dfy_train', req=True)
#         dfy_train = self.inputs['dfy_train'].value
#
#         # step2: assign inputs to parameters if necessary (param = @token)
#         self.paramFROMinput()
#         method = self.parameters.pop('func_method')
#
#         # step3: check the dimension of input data frame
#         dfx_train, _ = self.data_check('dfx_train', dfx_train, ndim=2, n0=None, n1=None, format_out='ar')
#         dfy_train, _ = self.data_check('dfy_train', dfy_train, ndim=2, n0=dfx_train.shape[0], n1=None, format_out='ar')
#
#         # step4: import module and make APIs
#         try:
#             from chemml.nn import mlp_hogwild
#             if method is None:
#                 model = mlp_hogwild(**self.parameters)
#             elif method == 'fit':
#                 model = mlp_hogwild(**self.parameters)
#                 self.required('dfx', req=True)
#                 dfx = self.inputs['dfx'].value
#                 dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
#                 self.required('dfy', req=True)
#                 dfy = self.inputs['dfy'].value
#                 dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=dfx.shape[0], n1=None, format_out='ar')
#                 model.fit(dfx,dfy)
#             elif method == 'predict':
#                 self.required('dfx', req=True)
#                 self.required('api', req=True)
#                 dfx = self.inputs['dfx'].value
#                 dfx, _ = self.data_check('dfx', dfx, ndim=2, n0=None, n1=None, format_out='ar')
#                 api = self.inputs['api'].value
#                 dfy_predict = api.predict(dfx)
#         except Exception as err:
#             msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
#                     err).__name__ + ': ' + str(err.message)
#             raise TypeError(msg)
#
#         # step5: process
#         # step6: send out
#         order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
#         for token in set(order):
#             if token not in self.outputs:
#                 msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
#                 raise NameError(msg)
#             elif token == 'api':
#                 self.set_value(token, model)
#                 self.outputs[token].count = order.count(token)
#                 self.Base.send[(self.iblock, token)] = self.outputs[token]
#             elif token == 'dfy_predict':
#                 self.set_value(token, dfy_predict)
#                 self.outputs[token].count = order.count(token)
#                 self.Base.send[(self.iblock, token)] = self.outputs[token]
#
#         # step7: delete all inputs from memory
#         del self.inputs
#
# class mlp_dsgd(BASE):
#     # must be run with slurm script
#     # Todo: first fix the slurm script function at chemml.initialization
#     # Todo: then embede the slurm commands in this class to run the slurm script
#     # Todo: or make the slurm script in this function too
#     def legal_IO(self):
#         self.legal_inputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
#         self.legal_outputs = {'dfy_train_pred': None, 'model': None}
#         requirements = ['chemml', 'pandas']
#         self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]
#
#     def fit(self):
#         from chemml.nn import nn_dsgd
#         chemml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
#         self.Base.chemml_type['regressor'].append(chemml_type)
#         dfx_train = self.type_check('dfx_train', chemml_type='df', req=True, py_type=pd.DataFrame).values
#         dfx_test = self.type_check('dfx_test', chemml_type='df', req=True, py_type=pd.DataFrame).values
#         dfy_train = self.type_check('dfy_train', chemml_type='df', req=True, py_type=pd.DataFrame)
#         dfy_header = dfy_train.columns
#         dfy_train = dfy_train.values
#         dfy_test = self.type_check('dfy_test', chemml_type='df', req=True, py_type=pd.DataFrame).values
#
#         try:
#             model = nn_psgd.train(dfx_train,dfx_test,dfy_train,dfy_test,**self.parameters)
#         except Exception as err:
#             msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
#                 err).__name__ + ': ' + str(err.message)
#             raise TypeError(msg)
#
#         dfy_pred = nn_psgd.output(dfx_train,model)
#         dfy_pred = pd.DataFrame(dfy_pred, columns=dfy_header)
#
#         order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
#         for token in set(order):
#             if token == 'model':
#                 self.Base.send[(self.iblock, token)] = [model, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             elif token == 'dfy_train_pred':
#                 self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token),
#                                                         (self.iblock, token, self.Host, self.Function)]
#             else:
#                 msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.Task, token)
#                 raise NameError(msg)
#         del self.legal_inputs

##################################################################### 6 Mix

class kfold_pool(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None, 'kfold':None, 'model':None, 'evaluator':None}
        self.legal_outputs = {'evaluation_results_': None, 'best_model_': None}
        requirements = ['scikit_learn']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step4: import module and make APIs
        try:
            self._reg_evaluation_params()
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ str(err.message)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0]==self.iblock]
        for token in set(order):
            if token == 'evaluator':
                self.Base.send[(self.iblock, token)] = [self.evaluator, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'evaluation_results_':
                # step1: check inputs
                dfy, dfy_info = self.input_check('dfy', req=True, py_type=pd.DataFrame)
                dfy_pred, dfy_pred_info = self.input_check('dfy_pred', req=True, py_type=pd.DataFrame)

                # step3: check the dimension of input data frame
                dfy, _ = self.data_check('dfy', dfy, ndim=2, n0=None, n1=None, format_out='df')
                dfy_pred, _ = self.data_check('dfy_pred', dfy_pred, ndim=2, n0=dfy.shape[0], n1=None, format_out='df')

                self._reg_evaluate(dfy, dfy_pred, self.evaluator)
                evaluation_results_ = self.results
                self.Base.send[(self.iblock, token)] = [pd.DataFrame(evaluation_results_), order.count(token),
                                                        (self.iblock,token,self.Host,self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock+1, self.Task, token)
                raise NameError(msg)

        #step7: delete all inputs from memory
        del self.legal_inputs

##################################################################### 8 Store

class SaveFile(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df', req=True)
        df = self.inputs['df'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            # from chemml.initialization import SaveFile
            # model = SaveFile(**self.parameters)
            # model.fit(df, self.Base.output_directory)
            df.to_csv(self.Base.output_directory)
            
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ str(err)
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'filepath':
                self.set_value(token, model.file_path)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class GA(BASE):
    def fit(self):
        self.paramFROMinput()
        # txt_files = list(self.parameters.keys())[:4]      #keys = evaluate, space, error_metric, objecive
        # ga_eval = self.parameters[txt_files[0]]
        # space = self.parameters[txt_files[1]]
        # error_metric = self.parameters[txt_files[2]]
        # single_obj = self.parameters[txt_files[3]]
        
        # for key in self.parameters:
        #     print(key," : ", self.parameters[key])
        
            
        try:
            from chemml.optimization import GeneticAlgorithm
            from sklearn.neural_network import MLPRegressor
            from sklearn.model_selection import train_test_split
            from datetime import date, datetime


            import numpy as np
            
            # for files in list(self.parameters.keys())[:4]:      #keys = evaluate, space, error_metric, objecive
            #     with open(self.parameters[files],'r') as f:
            #         contents = f.read()
            #         # print("files: ", files, "contents: ", contents)
            #     files = compile(contents, "<string>", "exec")
            
            for key in list(self.parameters.keys()):
                if key == 'fitness':
                    final_fit=[]
                    fitness = str(self.parameters[key])[-6:-3]
                    fitness = fitness[0]+fitness[1]+fitness[2]
                    final_fit.append(fitness)
                    final_fit = tuple(final_fit)
                    # print("fitness: ",final_fit)
                    # print("type(fitness): ", type(final_fit))
                elif key == 'pop_size':
                    pop_size = self.parameters[key]
                elif key == 'crossover_size':
                    crossover_size = self.parameters[key]
                elif key == 'mutation_size':
                    mutation_size = self.parameters[key]
                elif key == 'n_splits':
                    global n_splits
                    n_splits = self.parameters[key]
                elif key == 'crossover_type':
                    crossover_type = self.parameters[key]
                elif key == 'mutation_prob':
                    mutation_prob = self.parameters[key]
                elif key == 'initial_population':
                    initial_population = self.parameters[key]
                elif key == 'n_generations':
                    n_generations = self.parameters[key]
                elif key == 'early_stopping':
                    early_stopping = self.parameters[key]
                elif key == 'init_ratio':
                   init_ratio = self.parameters[key]
                elif key == 'crossover_ratio':
                    crossover_ratio = self.parameters[key]
                elif key == 'algorithm':
                    global algorithm
                    algorithm = self.parameters[key]

            #default in chemml.optimizaiton.geneticalgorithm
            if 'early_stopping' not in list(self.parameters.keys())[4:]:
                early_stopping = 10

            with open(self.parameters['error_metric'],'r') as f:
                contents = f.read()
                    # print("files: ", files, "contents: ", contents)
                code = compile(contents, "<string>", "exec")
                loc = {}
                try:
                    exec(code,globals(), loc)
                    global mae
                    mae = loc['error_metric']
                except:
                    print("Something wrong with the code...")
                    print("error_metric: ", mae)
                    print("type(error_metric): ",type(mae))

            with open(self.parameters['space'],'r') as f:
                contents = f.read()
                code = compile(contents, "<string>", "exec")
                loc = {}
                try:
                    exec(code,globals(), loc)
                    space = loc['space']
                except:
                    print("Something wrong with the code...")
                    print("Space: ", space)
                    print("type(space): ",type(space))

                            
            with open(self.parameters['single_obj'],'r') as f:
                contents = f.read()
                    # print("files: ", files, "contents: ", contents)
                code = compile(contents, "<string>", "exec")
                loc = {}
                try:
                    exec(code,globals(), loc)
                    global single_obj
                    single_obj = loc['single_obj']
                except:
                    print("Something wrong with the code...")
                    print("single_obj: ", single_obj)
                    print("type(single_obj): ",type(single_obj))
                
                # print("single_obj: ", single_obj)
                # print("type(single_obj): ",type(single_obj))

            with open(self.parameters['evaluate'],'r') as f:
                contents = f.read()
                    # print("files: ", files, "contents: ", contents)
                code = compile(contents, "<string>", "exec")
                loc = {}
                try:
                    exec(code,globals(), loc)
                    ga_eval = loc['ga_eval']
                except:
                    print("Something wrong with the code...")
                    print("ga_eval: ", ga_eval)
                    print("type(ga_eval): ",type(ga_eval))
                # print("ga_eval: ", ga_eval)
                # print("type(ga_eval): ",type(ga_eval))
            
            with open(self.parameters['test_hyperparameters'],'r') as f:
                contents = f.read()
                    # print("files: ", files, "contents: ", contents)
                code = compile(contents, "<string>", "exec")
                loc = {}
                try:
                    exec(code,globals(), loc)
                    test_hyp = loc['test_hyp']
                except:
                    print("Something wrong with the code...")
                    print("test_hyperparameters: ", test_hyperparameters)
                    print("type(test_hyperparameters): ",type(test_hyperparameters))


            ##### GA happening here#########
            def ga_mlpregressor(x_train, y_train, x_test, y_test, al=algorithm,n_splits=n_splits,n_generations=n_generations,early_stopping=early_stopping): 
                global X 
                global Y
                X=x_train
                Y=y_train 
                print("Hyperparameter optimization is a time consuming process - do not shutdown Kernel....\n")
                print('Total GA search iterations = ', n_generations*pop_size)
                gann = GeneticAlgorithm(evaluate=ga_eval, space=space, fitness=final_fit, pop_size = pop_size, crossover_size=crossover_size, mutation_size=mutation_size, algorithm=al)
                global MLPRegressor
                from sklearn.neural_network import MLPRegressor
                global KFold
                from sklearn.model_selection import KFold
                import warnings
                warnings.filterwarnings("ignore")  
                best_ind_df, best_individual = gann.search(n_generations=n_generations, early_stopping=early_stopping)                     # set pop_size<30, n_generations*pop_size = no. of times GA runs                      
                print("GeneticAlgorithm - complete!")
 
                all_items = list(gann.fitness_dict.items())
                all_items_df = pd.DataFrame(all_items, columns=['hyperparameters', 'Accuracy_score'])
                print("\n\ngenetic algorithm results for each generation: \n", best_ind_df, "\n\nbest particle: ", best_individual, "\n")
                print("Calculating accuracy on test data....")
                l = [best_individual['neurons1'], best_individual['neurons2'], best_individual['neurons3']]
                layers = [i for i in l if i != 0]
                ga_mlp = MLPRegressor(alpha=np.exp(best_individual['alpha']), activation=best_individual['activation'], hidden_layer_sizes=tuple(layers), learning_rate='invscaling', max_iter=20, early_stopping=True)
                ga_accuracy_test = test_hyp(mlp=ga_mlp, x=X, y=Y, xtest=x_test, ytest=y_test)
                print("\n\nTest set error_metric (default = MAE) for the best GA hyperparameter: ", ga_accuracy_test, "\n")
                return all_items_df , best_ind_df


            #Read data here
            self.required('dfx_train', req=True)
            dfx_train= self.inputs['dfx_train'].value
            self.required('dfy_train', req=True)
            dfy_train= self.inputs['dfy_train'].value
            self.required('dfx_test', req=True)
            dfx_test= self.inputs['dfx_test'].value
            self.required('dfy_test', req=True)
            dfy_test= self.inputs['dfy_test'].value


            # dfx_train = self.inputs['dfx_train'].value
            # dfy_train = self.inputs['dfy_train'].value
            # dfx_test = self.inputs['dfx_test'].value
            # dfy_test = self.inputs['dfy_test'].value
        
            # type of ML model defined here
            for key in list(self.parameters.keys()):
                if key == 'ml_model':
                    ml_model = self.parameters[key]
                    if ml_model == 'MLPRegressor':
                        
                        best_ind_df, best_individual = ga_mlpregressor(x_train=dfx_train, y_train=dfy_train, x_test=dfx_test,y_test=dfy_test, al = algorithm, n_splits=n_splits, n_generations=n_generations, early_stopping=early_stopping)
                        # print(all_items_df)


            os.remove("tmp.txt")            #remove tmp file to count  umber of GA iterations
            os.remove("GA.txt")             #remove file with all GA iterations

            # now = datetime.now()            #to save with current date and time
            # dt_string = now.strftime("%m-%d-%Y %H-%M-%S")
            # all_items_df.to_csv('best_ind_df' + str(dt_string) + '.csv')
            # best_ind_df.to_csv('best_individual' + str(dt_string) + '.csv')
            # print("GA DONEE!!!")

        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ str(err)
            raise TypeError(msg)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'best_ind_df':
                self.set_value(token, best_ind_df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'best_individual':
                self.set_value(token, best_individual)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs