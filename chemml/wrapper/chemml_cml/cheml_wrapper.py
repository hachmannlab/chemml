from __future__ import print_function

import pandas as pd
import numpy as np
import os
import warnings

from ..base import BASE

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
        print( 'from:', self.parameters['from_format'])
        print( 'to:', self.parameters['to_format'])
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
        for token in inputs:
            code = compile("%s = self.inputs['%s'].value"%(token,token), "<string>", "exec")
            exec(code)
        for line in sorted(self.parameters.keys()):
            code = compile(self.parameters[line], "<string>", "exec")
            exec(code)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'ov1':
                self.set_value(token, ov1)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov2':
                self.set_value(token, ov2)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov3':
                self.set_value(token, ov3)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov4':
                self.set_value(token, ov4)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov5':
                self.set_value(token, ov5)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'ov6':
                self.set_value(token, ov6)
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
            model = RDKitFingerprint(**self.parameters)
            model.MolfromFile(molfile,path,*arguments)
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
                self.set_value(token, model.Fingerprint())
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'removed_rows':
                self.set_value(token, model.removed_rows)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

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
            if method is None:
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
            if method is None:
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
            if method is None:
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
            if method is None:
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
            if method is None:
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

##################################################################### 7 Search
class GA_DEAP(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        # func_method default value
        if 'func_method' in self.parameters:
            method = self.parameters.pop('func_method')
        else:
            method = 'algorithm_1'
        if 'init_pop_frac' in self.parameters:
            init_pop_frac = self.parameters.pop('init_pop_frac')
        else:
            init_pop_frac = 0.35
        if 'crossover_pop_frac' in self.parameters:
            crossover_pop_frac = self.parameters.pop('crossover_pop_frac')
        else:
            crossover_pop_frac = 0.35

        # step4: import module and make APIs
        try:
            from chemml.search import GA_DEAP
            model = GA_DEAP(**self.parameters)
            model.fit()
            if method == 'algorithm_1':
                best_ind_df, best_individual = model.algorithm_1()
            elif method == 'algorithm_2':
                best_ind_df, best_individual = model.algorithm_2(init_pop_frac=init_pop_frac,
                                                                 crossover_pop_frac=crossover_pop_frac)
            elif method == 'algorithm_3':
                best_ind_df, best_individual = model.algorithm_3()
            elif method == 'algorithm_4':
                best_ind_df, best_individual = model.algorithm_4(crossover_pop_frac=crossover_pop_frac)
            else:
                msg = "@Task #%i(%s): the func_method is not valid. try one of these methods: ('algorithm_1', " \
                      "'algorithm_2', 'algorithm_3', 'algorithm_4')"% (self.iblock + 1, self.Task)
                raise Exception(msg)
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
            elif token == 'best_ind_df':
                self.set_value(token, best_ind_df)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'best_individual':
                removed_columns_ = pd.DataFrame(list(best_individual))
                self.set_value(token, removed_columns_)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs



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
            from chemeco.initialization import SaveFile
            model = SaveFile(**self.parameters)
            model.fit(df, self.Base.output_directory)
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
            elif token == 'filepath':
                self.set_value(token, model.file_path)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs


