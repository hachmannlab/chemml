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
            from cheml.initialization import XYZreader
            model = XYZreader(**self.parameters)
            molecules = model.read()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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
            from cheml.datasets import load_cep_homo
            output = load_cep_homo(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token not in self.outputs:
                msg = "@Task #%i(%s): not a valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
            elif token == 'df':
                if self.parameters['return_X_y']:
                    msg = 'parameter return_X_y, returns X and y separately: return_X_y=True. The output df is a list of both X and y'
                    warnings.warn(msg)
                self.set_value(token, output)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'smiles':
                if not self.parameters['return_X_y']:
                    msg = 'parameter return_X_y, returns X and y separately: return_X_y=False. The output smiles is a dataframe of X and y'
                    warnings.warn(msg)
                    self.set_value(token, output)
                else:
                    self.set_value(token, output[0])
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]
            elif token == 'dfy':
                if not self.parameters['return_X_y']:
                    msg = 'parameter return_X_y, returns X and y separately: return_X_y=False. The output dfy is a dataframe of X and y'
                    warnings.warn(msg)
                    self.set_value(token, output)
                else:
                    self.set_value(token, output[1])
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        del self.inputs


class ConvertFile(BASE):
    def fit(self):
        self.paramFROMinput()
        # self.required('file_path',req=True)
        # file_path=self.inputs['file_path'].value
        print 'from:', self.parameters['from_format']
        print 'to:', self.parameters['to_format']
        # if 'file_path' not in self.parameters and '@' not in file_path:
            # self.parameters['file_path']=file_path
        try:
            from cheml.initialization import ConvertFile
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
            # Script

class PyScript(BASE):
    def fit(self):
        # step1: check inputs
        inputs = [token for token in self.inputs if self.inputs[token].value is not None]
        for token in inputs:
            code = compile("%s = self.inputs['%s'].value"%(token,token), "<string>", "exec")
            exec code
        for line in sorted(self.parameters.keys()):
            code = compile(self.parameters[line], "<string>", "exec")
            exec code
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
            from cheml.chem import RDKitFingerprint
            model = RDKitFingerprint(**self.parameters)
            model.MolfromFile(molfile,path,*arguments)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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
        print 'molfile:', self.parameters['molFile']

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
            from cheml.chem import Dragon
            model = Dragon(**self.parameters)
            model.script_wizard(script, output_directory)
            model.run()
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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

class Coulomb_Matrix(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('molecules', req=True)
        molecules = self.inputs['molecules'].value
        try:
            from cheml.chem import Coulomb_Matrix
            model = Coulomb_Matrix(**self.parameters)
            df = model.represent(molecules)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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

class Bag_of_Bonds(BASE):
    def fit(self):
        self.paramFROMinput()
        self.required('molecules', req=True)
        molecules = self.inputs['molecules'].value
        try:
            from cheml.chem import Bag_of_Bonds
            model = Bag_of_Bonds(**self.parameters)
            df = model.represent(molecules)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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
            from cheml.chem import DistanceMatrix
            model = DistanceMatrix(**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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
            from cheml.chem import APEAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import ChargeDependentAttributeGenerator
            cd = ChargeDependentAttributeGenerator()
            df = cd.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import ElementalPropertyAttributeGenerator
            ep = ElementalPropertyAttributeGenerator()
            if 'elemental_properties' in self.parameters:
                if self.parameters['elemental_properties']:
                    ep.add_elemental_properties(self.parameters['elemental_properties'])
            df = ep.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import ElementFractionAttributeGenerator
            ef = ElementFractionAttributeGenerator()
            df = ef.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import ElementPairPropertyAttributeGenerator
            epp = ElementPairPropertyAttributeGenerator()
            if 'elemental_pair_properties' in self.parameters:
                if self.parameters['elemental_pair_properties']:
                    epp.add_elemental_pair_properties(
                        self.parameters['elemental_pair_properties'])
            df = epp.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import GCLPAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import IonicCompoundProximityAttributeGenerator
            icp = IonicCompoundProximityAttributeGenerator()
            if 'max_formula_unit' in self.parameters and self.parameters[
                'max_formula_unit'] != 14:
                icp.set_max_formula_unit(self.parameters['max_formula_unit'])
            df = icp.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import IonicityAttributeGenerator
            ig = IonicityAttributeGenerator()
            df = ig.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import MeredigAttributeGenerator
            ma = MeredigAttributeGenerator()
            df = ma.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import StoichiometricAttributeGenerator
            sg = StoichiometricAttributeGenerator()
            if 'p_norms' in self.parameters:
                if self.parameters['p_norms']:
                    sg.add_p_norms(self.parameters['p_norms'])
            df = sg.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import ValenceShellAttributeGenerator
            vs = ValenceShellAttributeGenerator()
            df = vs.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import YangOmegaAttributeGenerator
            yo = YangOmegaAttributeGenerator()
            df = yo.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import APRDFAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import ChemicalOrderingAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import CoordinationNumberAttributeGenerator
            cn = CoordinationNumberAttributeGenerator()
            df = cn.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import CoulombMatrixAttributeGenerator
            cm = CoulombMatrixAttributeGenerator()
            if 'n_eigenvalues' in self.parameters:
                if self.parameters['n_eigenvalues']:
                    cm.set_n_eigenvalues(
                        self.parameters['n_eigenvalues'])

            df = cm.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import EffectiveCoordinationNumberAttributeGenerator
            ecn = EffectiveCoordinationNumberAttributeGenerator()
            df = ecn.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import LatticeSimilarityAttributeGenerator
            ls = LatticeSimilarityAttributeGenerator()
            df = ls.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import LocalPropertyDifferenceAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import LocalPropertyVarianceAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import PackingEfficiencyAttributeGenerator
            pe = PackingEfficiencyAttributeGenerator()
            df = pe.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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
            from cheml.chem import PRDFAttributeGenerator
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
                err).__name__ + ': ' + err.message
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
            from cheml.chem import StructuralHeterogeneityAttributeGenerator
            sh = StructuralHeterogeneityAttributeGenerator()
            df = sh.generate_features(entries)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
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

# Preprocessor

class MissingValues(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        method = self.parameters.pop('func_method')
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # process
        try:
            from cheml.preprocessing import missing_values
            if method is None:
                model = missing_values(**self.parameters)
            elif method == 'fit_transform':
                model = missing_values(**self.parameters)
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
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(err).__name__ + ': ' + err.message
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

class Trimmer(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'dfx': None, 'dfy': None, 'api': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initializtion import Trimmer
        dfx = self.type_check('dfx', cheml_type='dfx', req=True, py_type=pd.DataFrame)
        dfy = self.lltype_check('dfy', cheml_type='dfy', req=True,
                               py_type=pd.DataFrame)
        try:
            model = Trimmer(**self.parameters)
            dfx, dfy = model.fit_transform(dfx,dfy)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'dfy':
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
        del self.legal_inputs

class Uniformer(BASE):
    def legal_IO(self):
        self.legal_inputs = {'dfx': None, 'dfy': None}
        self.legal_outputs = {'dfx': None, 'dfy': None, 'api': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.initializtion import Uniformer
        dfx = self.type_check('dfx', cheml_type='dfx', req=True, py_type=pd.DataFrame)
        dfy = self.type_check('dfy', cheml_type='dfy', req=True, py_type=pd.DataFrame)
        try:
            model = Uniformer(**self.parameters)
            dfx, dfy = model.fit_transform(dfx, dfy)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'dfx':
                self.Base.send[(self.iblock, token)] = [dfx, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'dfy':
                self.Base.send[(self.iblock, token)] = [dfy, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'api':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (
                    self.iblock + 1, self.Task, token)
                raise NameError(msg)
        del self.legal_inputs



# Basic Operators

class Merge(BASE):
    def fit(self):
        # step1: check inputs
        self.required('df1', req=True)
        df1 = self.inputs['df1'].value
        self.required('df2', req=True)
        df2 = self.inputs['df2'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        # self.paramFROMinput()

        # step3: check the dimension of input data frame
        df1, _ = self.data_check('df1', df1, ndim=2, n0=None, n1=None, format_out='df')
        df2, _ = self.data_check('df2', df2, ndim=2, n0=df1.shape[0], n1=None, format_out='df')

        # step4: import module and make APIs
        try:
            from cheml.initialization import Merge
            df = Merge(df1, df2)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
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

        # step7: delete all inputs from memory
        del self.inputs

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
            from cheml.initialization import Split
            split = Split(**self.parameters)
            df1, df2 = split.fit(df)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
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

class Constant(BASE):
    def fit(self):
        # parameters
        self.paramFROMinput()
        method = self.parameters.pop('func_method')
        # get df value only in case method is None, but output df is requested
        df = self.inputs['df'].value

        # step4: import module and make APIs
        try:
            from cheml.preprocessing import Constant
            if method is None:
                model = Constant()
            elif method == 'fit_transform':
                model = Constant()
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
                err).__name__ + ': ' + err.message
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


##################################################################### 3 Define Model

# Regression

class mlp_hogwild(BASE):
    def fit(self):
        # step1: check inputs
        self.required('dfx_train', req=True)
        dfx_train = self.inputs['dfx_train'].value
        self.required('dfy_train', req=True)
        dfy_train = self.inputs['dfy_train'].value

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()
        method = self.parameters.pop('func_method')

        # step3: check the dimension of input data frame
        dfx_train, _ = self.data_check('dfx_train', dfx_train, ndim=2, n0=None, n1=None, format_out='ar')
        dfy_train, _ = self.data_check('dfy_train', dfy_train, ndim=2, n0=dfx_train.shape[0], n1=None, format_out='ar')

        # step4: import module and make APIs
        try:
            from cheml.nn import mlp_hogwild
            if method is None:
                model = mlp_hogwild(**self.parameters)
            elif method == 'fit':
                model = mlp_hogwild(**self.parameters)
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
                api = self.inputs['api'].value
                dfy_predict = api.predict(dfx)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                    err).__name__ + ': ' + err.message
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
                self.set_value(token, dfy_predict)
                self.outputs[token].count = order.count(token)
                self.Base.send[(self.iblock, token)] = self.outputs[token]

        # step7: delete all inputs from memory
        del self.inputs

class mlp_dsgd(BASE):
    # must be run with slurm script
    # Todo: first fix the slurm script function at cheml.initialization
    # Todo: then embede the slurm commands in this class to run the slurm script
    # Todo: or make the slurm script in this function too
    def legal_IO(self):
        self.legal_inputs = {'dfx_train': None, 'dfx_test': None, 'dfy_train': None, 'dfy_test': None}
        self.legal_outputs = {'dfy_train_pred': None, 'model': None}
        requirements = ['cheml', 'pandas']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        from cheml.nn import nn_dsgd
        cheml_type = "%s_%s" % (self.Base.graph_info[self.iblock][0], self.Base.graph_info[self.iblock][1])
        self.Base.cheml_type['regressor'].append(cheml_type)
        dfx_train = self.type_check('dfx_train', cheml_type='df', req=True, py_type=pd.DataFrame).values
        dfx_test = self.type_check('dfx_test', cheml_type='df', req=True, py_type=pd.DataFrame).values
        dfy_train = self.type_check('dfy_train', cheml_type='df', req=True, py_type=pd.DataFrame)
        dfy_header = dfy_train.columns
        dfy_train = dfy_train.values
        dfy_test = self.type_check('dfy_test', cheml_type='df', req=True, py_type=pd.DataFrame).values

        try:
            model = nn_psgd.train(dfx_train,dfx_test,dfy_train,dfy_test,**self.parameters)
        except Exception as err:
            msg = '@Task #%i(%s): ' % (self.iblock + 1, self.Task) + type(
                err).__name__ + ': ' + err.message
            raise TypeError(msg)

        dfy_pred = nn_psgd.output(dfx_train,model)
        dfy_pred = pd.DataFrame(dfy_pred, columns=dfy_header)

        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'model':
                self.Base.send[(self.iblock, token)] = [model, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            elif token == 'dfy_train_pred':
                self.Base.send[(self.iblock, token)] = [dfy_pred, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): non valid output token '%s'" % (self.iblock + 1, self.Task, token)
                raise NameError(msg)
        del self.legal_inputs

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
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
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

##################################################################### 7 Store

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
            from cheml.initialization import SaveFile
            model = SaveFile(**self.parameters)
            model.fit(df, self.Base.output_directory)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
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


class StoreFile(BASE):
    def legal_IO(self):
        self.legal_inputs = {'input': None}
        self.legal_outputs = {'filepath': None}
        requirements = ['cheml']
        self.Base.requirements += [i for i in requirements if i not in self.Base.requirements]

    def fit(self):
        # step1: check inputs
        input, input_info = self.input_check('input', req=True)

        # step2: assign inputs to parameters if necessary (param = @token)
        self.paramFROMinput()

        # step3: check the dimension of input data frame
        # step4: import module and make APIs
        try:
            from cheml.initialization import StoreFile
            model = SaveFile(**self.parameters)
            model.fit(input, self.Base.output_directory)
        except Exception as err:
            msg = '@Task #%i(%s): '%(self.iblock+1, self.Task) + type(err).__name__ + ': '+ err.message
            raise TypeError(msg)

        # step5: process
        # step6: send out
        order = [edge[1] for edge in self.Base.graph if edge[0] == self.iblock]
        for token in set(order):
            if token == 'filepath':
                val = model.file_path
                self.Base.send[(self.iblock, token)] = [val, order.count(token),
                                                        (self.iblock, token, self.Host, self.Function)]
            else:
                msg = "@Task #%i(%s): asked to send a non valid output token '%s'" % (self.iblock+1,self.Task,token)
                raise NameError(msg)
        # step7: delete all inputs from memory
        del self.legal_inputs

