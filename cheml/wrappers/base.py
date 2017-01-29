class BASE(object):
    """
    Do not instantiate this class
    """
    def __init__(self, Base, parameters, iblock, SuperFunction):
        self.Base = Base
        self.parameters = parameters
        self.iblock = iblock
        self.SuperFunction = SuperFunction

    def run(self):
        self.legal_IO()
        self.receive()
        self.fit()

    def receive(self):
        recv = [edge for edge in self.Base.graph if edge[2] == self.iblock]
        self.Base.graph = tuple([edge for edge in self.Base.graph if edge[2] != self.iblock])
        # check received tokens to (1) be a legal input, and (2) be unique.
        count = {token: 0 for token in self.legal_inputs}
        for edge in recv:
            if edge[3] in self.legal_inputs:
                count[edge[3]] += 1
                if count[edge[3]] > 1:
                    msg = '@Task #%i(%s): only one input per each available input path/token can be received.' % (
                        self.iblock + 1, self.SuperFunction)
                    raise IOError(msg)
            else:
                msg = "@Task #%i(%s): received a non valid input token '%s', sent by function #%i" % (
                    self.iblock + 1, self.SuperFunction, edge[3], edge[0] + 1)
                raise IOError(msg)
        for edge in recv:
            key = edge[0:2]
            if key in self.Base.send:
                if self.Base.send[key][1] > 0:
                    value = self.Base.send[key][0]
                    # TODO: deepcopy is memory consuming
                #     value = copy.deepcopy(self.Base.send[key][0])
                # else:
                #     value = self.Base.send[key][0]
                # Todo: informative token should be a list of (int(edge[0]),edge[1])
                informative_token = (int(edge[0]), edge[1]) + self.Base.graph_info[int(edge[0])]
                self.legal_inputs[edge[3]] = (value, informative_token)
                del value
                self.Base.send[key][1] -= 1
                if self.Base.send[key][1] == 0:
                    del self.Base.send[key]
            else:
                msg = '@Task #%i(%s): broken pipe in token %s - nothing has been sent' % (
                    self.iblock + 1, self.SuperFunction, edge[3])
                raise IOError(msg)
        return self.legal_inputs

    def _error_type(self, token):
        msg = "@Task #%i(%s): The type of input with token '%s' is not valid" \
              % (self.iblock + 1, self.SuperFunction, token)
        raise IOError(msg)

    def type_check(self, token, cheml_type, req=False, py_type=False):
        if isinstance(self.legal_inputs[token], type(None)):
            if req:
                msg = "@Task #%i(%s): The input type with token '%s' is required." \
                      % (self.iblock + 1, self.SuperFunction, token)
                raise IOError(msg)
            else:
                return None
        else:
            slit0 = self.legal_inputs[token][0]
            slit1 = self.legal_inputs[token][1]
            if py_type:
                if not isinstance(slit0, py_type):
                    self._error_type(token)
            # if cheml_type == 'df':
            #     if not slit1[1][0:2] == 'df':
            #         self._error_type(token)
            # elif cheml_type == 'regressor':
            #     if slit1[2] + '_' + slit1[3] not in self.Base.cheml_type['regressor']:
            #         self._error_type(token)
            # elif cheml_type == 'preprocessor':
            #     if slit1[2] + '_' + slit1[3] not in self.Base.cheml_type['preprocessor']:
            #         self._error_type(token)
            # elif cheml_type == 'divider':
            #     if slit1[2] + '_' + slit1[3] not in self.Base.cheml_type['divider']:
            #         self._error_type(token)
            # else:
            #     msg = "@Task #%i(%s): The type of input with token '%s' must be %s not %s" \
            #           % (self.iblock + 1, self.SuperFunction, token, str(py_type), str(type(slit0)))
            #     raise IOError(msg)
            return slit0

class LIBRARY(object):
    """
    Do not instantiate this class
    """
    def references(self,module,function):
        if module == 'sklearn':
            ref_g = "https://github.com/scikit-learn/scikit-learn"
            ref_p = "Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011."
            self.refs['scikit-learn'] = {'github':ref_g, 'paper':ref_p}
        elif module == 'cheml':
            ref_g = "https://mhlari@bitbucket.org/hachmanngroup/cheml.git"
            ref_p = "no publicatin"
            self.refs['ChemML'] =  {'github': ref_g, 'paper': ref_p}
            if function == 'RDKitFingerprint':
                ref_g = "https://github.com/rdkit"
                ref_p = "no publication"
                self.refs['rdkit'] = {'github': ref_g, 'paper': ref_p}
            elif function == 'Dragon':
                ref_g = "http://www.talete.mi.it/products/dragon_description.htm"
                ref_p = "R. Todeschini,  V. Consonni,  R. Mannhold,H.  Kubinyi,  H.  Timmerman,  Handbook  ofMolecular Descriptors, Wiley-VCH, 2000."
                self.refs['Dragon'] = {'url': ref_g, 'paper': ref_p}
            elif function == 'CoulombMatrix':
                ref_g = "no software package"
                ref_p = "M.  Rupp,  A.  Tkatchenko,  K.-R.  Muller,O.  A.  von  Lilienfeld,   Fast  and  Accu-rate  Modeling  of  Molecular  AtomizationEnergies  with  Machine  Learning,   Physi-cal  Review  Letters  108  (5)  (2012)  058301.doi:10.1103/PhysRevLett.108.058301"
                self.refs['CoulombMatrix'] = {'url': ref_g, 'paper': ref_p}
            elif function == 'BagofBonds':
                ref_g = "no software package"
                ref_p = "Hansen, K.; Biegler, F.; Ramakrishnan, R.; Pronobis, W.; von Lilienfeld, O. A.; Muller, K.-R.; Tkatchenko, A. Machine Learning Predictions of Molecular Properties: Accurate Many-Body Potentials and Nonlocality in Chemical Space J. Phys. Chem. Lett. 2015, 6, 2326 2331, DOI: 10.1021/acs.jpclett.5b00831"
                self.refs['BagofBonds'] = {'url': ref_g, 'paper': ref_p}
        elif module == 'tf':
            ref_g = "https://github.com/tensorflow/tensorflow"
            ref_p = "M. Abadi,  P. Barham,  J. Chen,  Z. Chen,A. Davis, J. Dean, M. Devin, S. Ghemawat,G. Irving, M. Isard, M. Kudlur, J. Levenberg,R. Monga, S. Moore, D. G. Murray, B. Steiner,P. Tucker, V. Vasudevan, P. Warden, M. Wicke,Y. Yu, X. Zheng, Tensorflow:  A system forlarge-scale machine learning, in: 12th USENIXSymposium on Operating Systems Design andImplementation (OSDI 16), USENIX Associa-tion, GA, 2016, pp. 265-283"
            self.refs['tensorflow'] = {'github': ref_g, 'paper': ref_p}

    def _save_references(self):
        with open(self.Base.output_directory+'/citation.txt','w') as file:
            for module in self.refs:
                file.write(module+':\n')
                for source in self.refs[module]:
                    file.write('    '+source+': '+self.refs[module][source]+'\n')
                file.write('\n')
