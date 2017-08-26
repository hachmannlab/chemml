# class PolynomialFeatures_sklearn(object):
#     def __init__(self,degree=2, interaction_only=False, include_bias=True):
#         self.degree = 2, interaction_only = False, include_bias = True


# class cheml_RDKitFingerprint(BASE):
#     def display(self):
#         inputs = self._input_vbox()
#         params = self._param_vbox()
#         outputs = self._output_vbox()
#         buttons = self._buttons()
#
#         caption = widgets.Label(value='${RDKitFingerprint}$', \
#                                 layout=widgets.Layout(width='50%', margin='10px 0px 0px 440px'))
#         accordion = widgets.Accordion(children=[inputs, params, outputs], \
#                                       layout=widgets.Layout(width='80%', margin='10px 0px 10px 100px'))
#         accordion.set_title(0, 'input/receivers')
#         accordion.set_title(1, 'parameters')
#         accordion.set_title(2, 'output/senders')
#
#         layout = widgets.Layout(border='solid')
#         final = widgets.VBox([caption, accordion, buttons], layout=layout)
#         display(final)
#
#     def _buttons(self):
#         add = widgets.Button(description="Add")
#         # add.on_click(self.on_add_clicked)
#
#         cancel = widgets.Button(description="Cancel")
#         # cancel.on_click(self.on_cancel_clicked)
#
#         buttons = widgets.HBox([add, cancel], layout=widgets.Layout(margin='10px 0px 10px 350px'))
#         return buttons
#
#     def _param_vbox(self):
#         self.parameters = {'removeHs': True, 'FPtype': 'Morgan', 'vector': 'bit', 'nBits': 1024, 'radius ': 2}
#         self.function_counter += 1
#         # caption = widgets.Label(value='Parameters:',layout=widgets.Layout(width='50%'))
#         removeHs = widgets.Checkbox(
#             value=True,
#             description='removeHs:',
#             disabled=False)
#         FPtype = widgets.Dropdown(
#             options=['HAP', 'AP', 'MACCS', 'Morgan', 'HTT', 'TT'],
#             value='HAP',
#             description='FPtype:',
#             disabled=False,
#             button_style=''  # 'success', 'info', 'warning', 'danger' or ''
#         )
#         params = widgets.VBox([removeHs, FPtype])
#         return params
#
#     def _input_vbox(self):
#         df = widgets.Checkbox(
#             value=False,
#             description='df',
#             disabled=False)
#         tokens = widgets.Dropdown(
#             options=['HAP', 'AP', 'MACCS', 'Morgan', 'HTT', 'TT'],
#             value='HAP',
#             description='tokens:',
#             disabled=not df.value,
#             button_style=''  # 'success', 'info', 'warning', 'danger' or ''
#         )
#         buttons = widgets.HBox([df, tokens])
#         return buttons
#
#     def _output_vbox(self):
#         df = widgets.Checkbox(
#             value=False,
#             description='df',
#             disabled=False)
#         api = widgets.Checkbox(
#             value=False,
#             description='api',
#             disabled=False)
#         outputs = widgets.VBox([df, api])
#         return outputs




