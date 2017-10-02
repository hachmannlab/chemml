Ipython Notebook
================
::

                                                  Hello World!

.. code:: python

    from ipywidgets import *
    from IPython.display import display
    w1 = IntSlider()
    display(w1)
    
    w2 = Text()
    display(w2)
    
    button = Button(description="Submit")
    display(button)
    dict = {'val':w1.value,'text':w2.value}
    def on_button_clicked(b):
    #     print(dict)
        if dict['text']=='hi':
            w3 = Checkbox(value=False,description='Hi there!',disabled=False)
            display(w3)
    def handle_submit(sender):
        dict['text'] = w2.value
    def on_value_change(change):
        #print (change)
        dict['val'] = change['new']
    def on_text_change(change):
        #print (change)
        dict['text'] = change['new']
    
    
    # w2.on_submit(handle_submit)
    w1.observe(on_value_change, names = 'value')
    w2.observe(on_text_change, names='value')
    button.on_click(on_button_clicked)








.. code:: python

    if dict['val']==77:
        w4 = Checkbox(value=False,description='We love number 7!',disabled=True)
        display(w4)

.. code:: python

    from tabulate import tabulate
    import pandas as pd
    df = pd.DataFrame({"x":[1,2,3], "y":[6,4,3], "z":["testing","pretty","tables"], "f":[0.023432, 0.234321,0.5555]})
    print tabulate(df, headers='keys', tablefmt='psql')



.. parsed-literal::

    +----+----------+-----+-----+---------+
    |    |        f |   x |   y | z       |
    |----+----------+-----+-----+---------|
    |  0 | 0.023432 |   1 |   6 | testing |
    |  1 | 0.234321 |   2 |   4 | pretty  |
    |  2 | 0.5555   |   3 |   3 | tables  |
    +----+----------+-----+-----+---------+

