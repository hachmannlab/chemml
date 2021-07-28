====================
Input File Templates
====================

We present some of the input file templates in this section. They are available through the ChemML Wrapper GUI as well.



----
=======================================================
Input File Overview
=======================================================

The input file consists of methods that are connected to each other and make a machine learning workflow. We call this workflow
a computation graph. Thus, methods determine nodes
and are connected to each other using edges. Edges represent the flow of data between nodes. To determine edges of a graph,
fixed input and output tokens are defined for
each node and you should connect them using unique random integers. In the ChemML wrapper syntax we call a node of computation graph as a **block**.


Here you can see a pseudo block of ChemML Wrapper computation graph in the input file::

    ## Task
        << host = name        << function = name
        << parameter = value    << ...
        << ...                  << ...
        >> token id
        >> id token

All the methods in the ChemML Wrapper are available under these 8 tasks:
    - Enter
    - Represent
    - Prepare
    - Model
    - Search
    - Mix
    - Visualize
    - Store


.. note:: you always need an Enter method in your workflow to initialize the computation graph with some data.

----
=======================================================
Specific Characters
=======================================================

Only five specific characters (#, <, =, >, @) are defined and can be used in the input files.

- Pound sign (#)
    - Pound sign (#) determines a computation block.
    - It must be used to separate different blocks from each other.
    - A double pound sign (##) is for an active block and a single pound sign (#) keeps the block out of the graph.
    - No other specific signs can be in the same line as pound sign. Having a task name or any other text after the pound sign is arbitrary.

- Less-than sign (<)
    - Less-than sign (<) are used to pass the parameters' value.
    - The parameter's name comes right after this sign.
    - A Double less-than sign (<<) keeps a parameter in the block and single less-than sign (<) ignore the assigned value to that parameter.
    - If you are going to ignore a parameter value make sure it's not a required parameter.

    .. note:: two parameters are mandatory in all the blocks:

                - **host**: to specify the main library/dependency that you want to get the method from
                - **function**: to specify the ChemML Wrapper method
                You can find a comprehensive list of the available methods in the :ref:`ChemML_Wrapper_Table`

- Equals sign (=)
    - Equals sign (=) is for setting the value of parameters.
    - It comes right after the parameter name and is followed by its value.
    - The parameter value should be in its python format.

- Greater-than sign (>)
    - Greater-than sign (>) is all you need to connect blocks to each other (to send outputs or to receive inputs).
    - To keep track of inputs and outputs you have to assign a unique identification (ID) number to input/output tokens.
    - The ID number can be any integer. The ChemML Wrapper will extract, and assign the sent output of one block to the received input of another block through these unique IDs.
    - Note that the tokens are predefined for each block and can be found in the :ref:`ChemML_Wrapper_Table`.
    - To distinguish the send and receive actions you just need to switch the position of token and ID as described below:

        - to send an output token:
                        >> token  ID
                e.g.    >> molfile 7

        - to receive an input token:
                        >> ID token
                e.g.    >> 7 molfile

- At sign (@)
    - At sign (@) can be used to get a parameter value from the input/output values.
    - It comes right after equals sign (=) and should be followed by one of the input tokens (e.g. parameter = @df).

.. note:: please note that the first three characters (#, <, >) are reserved and you should avoid using them in the parameter values.

----
=======================================================
General Rules
=======================================================
A few general restrictions:

    - You are not allowed to have two different specific charecters in one line of input file (except '=' and '@' signs).
    - The input tokens and output tokens of each block may be similar but they might not have similar values.
    - Only one input per an input token can be received.
    - You are allowed to send output tokens to as many input tokens of different block as you want.
    - Avoid any type of short loop. A short loop will be made when inputs of any block_i are received from one or a set of blocks that they require an output of block_i.
    - If you make a short loop any place inside your workflow your run will be aborted immediately.