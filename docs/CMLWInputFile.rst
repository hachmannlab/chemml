=============================
Input File Format
=============================

The input file should be produced by using the Input builder available in the website. However, the file is a text file
that contains all the information needed for making the entire workflow. Thus the user can manually edit or build it from
scratch with any text editor. As the input file could be quite complex, it is strongly suggested always to use the Input builder.

----
=======================================================
Input File Structure
=======================================================

Here is how a block of ChemML Wrapper workflow looks in the input file::

    ## Task
        << host = name        << function = name
        << parameter = value    << ...
        << ...                  << ...
        >> token id
        >> id token

Here is a list of 7 available tasks in the ChemML Wrapper:
    - Enter
    - Prepare
    - Model
    - Search
    - Mix
    - Visualize
    - Store


.. note:: None of Tasks are mandatory, except a block of Enter Data to initialize the workflow with a data set.

----
=======================================================
Specific Characters
=======================================================

Only five specific characters (#, <, =, >, @) are defined and can be used in the input files.

- Pound sign (#)
    Pound sign (#) is for starting a block and must be followed by the task name. A double pound sign (##) keeps a block
    in the workflow and single pound sign (#) put the entire block out.

- Less-than sign (<)
    Less-than sign (<) should be used for identifying parameters in the block. A parameter name comes right after this sign.
    A Double less-than sign (<<) keeps a parameter in the block and single less-than sign (<) ignore the assigned value to that
    parameter. If you are going to ignore a parameter value make sure it's not a required  parameter or at least have a default
    value.

    .. note:: two parameters are mandatory for all the blocks:

                - **host**: to specify the python library that must be used for the task
                - **function**: to specify the ChemML Wrapper function that makes binding with the main function in the specified library.
- Equals sign (=)
    Equals sign (=) is for setting the value of parameters. It comes right after the parameter name and is followed by its
    value. The parameter value should be in the same format as the python parameter. For example, the string format must always
    be inside the quotation marks.

- Greater-than sign (>)
    Greater-than sign (>) is all you need to design the send/receive stream ( to send outputs or to receive inputs). Thus, it facilitates connection
    between different blocks. Every block includes some predefined tokens (container names) to receive information from other
    blocks or to send a piece of information out. This way a workflow of blocks can be designed.

    To keep track of inputs and outputs you have to assign a unique identification (ID) number to each token. This number
    can be any integer. The ChemML Wrapper will extract, and assign the sent output of one block to the received input
    of another block through these unique IDs. However the name of tokens are predefined by each block and can be found
    in the :ref:`ChemML_Wrapper_Reference`. To distinguish the send and receive actions you just need to switch the position of token
    and ID, in a way that:

    - to send:
                    >> token  ID
            e.g.    >> molfile 7

    - to receive:
                    >> ID token
            e.g.    >> 7 molfile

- At sign (@)
    At sign (@) can be used to get a parameter value from the send/receive stream. It comes right after equals sign (=)
    and should be followed by one of the input tokens (e.g. parameter = @df).

----
=======================================================
General Rules
=======================================================
Be aware of some general restrictions:

    - You are not allowed to have two different specific charecters in one line of input file (except '=' and '@' signs).
    - The input tokens and output tokens for each block may be similar but they don't have to have similar values.
    - Only one input per legal input token can be received.
    - You are allowed to receive one sent token in several blocks.
    - Avoid any type of short loop. A short loop will be made when inputs of any block_i are going to be received from one or a set of blocks that they also require the outputs from block_i.
    - If you make a short loop any place inside your workflow your run will be aborted immediately.