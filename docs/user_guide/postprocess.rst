.. _postprocessing:

***************
Postprocessing
***************

Here we cover how to export the trained embeddings for use. See :ref:`user_guide_marius_postprocess` for details on ``marius_postprocess``.
We assume ``<path.base_directory>/<general.experiment_name>`` is ``./data/marius/``
and ``./training_data/`` contains all the preprocessed dataset files.

Supported formats for exporting embeddings:
-------------------------------------------

PyTorch tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    marius_postprocess ./data/marius/ ./training_data/ --output_directory ./embeddings/ --format Tensor

CSV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    marius_postprocess ./data/marius/ ./training_data/ --output_directory ./embeddings/ --format CSV

TSV file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

    marius_postprocess ./data/marius/ ./training_data/ --output_directory ./embeddings/ --format TSV

Accessing embeddings by original IDs
-------------------------------------------

During preprocessing, a unique integer id was assigned to each node and edge-type. The embedding table is ordered accorded to the mapped integer id, such that the first row (embedding) in the embedding table corresponds to the embedding for node ID 0.

We store the mapping of the original ids to the integer ids in ``<preprocess_dir>/node_mapping.txt``, where this file is two column TSV. The first column is the original ID (can be a string, integer, etc.) and the second column is the corresponding row index in the embedding table. By looking up the ID in this mapping the index in the embedding table can be found.