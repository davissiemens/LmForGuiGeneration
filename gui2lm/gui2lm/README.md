# GUI Generation via Language Modeling 
We create a GUI generation framework based on language modeling in this work. It consists of the following classes and dictionaries:
- **configuration**: Contains the configuration class including file paths and abstraction settings.
- **data_abstracting**: Abstractor to process the RICO dataset into the GUI-Abstraction dataset. 
  It contains the main class abstractor.py and multiple helper classes. Filter.py is used to filter GUIs based on their attributes.
  labelcounter.py counts the number of GUI element classes in the dataset. In the guiclasses-directory classes representing GUIs and GUI-elements can be found. 
- **language_model**: Contains the implementation of the language model. Languagemodel_with_hypertuning_V2.py is the 
  main class implementing the language model. lm_optimizer.py creates a new language model and finds its best hyperparameters if started. 
  However, one has to keep in mind that this is a very computational intensive step. printer.py is a helper class to create and store pretty prints. 
  - **console_output**: "hypertune_standard-model" contains the ouput for the hypertune process described in the paper
  - **hyper_param**: stored best hyperparameter values
  - **logs**: tensorboard log files
  - **text_generations**: here the GUI generations of the language models gets stored
- **models**: Contains the weights of the trained models. The standard model is stored as "Overreach1". 
- **preprocessing**: Contains the preprocessor.py which translates the abstracted GUI dataset to GUI-language dataset. 
  The preprocessor.py skript contains various helper functions to ,e.g., translate only single abstraction, find a GUIs 
  number by it's char string etc. Token.py represents the vocabulary of the GUI-language. 
- **ressources**: Contains the datasets. Please add required datasets as described in the prerequisites before first time use. 
  - **abstractions**: Contains .json files of abstracted GUIs
  - **preprocessed**: Contains readable and char strings of the GUI-language dataset. 
- **main.py:** Main skript to preprocess dataset, train the model or generate GUIs. Instructions are given within the skript. 

## Prerequisites 
To make sure the program is fully executable following steps must be taken. 
- **Dataset**: Please add the RICO datasets, which can be downloaded from the following website:
    https://interactionmining.org/rico. The datasets *"UI Screenshots and View Hierarchies"* must be added to 
    the "resources/combined" folder; *"UI Screenshots and Hierarchies with Semantic Annotations"* to the "resources/semantic_annotations" 
    folder. The two CSV files *"UI Metadata"* and *"Play Store Metadata"* need to be provided in this directory as well.
- **Tensorflow & Keras**: For this project the tensoflow, the keras as well as the tensorboard package must be installed. 
- **Configuration**: Before start, the root path in configuration.py must be configured accordingly (see comments in configuration.py).

## Generating GUIs
To start the language model, the main.py skript can be used. The skript also includes instructions to start the abstractor and preprocessor,
as well as to train a new language model. This is only required if the grid structure in the configuration.py is changed. 