# Note

Thank you for your interest at our work. This repo contains the code which we use ourselves (and probably difficult to use for other people). We will, however, try our best to make it into a friendly version and update it as soon as possible. The dataset will also be published soon.

# How to use our code for KNET

Our codes are divided into three parts: Training, Testing and Manual_Test, and they are put into corresponding folders. Testing is carried on `WIKI-AUTO` set and Manual_Test is carreid on `WIKI-MAN` set.

Because we have to upload codes and datasets seperately, the path for loading and storing data in all the codes inevitably require manual adjustment according to your own environments.

### Prerequisite

*   python 2.7
*   numpy 1.11.3
*   tensorflow 0.12.1

All the codes are tested under Ubuntu 16.04.

### Training

Each model has a corresponding folder under `./Training`. 

Take KA+D for example, goto the folder `./Training/KA+D`, and then simply run `$ python ka+d.py`.

The code will report results on both validation set and test set regularly, and also store parameters in corresponding files.

### Testing

Each model has a corresponding folder under `./Testing`.

Take KA+D for example, goto the folder `./Testing/KA+D`, and then simply run `$ python ka+d.py`. The results will be printed on screen.

Note that the file `linktest.npy` and `linkman.npy` from the datasets contain the information about which entity mentions have been successfully disambiguated to the right entities. This, however, differs for every training results of KA+D, and should be adjusted accordingly. Our version is based on our trained model of KA+D.

### Manual_test

This part is similar to Testing, the only difference is the dataset.

### Dataset

Dataset comprises a number of `.npy` files, which can be loaded by using `numpy.load()`.

*   `embedding.npy` contains entity embeddings trained by TransE. Each row is an entity embedding vector.
*   `link*.npy` contains the information about which entity mentions have been successfully disambiguated to the right entities. The latter part of file name indicates which dataset it is. They are one-dimensional vectors, 1 for link successful, 0 for missed.
*   `*_entity.npy` contains entity mention part of each dataset. Each row is one entity mention, i.e., an array of strings, e.g., ['John', 'H.', 'Watson'].
*   `*_context.npy`contains context part of each dataset. Each row is the context of one sentence, e.g., ['lives', 'in', '221B', 'Baker', 'Street'].
*   `*_fbid.npy` contains the Freebase ID for each entity. They are one-dimensional vectors, each entry is the ID, i.e., an integer.
*   `*_label.npy` contains the ground-truth types for each entity. Each row is a 74-dimensional vector, 1 for true, 0 for false.
