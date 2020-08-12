import logging
import numpy as np
import os
import matplotlib.pyplot as plt


def get_data(label, centered=False, flatten=False,
     rand_state = np.random.RandomState(1), label_is_normal=True):
    """Gets the adapted dataset for the experiments

    Args :
            split (str): train, valid or test
            label (int): int in range 0 to 10, is the class/digit which is considered normal
            centered (bool): (Default=False) data centered to [-1, 1]
            flatten (bool): (Default=False) flatten the data
            label_is_normal(bool): wheter given lebel argument is normal or anomaly
    Returns :
            (tuple): <training, testing> images and labels
    """

    path = '/home/mlg/mnist'
    data = np.load(os.path.join(path, 'mnist.npz'))

    x_train = np.concatenate([data['x_train'], data['x_valid']], axis=0)
    y_train = np.concatenate([data['y_train'], data['y_valid']], axis=0)

    normal_idx = [i for i,e in enumerate(y_train) if e == label]
    x_train = x_train[normal_idx]

    dataset = {}
    dataset['x_train'] = x_train/255.
    dataset['x_test'] = data['x_test']/255.
    dataset['y_test'] = np.array([0 if e == label else 1 for e in data['y_test']], dtype=int)

    perm_idx = rand_state.permutation(dataset['x_train'].shape[0])
    dataset['x_train'] = dataset['x_train'][perm_idx]
    perm_idx = rand_state.permutation(dataset['x_test'].shape[0])
    dataset['x_test'] = dataset['x_test'][perm_idx]
    dataset['y_test'] = dataset['y_test'][perm_idx]
   
    if centered:
        dataset['x_train'] = dataset['x_train']*2-1
        dataset['x_test'] = dataset['x_test']*2-1

    if not flatten:
        dataset['x_train'] = np.reshape(dataset['x_train'], [dataset['x_train'].shape[0], 28, 28, 1])
        dataset['x_test'] = np.reshape(dataset['x_test'], [dataset['x_test'].shape[0], 28, 28, 1])

    return dataset 

if __name__ == '__main__':
    dataset = get_dataset(label=8)

    x_train = dataset['x_train']
    x_test = dataset['x_test']
    y_test = dataset['y_test']

    x_train = np.random.permutation(x_train)
    perm_idx = np.random.permutation(x_test.shape[0])
    x_test = x_test[perm_idx]
    y_test = y_test[perm_idx]

    print(x_train.shape)
    print(x_test.shape)
    print(y_test.shape)


    for i in range(5):
        print(y_test[5*i:5*i+5])

    plt.figure()
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(x_test[i][:,:,0], cmap='gray')    


    plt.figure()
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(x_train[i][:,:,0], cmap='gray')    

    plt.show()




