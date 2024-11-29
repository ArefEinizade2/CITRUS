import torch
import numpy as np
torch.set_default_tensor_type(torch.FloatTensor)

class _data:
    # Internal supraclass from which all data sets will inherit.
    # There are certain methods that all Data classes must have:
    #   getSamples(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    # noinspection PyDictCreation
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['labels'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['labels'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['labels'] = None

    def getSamples(self, samplesType, *args):
        # type: # train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
               or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0]  # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size=args[0],
                                                   replace=False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices, :].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0]]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return torch.from_numpy(x).to(self.device).float(), torch.from_numpy(y).to(self.device)

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This
        # methods should still be initialized within the data classes, if more
        # attributes are used.

        # The labels could be integers as created from the dataset, so if they
        # are, we need to be sure they are integers. To do this we need to
        # match the desired dataType to its int counterpart. Typical examples
        # are:
        #   numpy.float64 -> numpy.int64
        #   numpy.float32 -> numpy.int32
        #   torch.float64 -> torch.int64
        #   torch.float32 -> torch.int32

        labelType = str(self.samples['train']['labels'].dtype)
        if 'int' in labelType:
            if 'numpy' in repr(dataType) or 'np' in repr(dataType):
                if '64' in labelType:
                    labelType = np.int64
                elif '32' in labelType:
                    labelType = np.int32
            elif 'torch' in repr(dataType):
                if '64' in labelType:
                    labelType = torch.int64
                elif '32' in labelType:
                    labelType = torch.int32
        else:  # If there is no int, just stick with the given dataType
            labelType = dataType

        # Now that we have selected the dataType, and the corresponding
        # labelType, we can proceed to convert the data into the corresponding
        # type
        if 'torch' in repr(dataType):  # If it is torch
            for key in self.samples.keys():
                self.samples[key]['signals'] = \
                    torch.tensor(self.samples[key]['signals']).type(dataType)
                self.samples[key]['labels'] = \
                    torch.tensor(self.samples[key]['labels']).type(labelType)
        else:  # If it is not torch
            for key in self.samples.keys():
                self.samples[key]['signals'] = \
                    dataType(self.samples[key]['signals'])
                self.samples[key]['labels'] = \
                    labelType(self.samples[key]['labels'])

        # Update attribute
        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                        = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device


class _DataForClassification(_data):
    # Internal supraclass from which data classes inherit when they are used
    # for classification. This renders the .evaluate() method the same in all
    # cases (how many examples are correctly labels) so justifies the use of
    # another internal class.

    def __init__(self):

        super().__init__()

    def evaluate(self, yHat, y, tol=1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim=1)
            #   And compute the error
            y = y.type(torch.long)
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.type(self.dataType) / N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis=1)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType) / N
        #   And from that, compute the accuracy
        return accuracy


class SourceLocalizationFromDataset(_DataForClassification):
    def __init__(self, trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels, dataType=np.float64,
                 device='cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device

        self.nTrain = trn_data.shape[0]
        self.nValid = val_data.shape[0]
        self.nTest = tst_data.shape[0]

        print(f"{self.nTrain} training samples. \n{self.nValid} validation samples. \n{self.nTest} test samples.")
        print(f"Train shape: {trn_data.shape}")

        # Split and save them
        self.samples['train']['signals'] = trn_data
        self.samples['train']['labels'] = trn_labels
        self.samples['valid']['signals'] = val_data
        self.samples['valid']['labels'] = val_labels
        self.samples['test']['signals'] = tst_data
        self.samples['test']['labels'] = tst_labels
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
