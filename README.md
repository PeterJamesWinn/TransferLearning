# TransferLearning
 Some experiments to practice transfer learning.

 The first set of Jupyter notebooks from Google Colab are based on 
 a tutorial: https://www.learnpytorch.io/06_pytorch_transfer_learning/
 which is provided with the standard MIT licence. That tutorial splits the
 data into a train and test set, i.e. is there is no split into train 
 test and validation and the term test would probably better be replaced
 with validation in this context, being used to check for evidence
 of overfitting and model selection. 


 The tutorial uses the Efficient Net B0 model and retrains 
 the classifier layer on 
 training data consisting of 78 images of pizza, 75 of steak and 72 of 
 sushi. 
 The test set has 25 images of pizza, 19 of steak and 31 of sushi. 

Not really enough data below to come to a conclusion of the
trend in the data compared to variation between optimisations, but AdamW
seems to perform better for generalisation, as expected from the literature,
and unfreezing the Conv2D layer may be helpful compared to just the 
classifier layer ... but I've not done enough calculations to be sure. 

 transfer_learning_basics.ipynb shows freezing the classifier layer and 
 retraining the network, followed by saving the output. Adam was used
 as the optimiser
 Epoch: 1 | train_loss: 1.0977 | train_acc: 0.3828 | test_loss: 0.8725 | test_acc: 0.7225
Epoch: 2 | train_loss: 0.8703 | train_acc: 0.7812 | test_loss: 0.7731 | test_acc: 0.8456
Epoch: 3 | train_loss: 0.8145 | train_acc: 0.7031 | test_loss: 0.7206 | test_acc: 0.8163
Epoch: 4 | train_loss: 0.7041 | train_acc: 0.7617 | test_loss: 0.5957 | test_acc: 0.8864
Epoch: 5 | train_loss: 0.6082 | train_acc: 0.7500 | test_loss: 0.5924 | test_acc: 0.9167
Followed by
Epoch: 1 | train_loss: 0.5846 | train_acc: 0.7656 | test_loss: 0.5288 | test_acc: 0.8759
Epoch: 2 | train_loss: 0.4961 | train_acc: 0.9336 | test_loss: 0.5178 | test_acc: 0.8258
the last two epochs seem to show evidence of overfitting even though the 
classification layer has dropout. Although could be within the bounds
of the random fluctuation. 

transfer_learning_basics_reloading_original_model.ipynb is a reload of 
the original transfer learning model, followed by unfreezing of the 
last Conv layer, which precedes the classification layer, and training 
for two epochs. The initial results suggest this leads to overfitting 
and reduced performance, since the "test" loss barely decreases, 
and the accuracy gets worse, whereas the train loss halves and the 
train accuracy massively improves.
Epoch: 1 | train_loss: 0.5078 | train_acc: 0.8945 | test_loss: 0.3762 | test_acc: 0.9271
Epoch: 2 | train_loss: 0.2782 | train_acc: 0.9766 | test_loss: 0.3542 | test_acc: 0.8456

transfer_learning_basics_repeat_with_Adam.ipynb
as the first but with different random seed. Improved results over both 
the previous two showing that random variation is bigger than choice
of optimiser or whether to freeze one or two layers. 
Epoch: 1 | train_loss: 0.5668 | train_acc: 0.7833 | test_loss: 0.3417 | test_acc: 0.9250
Epoch: 2 | train_loss: 0.3240 | train_acc: 0.9208 | test_loss: 0.2929 | test_acc: 0.9068
Epoch: 3 | train_loss: 0.2593 | train_acc: 0.8917 | test_loss: 0.2442 | test_acc: 0.9318
Epoch: 4 | train_loss: 0.2300 | train_acc: 0.9167 | test_loss: 0.2277 | test_acc: 0.9443

transfer_learning_basics_reloading_original_model_AdamW.ipynb
starting from the first model after the first five epochs of training,
unfreezing the weights in the last Conv2D layer, training with AdamW, 
but otherwise the same. This combination seems to be the best, since
see improvements here and in the next one. 
Epoch: 1 | train_loss: 0.5633 | train_acc: 0.8542 | test_loss: 0.3508 | test_acc: 0.9386
Epoch: 2 | train_loss: 0.3164 | train_acc: 0.8833 | test_loss: 0.2544 | test_acc: 0.9250
Epoch: 3 | train_loss: 0.2089 | train_acc: 0.9500 | test_loss: 0.1811 | test_acc: 0.9500
Epoch: 4 | train_loss: 0.2012 | train_acc: 0.9125 | test_loss: 0.1825 | test_acc: 0.9375

transfer_learning_basics_reloading_original_model_AdamW_second_repeat.ipynb
as above, but with different random seed for the epochs of training after
unfreezing the Conv2D layer.
Epoch: 1 | train_loss: 0.5668 | train_acc: 0.7833 | test_loss: 0.3418 | test_acc: 0.9250
Epoch: 2 | train_loss: 0.3241 | train_acc: 0.9208 | test_loss: 0.2930 | test_acc: 0.9068
Epoch: 3 | train_loss: 0.2593 | train_acc: 0.8917 | test_loss: 0.2442 | test_acc: 0.9318
Epoch: 4 | train_loss: 0.2300 | train_acc: 0.9167 | test_loss: 0.2277 | test_acc: 0.9443
