# TransferLearning
 Some experiments to practice transfer learning.
 Jupyter notebooks from Google Colab.
 transfer_learning_basics.ipynb shows freezing the classifier layer and 
 retraining the network, followed by saving the output.
transfer_learning_basics_reloading_original_model.ipynb is a reload of 
the original transfer learning model, followed by unfreezing of the 
last Conv layer, which precedes the classification layer, and training 
for two epochs. The initial results suggest this leads to overfitting 
and reduced performance. I'm initially presuming that this is because
the additional parameters are disproportionately large compared to 
the training data available, but haven't looked into the numbers. 
