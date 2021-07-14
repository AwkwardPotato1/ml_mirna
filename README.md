# ml_mirna
I have a dataset (link here) for binary classification containing 27149 rows (observations) and 93 columns (features). The data has 23,293 negative samples and 3855 positive samples. I used SMOTE and Near-Miss to handle the imbalance and tuned the hyper-parameters using GridSearchCV, with K=5 Folds. The initial results were very overwhelming and I was certain that it was overfitting:

CV Results

As standard procedure, I took the dataset and split it for training & testing, and applied SMOTE and Near-Miss to the training data along with the respective hyperparameters for each classifier obtained from CV score. The testing dataset was kept constant with 5852 negative and 936 positive test samples.

The performance of the classifiers on the test dataset was overwhelming as well.

E.g.
SVM trained on SMOTE:
TP: 879, FN: 57
FP: 51, TN: 5801 

Random Forest trained on SMOTE:
TP: 875, FN: 61
FP: 62, TN: 5790
Hence, to generalize this hypothesis I took another set of testing data generated from the source. (link here). This dataset contains 463 positive and 535 negative samples. This is where the problem starts. The performance of the trained classifiers lowers drastically with this dataset.

Same Random Forest now gives
TP: 263, FN: 200
FP: 73, TN: 462
And that's the best performing classifier among all the previously trained!

Initially I thought by some chance the new test data was somehow way different from the training data and hence, replaced some instances with the original master dataset but the performance did not change.

I do not understand how performance on split test data can be so different from unseen raw data.

Things went bad to worse when I tried to make a copy of the True observations from the new test dataset and feed to the model.

df_Pos = df[ df["SEQ"] == 1]  #SEQ = y

And the performance dropped to

 TP: 125, FN: 338
How can a model discriminate in the classification of the same instances in different dataframes?

Why is there no consistency in the predictions?
