EN.601.461 - Computer Vision
Homework 3
Ye Chan Kim - ykim160

Make Deep Learning Happen With Faces:

Google Drive Liink: https://drive.google.com/drive/folders/1G2lAS5jJu7rQXAZodv1qhANymWslMrfS?usp=sharing 

 - aug_model is the model using augmentation for part a)
 - aug_modelb is the model using augmentation for part b)

1)
    a) For this section I created a Siamese Network as structured in the assignemnt.
       Since part a) uses Binary-classification for the last step of the CCN I 
       concatenated the outputs of f1 and f2 and returned it. I used the given BCELoss
       function of torch and tested the training and testing with and without agumentation.
       As expected after training the CNN with the training set when I tested the
       training data, I got a result of 100% accuracy. This varies a little depending on
       the values I set for the batch_size, number_workers, and epochs but most of my
       results were fairly close to 100%. The testing data without augmentation gave me
       a result of 53.10% when I used batch_size=16, num_workers=2, learning_rate=1e-6,
       and epochs=10. This is expected because the training set is not that big and the
       images in the test data have never been seen before by the CNN.

       When training the neural network using augmentation with probability of 70% and
       probabiliy of 50% for each augmentation, the results were different as expected.
       The Accuracy for the training data set decreased to 94.41% because we are looking
       at slightly different pictures. Meanwhile the accuracy for testing dataset increased
       to 54.90%. This might be because the random angles chosen for the pictures match
       slightly better with the pictures of the testing data set.

       Results:
       - Running without augmentation:
           - Training Data Accuracy: 100%
           - Testing Data Accuracy: 53.10%
       - Running with augmentation:
           - Training Data Accuracy: 94.41%
           - Testing Data Accuracy: 54.90%

       Maybe a way to improve the Accuracy of the testing data set is training the CNN less
       by running less epochs but I don't think that would do very much in this case.

    b) The Second implementation was Contrastive-loss function. Most of the code is very similar
       as the first section except the CNN and the way to classify the correct predictions.
       For the CNN in this case instead of concatenating the two f results it just outputs both
       of them out to calculate the distance for contrastive loss. I use the torch pairwise_distance
       and then calculate function L using the equation provied in the homework. My expectations
       before running the test cases was that the accuracy of training data would go down but
       the accuracy of the testing data would go up because BCEloss is more like a binary decission
       as to whether the features match therefore, if the face has not been seen by the CNN it
       would have a hard time figuring it out but for the contrastive loss its more like comparing
       to each other and clumping the simlar faces therefore being able to compare new data sets.

       However, the results didn't match my expectations. I think this is mainly because I didn't run
       for enough epochs since I trained it for only epochs = 15. For this specific parameters I think
       I found the most optimal threshold to used to get the best results. However, I don't think
       I lowered the loss enough to make it work. This is because the accuracy of training data set
       is fairly low meaning the CNN didn't really learn. This is probably because I didn't run it
       enough times because of time contrains. I think the loss should start decreasing around epochs = 80.

       Results:
       - Running without augmentation:
           - Training Data Accuracy: 68.09 
           - Testing Data Accuracy: 52.20
       - Running with augmentation:
           - Training Data Accuracy: 74.41
           - Testing Data Accuracy: 52.50

       The results for Contrastive-loss should have been better than binary-classification but finding
       the best margin, and threshold seems to be a challenge for this loss function type.
