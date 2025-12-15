from matplotlib import pyplot as plt
from model.ffnn import NeuralNetwork, compute_loss
import numpy as np


def batch_train(X, Y, model, train_flag=False):
    ################################# STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) Use your neural network to predict the intent
    #         (without any training) and calculate the accuracy 
    #         of the classifier. Should you be expecting high
    #         numbers yet?
    #         2) if train_flag is true, run the training for 1000 epochs using 
    #         learning rate = 0.005 and use this neural network to predict the 
    #         intent and calculate the accuracy of the classifier
    #         3) Then, plot the cost function for each iteration and
    #         compare the results after training with results before training

    ''' Part 1: Prediction without training 
    To compare and calculate the accuracy, class indices are obtained from the preidictions and ground truth labels.
    Accuracy is calculated as the ratio of correct predictions to total predictions '''

    predictions = model.predict(X) # shape for predictions : (K, M)
    predicted_classes = np.argmax(predictions, axis=0)  # shape: (M,)
    true_classes = np.argmax(Y, axis=0)  # shape: (M,)
    accuracy_before_training = np.mean(predicted_classes == true_classes) # compares element-wise and takes mean
    print(f"Accuracy before training: {accuracy_before_training * 100:.3f}%")

    ''' Part 2: Training the neural network if train_flag is True with Batch Gradient descent. Update weight and bias once per epoch '''

    if train_flag:
        learning_rate = 0.005
        num_epochs = 1000
        M = X.shape[1]  # number of examples
        cost_history = []

        for epoch in range(num_epochs):
            # Forward pass
            Y_hat = model.forward(X)

            # Compute loss
            loss = compute_loss(Y_hat, Y)
            cost_history.append(loss)

            # Backward pass
            dw1, db1, dw2, db2 = model.backward(X, Y)

            # Update weight and bias matrices
            model.w1 -= learning_rate * dw1 / M
            model.b1 -= learning_rate * db1 / M
            model.w2 -= learning_rate * dw2 / M
            model.b2 -= learning_rate * db2 / M

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
        # Evaluate after training
        final_predictions = model.predict(X)
        final_pred_classes = np.argmax(final_predictions, axis=0)
        final_accuracy = np.mean(final_pred_classes == true_classes)
        print(f"Final accuracy (after training): {final_accuracy * 100:.2f}%")
        
        # 3) Plot the cost function
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history, label='Cost')
        plt.title('Loss vs Epochs - Batch Gradient Descent')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('batchGD_comparison.png')
        plt.close()
    ###############################################################################


def minibatch_train(X, Y, model, train_flag=False):
    ########################## STUDENT SOLUTION #############################
    # YOUR CODE HERE
    #     TODO:
    #         1) As bonus, train your neural network with batch size = 64
    #         and SGD (batch size = 1) for 1000 epochs using learning rate
    #         = 0.005. Then, plot the cost vs iteration for both cases.
    #print(train_flag)
    true_classes = np.argmax(Y, axis=0)  # shape: (M,)

    '''Note: As discussed with Juan during the lecture, in my implementation the "--minibatch" flag triggers training only when `train_flag` is False. 
    Alternatively, training can be triggered using "--minibatch --train" when `train_flag` is True, depending on how the condition is set.'''

    if not train_flag:
        learning_rate = 0.005
        num_epochs = 1000
        batch_sizes = [64, 1]  # Mini-batch and SGD
        M = X.shape[1]  # number of examples

        # Dictionary to store cost histories
        cost_histories = {size: [] for size in batch_sizes}

        for size in batch_sizes:
            print(f"\nTraining with batch size = {size}")

            # Reset model for fair comparison
            model.__init__(model.w1.shape[1], model.w1.shape[0], model.w2.shape[0])

            for epoch in range(num_epochs):
                epoch_cost = 0
                # Shuffle data at beginning of each epoch
                perm = np.random.permutation(M)
                # Process mini-batches or single samples depending on batch size
                for start_idx in range(0, M, size):
                    end_idx = min(start_idx + size, M)
                    batch_indices = perm[start_idx:end_idx]
                
                    # Get current batch
                    x_batch = X[:, batch_indices]
                    y_batch = Y[:, batch_indices]
               
                    # Forward pass
                    Y_hat = model.forward(x_batch)

                    # Compute loss
                    loss = compute_loss(Y_hat, y_batch)
                    epoch_cost += loss * (end_idx - start_idx)  # accumulate weighted by batch size

                    # Backward pass
                    dw1, db1, dw2, db2 = model.backward(x_batch, y_batch)

                    # Update weights and biases
                    model.w1 -= learning_rate * dw1 / x_batch.shape[1]
                    model.b1 -= learning_rate * db1 / x_batch.shape[1]
                    model.w2 -= learning_rate * dw2 / x_batch.shape[1]
                    model.b2 -= learning_rate * db2 / x_batch.shape[1] 
                    
                # Average loss per epoch, normalises the loss across different batch sizes.
                epoch_cost /= M
                cost_histories[size].append(epoch_cost)   

                # Print every 100 epochs
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {epoch_cost:.4f}")

            # Final accuracy
            final_pred = model.predict(X)
            final_acc = np.mean(np.argmax(final_pred, axis=0) == true_classes)
            print(f"Final accuracy (batch size {size}): {final_acc * 100:.2f}%")

            # Plot and save cost curve for this batch size
            plt.figure(figsize=(10, 6))
            label = "SGD (batch size = 1)" if size == 1 else "Mini-batch (batch size = 64)"
            plt.plot(cost_histories[size], label=label)
            plt.title(f"Loss vs Epoch - {label}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            filename = "stochasticGD_comparison.png" if size == 1 else "minibatchGD_comparison.png"
            plt.savefig(filename)
            plt.close()
    #########################################################################