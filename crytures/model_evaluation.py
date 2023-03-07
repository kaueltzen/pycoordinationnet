
import torch
import pytorch_lightning as pl

from monty.serialization import dumpfn

## Cross-validation
## ----------------------------------------------------------------------------

def eval_kfold(
        # Arguments
        get_model, data, filename_result,
        # Options
        patience = 100, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = None,
        ):

    y_hat = torch.tensor([], dtype = torch.float)
    y     = torch.tensor([], dtype = torch.float)

    for fold in range(data.n_splits):

        print(f'Training fold {fold+1}/{data.n_splits}...')
        data.setup_fold(fold)

        # Create a new model
        model = get_model()

        # Train and test model
        best_val_score, test_y, test_y_hat = model.train_model_and_test(data)

        # Print score
        print(f'Best validation score: {best_val_score}')

        # Save predictions for model evaluation
        y_hat = torch.cat((y_hat, test_y_hat))
        y     = torch.cat((y    , test_y    ))

    # Compute final test score
    mae = torch.nn.L1Loss()(y_hat, y).item()

    print('Final MAE:', mae)

    # Save result
    dumpfn({'y_hat': y_hat.tolist(),
            'y'    : y    .tolist(),
            'mae'  : mae },
            filename_result)
