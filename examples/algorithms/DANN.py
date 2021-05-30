import torch
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.layers import BranchedModules, RevGradLayer
from wilds.common.utils import split_into_groups

class DANN(SingleModelAlgorithm):
    """
    Domain Adversarial Neural Networks.
    For binary domain

    Original paper: https://arxiv.org/abs/1505.07818
    """
    def __init__(self, config, d_out, grouper, loss, domain_loss, metric, n_train_steps):
        # check config
        assert config.train_loader == 'group'
        assert config.uniform_over_groups
        assert config.distinct_groups
        # initialize models
        featurizer, label_classifier = initialize_model(config, d_out=d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        label_classifier = label_classifier.to(config.device)
        reverse_gradient = RevGradLayer().to(config.device)
        if 1:
            domain_classifier = torch.nn.Sequential(reverse_gradient, torch.nn.Linear(in_features=featurizer.d_out, out_features=32),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(in_features=32, out_features=1)).to(config.device)
        else:
            domain_classifier = torch.nn.Sequential(reverse_gradient, torch.nn.Linear(in_features=featurizer.d_out, out_features=1)).to(config.device)


        classifier = BranchedModules(['lc', 'dc'], {'lc': label_classifier, 'dc': domain_classifier}, cat_dim=-1).to(config.device)

        model = torch.nn.Sequential(featurizer, classifier).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.domain_loss = domain_loss
        # algorithm hyperparameters
        self.dann_lambda = config.dann_lambda
        # additional logging
        self.logged_fields.append('domain_classifier_loss')
        # set model components
        self.featurizer = featurizer
        self.label_classifier = label_classifier
        self.domain_classifier = domain_classifier
        self.classifier = classifier

    def process_batch(self, batch):
        """
        Override
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        d = metadata[:, -1].to(self.device)  # domain
        features = self.featurizer(x)
        label_outputs = self.label_classifier(features)
        domain_outputs = self.domain_classifier(features)

        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': label_outputs,
            'd': d,
            'd_pred': domain_outputs,
            'metadata': metadata,
            'features': features,
            }
        return results

    def objective(self, results):
        d = results['d']
        y_pred_train = results['y_pred'][d == 0]
        y_true_train = results['y_true'][d == 0]

        # classifier loss on training groups
        avg_loss = self.loss.compute(y_pred_train, y_true_train, return_dict=False)

        # domain loss on all groups
        dloss = self.domain_loss.compute(results['d_pred'], d[:, None], return_dict=False)

        results['domain_classifier_loss'] = dloss.item()

        return avg_loss - self.dann_lambda * dloss
