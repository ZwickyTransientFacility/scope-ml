import datetime
import os
import pathlib
import tensorflow as tf
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from .utils import make_confusion_matrix, plot_roc, plot_pr
import numpy as np
import wandb
import json

from .models import AbstractClassifier


class DenseBlock(tf.keras.models.Model):
    def __init__(
        self, units: int, activation: str = "relu", repetitions: int = 1, **kwargs
    ):
        """Dense block to process light curve features

        :param units: tuple of size 2 with number of units in repeated dense blocks
        :param activation:
        :param repetitions:
        """
        super(DenseBlock, self).__init__(**kwargs)
        self.units = units
        self.repetitions = repetitions
        self.activation = activation

        for i in range(self.repetitions):
            vars(self)[f"dense_{i}"] = tf.keras.layers.Dense(
                units=self.units, activation=self.activation
            )

    def call(self, inputs, **kwargs):
        x = self.dense_0(inputs)
        for i in range(1, self.repetitions):
            x = vars(self).get(f"dense_{i}")(x)

        return x


class ConvBlock(tf.keras.models.Model):
    def __init__(
        self,
        filters: int,
        kernel_size: tuple,
        activation: str = "relu",
        pool_size: tuple = (2, 2),
        repetitions: int = 1,
        **kwargs,
    ):
        """Convolutional block to process dmdt's constructed from light curves

        :param filters:
        :param kernel_size:
        :param activation:
        :param pool_size:
        :param repetitions:
        :param kwargs:
        """
        super(ConvBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool_size = pool_size
        self.repetitions = repetitions

        for i in range(self.repetitions):
            vars(self)[f"conv_{i}"] = tf.keras.layers.SeparableConv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation=self.activation,
            )

        self.max_pool = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size)

    def call(self, inputs, **kwargs):
        x = self.conv_0(inputs)
        for i in range(1, self.repetitions):
            x = vars(self).get(f"conv_{i}")(x)
        x = self.max_pool(x)

        return x


class ScopeNet(tf.keras.models.Model):
    def __init__(
        self,
        dense_branch: bool = True,
        conv_branch: bool = True,
        dropout_rate: float = 0.25,
        **kwargs,
    ):
        """Deep Neural Net architecture for the ZTF Source Classification project

        See https://arxiv.org/pdf/2102.11304.pdf for the details.

        :param dense_branch: include dense branch to process features?
        :param conv_branch: include convolutional branch to process dmdt's?
        :param dropout_rate: rate to use in dropout layers
        :param kwargs:
        """
        super(ScopeNet, self).__init__(**kwargs)

        if (not dense_branch) and (not conv_branch):
            raise ValueError("Model must have at least one branch")

        self.dense_branch = dense_branch
        self.conv_branch = conv_branch
        self.dropout_rate = dropout_rate

        self.dropout_0 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout_1 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.dense_0 = DenseBlock(units=256, repetitions=1)
        self.dense_1 = DenseBlock(units=32, repetitions=1)
        self.dense_2 = DenseBlock(units=16, repetitions=1)

        self.conv_0 = ConvBlock(filters=16, kernel_size=(3, 3), pool_size=(2, 2))
        self.conv_1 = ConvBlock(filters=32, kernel_size=(3, 3), pool_size=(2, 2))

        self.dense_out = tf.keras.layers.Dense(
            units=1, activation="sigmoid", name="score"
        )

    def call(self, inputs, **kwargs):
        features_input = inputs.get("features")
        dmdt_input = inputs.get("dmdt")

        # dense branch to digest features
        if self.dense_branch:
            x_dense = self.dense_0(features_input)
            x_dense = self.dropout_0(x_dense)
            x_dense = self.dense_1(x_dense)

        # CNN branch to digest dmdt
        if self.conv_branch:
            x_conv = self.conv_0(dmdt_input)
            x_conv = self.dropout_1(x_conv)
            x_conv = self.conv_1(x_conv)
            x_conv = self.dropout_2(x_conv)

            x_conv = self.global_average_pool(x_conv)

        # concatenate
        if self.dense_branch and self.conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif self.dense_branch:
            x = x_dense
        elif self.conv_branch:
            x = x_conv

        # one more dense layer
        x = self.dense_2(x)

        # Logistic regression to output the final score
        x = self.dense_out(x)

        return x

    def summary(self, **kwargs):

        features = tf.keras.layers.Input(shape=(40,))
        dmdt = tf.keras.layers.Input(shape=(26, 26, 1))
        model = tf.keras.models.Model(
            inputs=[features, dmdt],
            outputs=self.call({"features": features, "dmdt": dmdt}),
        )
        return model.summary()


class DNN(AbstractClassifier):
    """Baseline model with a statically-defined architecture"""

    def setup(
        self,
        dense_branch: bool = True,
        conv_branch: bool = True,
        loss: str = "binary_crossentropy",
        optimizer: str = "adam",
        callbacks: tuple = ("reduce_lr_on_plateau", "early_stopping"),
        tag: Optional[str] = None,
        logdir: str = "logs",
        **kwargs,
    ):

        tf.keras.backend.clear_session()

        self.model = self.build_model(
            dense_branch=dense_branch, conv_branch=conv_branch, **kwargs
        )

        self.meta["loss"] = loss
        if optimizer == "adam":
            lr = kwargs.get("lr", 3e-4)
            beta_1 = kwargs.get("beta_1", 0.9)
            beta_2 = kwargs.get("beta_2", 0.999)
            epsilon = kwargs.get("epsilon", 1e-7)  # None?
            decay = kwargs.get("decay", 0.0)
            amsgrad = kwargs.get("amsgrad", 3e-4)
            self.meta["optimizer"] = tf.keras.optimizers.legacy.Adam(
                learning_rate=lr,
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                decay=decay,
                amsgrad=amsgrad,
            )
        elif optimizer == "sgd":
            lr = kwargs.get("lr", 3e-4)
            momentum = kwargs.get("momentum", 0.9)
            decay = kwargs.get("epsilon", 1e-6)
            nesterov = kwargs.get("nesterov", True)
            self.meta["optimizer"] = tf.keras.optimizers.SGD(
                learning_rate=lr, momentum=momentum, decay=decay, nesterov=nesterov
            )
        else:
            print("Could not recognize optimizer, using Adam with default params")
            self.meta["optimizer"] = tf.keras.optimizers.Adam(
                learning_rate=3e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                decay=0.0,
                amsgrad=False,
            )

        self.meta["metrics"] = [
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

        self.set_callbacks(callbacks, tag, **kwargs)

        run_eagerly = kwargs.get("run_eagerly", False)
        self.model.compile(
            optimizer=self.meta["optimizer"],
            loss=self.meta["loss"],
            metrics=self.meta["metrics"],
            run_eagerly=run_eagerly,
        )

    @staticmethod
    def build_model(
        dense_branch: bool = True,
        conv_branch: bool = True,
        **kwargs,
    ):

        # fixme: subclassed tf.keras.models.Model is misbehaving, need to investigate
        # m = ScopeNet(dense_branch=dense_branch, conv_branch=conv_branch, **kwargs)

        # fixme: for now, simply use Keras' Functional API
        if (not dense_branch) and (not conv_branch):
            raise ValueError("model must have at least one branch")

        features_input = tf.keras.Input(
            shape=kwargs.get("features_input_shape", (40,)), name="features"
        )
        dmdt_input = tf.keras.Input(
            shape=kwargs.get("dmdt_input_shape", (26, 26, 1)), name="dmdt"
        )

        # dense branch to digest features
        if dense_branch:
            x_dense = tf.keras.layers.Dense(256, activation="relu", name="dense_fc_1")(
                features_input
            )
            x_dense = tf.keras.layers.Dropout(0.25)(x_dense)
            x_dense = tf.keras.layers.Dense(32, activation="relu", name="dense_fc_2")(
                x_dense
            )

        # CNN branch to digest dmdt
        if conv_branch:
            x_conv = tf.keras.layers.SeparableConv2D(
                16, (3, 3), activation="relu", name="conv_conv_1"
            )(dmdt_input)
            # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(
                16, (3, 3), activation="relu", name="conv_conv_2"
            )(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
            x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            x_conv = tf.keras.layers.SeparableConv2D(
                32, (3, 3), activation="relu", name="conv_conv_3"
            )(x_conv)
            # x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
            x_conv = tf.keras.layers.SeparableConv2D(
                32, (3, 3), activation="relu", name="conv_conv_4"
            )(x_conv)
            x_conv = tf.keras.layers.Dropout(0.25)(x_conv)
            # x_conv = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x_conv)

            x_conv = tf.keras.layers.GlobalAveragePooling2D()(x_conv)

        # concatenate
        if dense_branch and conv_branch:
            x = tf.keras.layers.concatenate([x_dense, x_conv])
        elif dense_branch:
            x = x_dense
        elif conv_branch:
            x = x_conv
        x = tf.keras.layers.Dropout(0.4)(x)

        # one more dense layer?
        x = tf.keras.layers.Dense(16, activation="relu", name="fc_1")(x)

        # Logistic regression to output the final score
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)

        m = tf.keras.Model(inputs=[features_input, dmdt_input], outputs=x)

        return m

    def set_callbacks(self, callbacks, tag=None, **kwargs):
        self.meta["callbacks"] = []
        for callback in set(callbacks):
            if callback == "early_stopping":
                # halt training if no gain in <validation loss> over <patience> epochs
                monitor = kwargs.get("monitor", "val_loss")
                patience = kwargs.get("patience", 10)
                restore_best_weights = kwargs.get("restore_best_weights", True)
                early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    restore_best_weights=restore_best_weights,
                )
                self.meta["callbacks"].append(early_stopping_callback)

            elif callback == "tensorboard":
                # logs for TensorBoard:
                if tag:
                    log_tag = f'{self.name.replace(" ", "_")}-{tag}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                else:
                    log_tag = f'{self.name.replace(" ", "_")}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
                logdir_tag = os.path.join("logs", log_tag)
                tensorboard_callback = tf.keras.callbacks.TensorBoard(
                    os.path.join(logdir_tag, log_tag), histogram_freq=1
                )
                self.meta["callbacks"].append(tensorboard_callback)

            elif callback == "reduce_lr_on_plateau":
                monitor = kwargs.get("monitor", "val_loss")
                patience = kwargs.get("patience", 10)
                lr_reduction_factor = kwargs.get("lr_reduction_factor", 0.1)
                reduce_lr_on_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor,
                    factor=lr_reduction_factor,
                    patience=patience,
                    verbose=0,
                    mode="auto",
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=0,
                )
                self.meta["callbacks"].append(reduce_lr_on_plateau_callback)

    def assign_datasets(
        self,
        features_input_shape,
        train_dataset_repeat,
        val_dataset_repeat,
        steps_per_epoch_train,
        steps_per_epoch_val,
        train_dataset,
        val_dataset,
        wandb_token,
    ):
        self.meta["features_input_shape"] = features_input_shape
        self.meta["train_dataset_repeat"] = train_dataset_repeat
        self.meta["val_dataset_repeat"] = val_dataset_repeat
        self.meta["steps_per_epoch_train"] = steps_per_epoch_train
        self.meta["steps_per_epoch_val"] = steps_per_epoch_val
        self.meta["train_dataset"] = train_dataset
        self.meta["val_dataset"] = val_dataset

        wandb.login(key=wandb_token)

    def sweep(
        self,
    ):
        wandb.init(
            job_type="sweep",
        )

        wandb_epochs = wandb.config.epochs
        wandb_dense_branch = wandb.config.dense_branch
        wandb_conv_branch = wandb.config.conv_branch
        wandb_loss = wandb.config.loss
        wandb_optimizer = wandb.config.optimizer
        wandb_lr = wandb.config.lr
        wandb_momentum = wandb.config.momentum
        wandb_monitor = wandb.config.monitor
        wandb_patience = wandb.config.patience
        wandb_callbacks = wandb.config.callbacks
        wandb_run_eagerly = wandb.config.run_eagerly
        wandb_beta_1 = wandb.config.beta_1
        wandb_beta_2 = wandb.config.beta_2
        wandb_epsilon = wandb.config.epsilon
        wandb_amsgrad = wandb.config.amsgrad
        wandb_decay = wandb.config.decay

        self.setup(
            features_input_shape=self.meta["features_input_shape"],
            dense_branch=wandb_dense_branch,
            conv_branch=wandb_conv_branch,
            dmdt_input_shape=(26, 26, 1),
            loss=wandb_loss,
            optimizer=wandb_optimizer,
            momentum=wandb_momentum,
            monitor=wandb_monitor,
            patience=wandb_patience,
            callbacks=wandb_callbacks,
            run_eagerly=wandb_run_eagerly,
            learning_rate=wandb_lr,
            beta_1=wandb_beta_1,
            beta_2=wandb_beta_2,
            epsilon=wandb_epsilon,
            amsgrad=wandb_amsgrad,
            decay=wandb_decay,
        )

        self.train(
            train_dataset=self.meta["train_dataset_repeat"],
            val_dataset=self.meta["val_dataset_repeat"],
            steps_per_epoch_train=self.meta["steps_per_epoch_train"],
            steps_per_epoch_val=self.meta["steps_per_epoch_val"],
            epochs=wandb_epochs,
        )

        stats_train = self.evaluate(self.meta["train_dataset"], name="train", verbose=0)
        stats_val = self.evaluate(self.meta["val_dataset"], name="val", verbose=0)

        wandb.log(
            {
                "dense_branch": wandb_dense_branch,
                "conv_branch": wandb_conv_branch,
                "loss": wandb_loss,
                "optimizer": wandb_optimizer,
                "lr": wandb_lr,
                "momentum": wandb_momentum,
                "monitor": wandb_monitor,
                "patience": wandb_patience,
                "callbacks": wandb_callbacks,
                "run_eagerly": wandb_run_eagerly,
                "beta_1": wandb_beta_1,
                "beta_2": wandb_beta_2,
                "epsilon": wandb_epsilon,
                "amsgrad": wandb_amsgrad,
                "decay": wandb_decay,
                "epochs": wandb_epochs,
                "train_loss": stats_train[0],
                "val_loss": stats_val[0],
            }
        )

    def train(
        self,
        train_dataset,
        val_dataset,
        steps_per_epoch_train,
        steps_per_epoch_val,
        epochs=300,
        class_weight=None,
        verbose=0,
    ):

        if class_weight is None:
            # all our problems here are binary classification ones:
            class_weight = {i: 1 for i in range(2)}

        self.meta["history"] = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch_train,
            validation_data=val_dataset,
            validation_steps=steps_per_epoch_val,
            class_weight=class_weight,
            callbacks=self.meta["callbacks"],
            verbose=verbose,
        )

    def evaluate(self, eval_dataset, name="test", **kwargs):
        y_eval = np.concatenate([y for _, y in eval_dataset], axis=0)
        y_pred = np.around(self.predict(eval_dataset, name=f"_{name}", **kwargs))

        self.meta[f"y_{name}"] = y_eval

        # Generate confusion matrix
        self.meta[f"cm_{name}"] = confusion_matrix(y_eval, y_pred, normalize="all")

        return self.model.evaluate(eval_dataset, **kwargs)

    def predict(self, X, name=None, **kwargs):
        y_pred = self.model.predict(X)

        if name is not None:
            self.meta[f"y_pred{name}"] = y_pred
        else:
            self.meta["y_pred"] = y_pred

        return y_pred

    def load(self, path_model, weights_only: bool = False, **kwargs):
        # Original functionality
        if weights_only:
            self.model.load_weights(path_model, **kwargs)
        # New functionality to load whole model from HDF5 file
        else:
            self.model = tf.keras.models.load_model(path_model, **kwargs)

    def save(
        self,
        tag: str,
        output_path: str = "./",
        output_format: str = "h5",
        plot: bool = False,
        names: list = ["train", "val", "test"],
        cm_include_count: bool = False,
        cm_include_percent: bool = True,
        annotate_scores: bool = False,
        **kwargs,
    ):

        if output_format not in ("h5",):
            raise ValueError("unknown output format")

        output_path = pathlib.Path(output_path)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)

        output_name = self.name if not tag else tag
        if not output_name.endswith(".h5"):
            output_name += ".h5"
        self.model.save(output_path / output_name, save_format=output_format)

        stats_dct = {}

        # Save diagnostic plots
        for name in names:
            if plot:
                path = output_path / f"{tag}_plots" / name
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                cmpdf = tag + "_cm.pdf"
                recallpdf = tag + "_recall.pdf"
                rocpdf = tag + "_roc.pdf"
                stats_json = tag + "_stats.json"

                if self.meta[f"cm_{name}"] is not None:
                    cname = tag.split(".")[0]
                    accuracy, precision, recall, f1_score = make_confusion_matrix(
                        self.meta[f"cm_{name}"],
                        figsize=(8, 6),
                        cbar=False,
                        count=cm_include_count,
                        percent=cm_include_percent,
                        categories=["not " + cname, cname],
                        annotate_scores=annotate_scores,
                    )
                    stats_dct["accuracy"] = accuracy
                    stats_dct["precision"] = precision
                    stats_dct["recall"] = recall
                    stats_dct["f1_score"] = f1_score
                    sns.set_context("talk")
                    plt.title(cname)
                    plt.savefig(path / cmpdf, bbox_inches="tight")
                    plt.close()

                y_compare = self.meta.get(f"y_{name}", None)
                y_pred = self.meta.get(f"y_pred_{name}", None)

                if (y_compare is not None) & (y_pred is not None):

                    fpr, tpr, _ = roc_curve(y_compare, y_pred)
                    roc_auc = auc(fpr, tpr)
                    precision, recall, _ = precision_recall_curve(y_compare, y_pred)
                    stats_dct["roc_auc"] = roc_auc

                    plot_roc(fpr, tpr, roc_auc)
                    plt.savefig(path / rocpdf, bbox_inches="tight")
                    plt.close()

                    plot_pr(recall, precision)
                    plt.savefig(path / recallpdf, bbox_inches="tight")
                    plt.close()

                with open(path / stats_json, "w") as f:
                    json.dump(stats_dct, f)
