from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["train"]
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

        # optimizing discriminator
        gen_audio = self.model(**batch)
        batch.update(gen_audio)

        disc_results = self.model.discriminate(detach=True, **batch)
        batch.update(disc_results)

        disc_loss = self.d_loss(**batch)
        batch.update(disc_loss)
        batch['disc_loss'].backward()
        self._clip_grad_norm("discriminator")
        self.d_optimizer.step()

        # optimizing generator
        disc_results = self.model.discriminate(**batch)
        batch.update(disc_results)

        melspec_transform = self.batch_transforms["train"]["gt_melspec"]
        batch["gen_melspec"] = melspec_transform(batch["gen_audio"].squeeze(1))

        gen_loss = self.g_loss(**batch)
        batch.update(gen_loss)
        batch['gen_loss'].backward()
        self._clip_grad_norm("generator")
        self.g_optimizer.step()

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        self._log_spectrogram(**batch)
        self._log_audio(**batch)
    
    def _log_spectrogram(self, gt_melspec, gen_melspec, **batch):
        specs_for_plot = [
            gt_melspec[0].detach().cpu(),
            gen_melspec[0].detach().cpu(),
        ]
        names = ["gt_melspec", "gen_melspec"]

        for spec_for_lot, name in zip(specs_for_plot, names):
            image = plot_spectrogram(spec_for_lot)
            self.writer.add_image(name, image)
        
    def _log_audio(self, **batch):
        num_audio = 3
        self.writer.add_tf_audio_table(num_audio=num_audio, **batch)
